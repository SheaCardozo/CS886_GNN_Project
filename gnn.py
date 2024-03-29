import numpy as np
import torch
import dgl
import time

from modules import *
from utils import *
from dag_utils import *
from metrics import *
from relation_header import FJMPRelationHeader

import horovod.torch as hvd 
from mpi4py import MPI

comm = MPI.COMM_WORLD
dev = 'cuda:{}'.format(0)

class GNN(torch.nn.Module):
    def __init__(self, config):
        super(GNN, self).__init__()
        self.config = config
        self.dataset = config["dataset"]
        self.num_train_samples = config["num_train_samples"]
        self.num_val_samples = config["num_val_samples"]
        self.num_agenttypes = config["num_agenttypes"]
        self.switch_lr_1 = config["switch_lr_1"]
        self.switch_lr_2 = config["switch_lr_2"]
        self.lr_step = config["lr_step"]
        self.mode = config["mode"]
        self.input_size = config["input_size"]
        self.observation_steps = config["observation_steps"]
        self.prediction_steps = config["prediction_steps"]
        self.num_edge_types = config["num_edge_types"]
        self.h_dim = config["h_dim"]
        self.num_joint_modes = config["num_joint_modes"]
        self.num_proposals = config["num_proposals"]
        self.learning_rate = config["lr"]
        self.max_epochs = config["max_epochs"]
        self.log_path = config["log_path"]
        self.batch_size = config["batch_size"]
        self.decoder = config["decoder"]
        self.num_heads = config["num_heads"]
        self.learned_relation_header = config["learned_relation_header"]
        self.resume_training = config["resume_training"]
        self.proposal_coef = config["proposal_coef"]
        self.rel_coef = config["rel_coef"]
        self.proposal_header = config["proposal_header"]
        self.two_stage_training = config["two_stage_training"]
        self.training_stage = config["training_stage"]
        self.ig = config["ig"]
        self.focal_loss = config["focal_loss"]
        self.gamma = config["gamma"]
        self.weight_0 = config["weight_0"]
        self.weight_1 = config["weight_1"]
        self.weight_2 = config["weight_2"]
        self.teacher_forcing = config["teacher_forcing"]
        self.scheduled_sampling = config["scheduled_sampling"]
        self.eval_training = config["eval_training"]
        self.supervise_vehicles = config["supervise_vehicles"]
        self.no_agenttype_encoder = config["no_agenttype_encoder"]
        self.train_all = config["train_all"]
        
        if self.two_stage_training and self.training_stage == 2:
            self.pretrained_relation_header = None
        
        self.build()

    def build(self):

        self.feature_encoder = FJMPFeatureEncoder(self.config).to(dev)
        
        self.relation_header = FJMPRelationHeader(self.config).to(dev)
        
        if self.proposal_header:
            self.proposal_decoder = FJMPTrajectoryProposalDecoder(self.config).to(dev)
        
        if (self.two_stage_training and self.training_stage == 2) or not self.two_stage_training:
            self.trajectory_decoder = LaneGCNHeader(self.config).to(dev)

    def _train(self, train_loader, val_loader, optimizer, start_epoch, val_best, ade_best, fde_best, val_edge_acc_best):        
        hvd.broadcast_parameters(self.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        for epoch in range(start_epoch, self.max_epochs + 1):   
            # this shuffles the training set every epoch         
            train_loader.sampler.set_epoch(int(epoch))
            
            t_start_epoch = time.time()
            self.train()
            tot_log = self.num_train_samples // (self.batch_size * hvd.size())  

            results = {}
            loc_preds, gt_locs_all, batch_idxs_all, has_preds_all, has_last_all = [], [], [], [], []
            gt_psirads_all, shapes_all, agenttypes_all, gt_ctrs_all, gt_psirads_all, feat_psirads_all, gt_vels_all, theta_all = [], [], [], [], [], [], [], []
            is_scored_all = []

            if self.proposal_header:
                proposals_all = []
            
            if self.learned_relation_header:
                ig_preds = []
                ig_labels_all = [] 
            
            if self.scheduled_sampling:
                prop_ground_truth = 1 - (epoch - 1) / (self.max_epochs - 1)   
            elif self.teacher_forcing:
                prop_ground_truth = 1.  
            else:
                prop_ground_truth = 0. 
            
            # set learning rate accordingly
            for e, param_group in enumerate(optimizer.param_groups):
                if epoch == self.switch_lr_1 or epoch == self.switch_lr_2:
                    param_group["lr"] = param_group["lr"] * (self.lr_step)
                
                if e == 0:
                    cur_lr = param_group["lr"]  
            
            tot = 0
            accum_gradients = {}
            for i, data in enumerate(train_loader):      

                # get data dictionary for processing batch
                dd = process_data(data, self.config)

                dgl_graph = self.init_dgl_graph(dd['batch_idxs'], dd['ctrs'], dd['orig'], dd['rot'], dd["agenttypes"], dd['world_locs'], dd['has_preds']).to(dev)
                # only process observed features
                dgl_graph = self.feature_encoder(dgl_graph, dd['feats'][:,:self.observation_steps], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])

                if self.two_stage_training and self.training_stage == 2:
                    stage_1_graph = self.build_stage_1_graph(dgl_graph, dd['feats'][:,:self.observation_steps], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])
                else:
                    stage_1_graph = None

                ig_dict = {}
                ig_dict["ig_labels"] = dd["ig_labels"] 
                
                # produces dictionary of results
                res = self.forward(dd["scene_idxs"], dgl_graph, stage_1_graph, ig_dict, dd['batch_idxs'], dd["batch_idxs_edges"], dd["actor_ctrs"], prop_ground_truth=prop_ground_truth, eval=False)

                loss_dict = self.get_loss(dgl_graph, dd['batch_idxs'], res, dd['agenttypes'], dd['has_preds'], dd['gt_locs'], dd['batch_size'], dd["ig_labels"], epoch)
                
                loss = loss_dict["total_loss"]
                optimizer.zero_grad()
                loss.backward()
                accum_gradients = accumulate_gradients(accum_gradients, self.named_parameters())
                optimizer.step()
                
                if i % 100 == 0:
                    print_("Training data: ", "{:.2f}%".format(i * 100 / tot_log), "lr={:.3e}".format(cur_lr), "rel_coef={:.1f}".format(self.rel_coef),
                        "\t".join([k + ":" + f"{v.item():.3f}" for k, v in loss_dict.items()]))

                if self.eval_training:
                    if self.proposal_header:
                        proposals_all.append(res["proposals"].detach().cpu())
                    
                    if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):                    
                        loc_preds.append(res["loc_pred"].detach().cpu())
                    
                    if self.learned_relation_header:
                        ig_preds.append(res["edge_probs"].detach().cpu())
                        ig_labels_all.append(dd["ig_labels"].detach().cpu())                    

                    gt_locs_all.append(dd['gt_locs'].detach().cpu())
                    batch_idxs_all.append(dd['batch_idxs'].detach().cpu() + tot)
                    has_preds_all.append(dd['has_preds'].detach().cpu())
                    has_last_all.append(dd['has_last'].detach().cpu())
                    gt_psirads_all.append(dd['gt_psirads'].detach().cpu())
                    feat_psirads_all.append(dd['feat_psirads'].detach().cpu())
                    gt_vels_all.append(dd['gt_vels'].detach().cpu())
                    theta_all.append(dd['theta'].detach().cpu())

                    if self.dataset == "argoverse2":
                        is_scored_all.append(dd['is_scored'].detach().cpu())
                    if self.dataset == "interaction":
                        shapes_all.append(dd['shapes'][:,0,:].detach().cpu())
                    
                    agenttypes_all.append(dd['agenttypes'].detach().cpu())  
                    # map back to gt coordinate system              
                    gt_ctrs_all.append((torch.matmul(dd['ctrs'].unsqueeze(1), dd["rot"]).squeeze(1) + dd['orig']).detach().cpu())

                tot += dd['batch_size']
            
            for key in accum_gradients.keys():
                accum_gradients[key] /= i

            # plot gradient norms
            accum_gradients = sync3(accum_gradients, comm)
            if hvd.rank() == 0:
                plot_grad_flow(accum_gradients, epoch, self.log_path)

            if self.eval_training:
                self.eval()
                print_('Calculating training metrics...')

                has_last_mask = np.concatenate(has_last_all, axis=0).astype(bool)
                if self.dataset=='interaction':
                    eval_agent_mask = np.concatenate(agenttypes_all, axis=0)[:, 1].astype(bool)
                else:
                    eval_agent_mask = np.ones(np.concatenate(agenttypes_all, axis=0)[:, 1].shape).astype(bool)

                init_mask = has_last_mask * eval_agent_mask
                results['gt_locs_all'] = np.concatenate(gt_locs_all, axis=0)
                results['has_preds_all'] = np.concatenate(has_preds_all, axis=0)
                results['batch_idxs'] = np.concatenate(batch_idxs_all)       
                results['gt_psirads_all'] = np.concatenate(gt_psirads_all, axis=0)
                results['feat_psirads_all'] = np.concatenate(feat_psirads_all, axis=0)
                results['gt_vels_all'] = np.concatenate(gt_vels_all, axis=0)
                results['theta_all'] = np.concatenate(theta_all, axis=0)
                results['gt_ctrs_all'] = np.concatenate(gt_ctrs_all, axis=0)
                if self.dataset == "interaction":
                    results['shapes_all'] = np.concatenate(shapes_all, axis=0) 
                if self.proposal_header:
                    results["proposals_all"] = np.concatenate(proposals_all, axis=0)
                if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
                    results['loc_pred'] = np.concatenate(loc_preds, axis=0)    
                if self.learned_relation_header:
                    results['ig_preds'] = np.concatenate(ig_preds, axis=0)
                    results["ig_labels_all"] = np.concatenate(ig_labels_all, axis=0)   

                mask = init_mask
            
                eval_results = calc_metrics(results, self.config, mask, identifier='reg')
                eval_results = sync(eval_results, self.config, comm)
                
                print_("Epoch {} training-set results: ".format(epoch),
                      "\t".join([f"{k}: {v}" if type(v) is np.ndarray else f"{k}: {v:.3f}" for k, v in eval_results.items()]))
            
            if self.train_all:
                self.eval()
                if hvd.rank() == 0:
                    if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
                        print_("Saving model")
                        self.save(epoch, optimizer, val_best, ade_best, fde_best)
                    else:
                        print_("Saving relation header")
                        self.save_relation_header(epoch, optimizer, val_edge_acc_best) 

            else:
                self.eval()
                
                val_eval_results = self._eval(val_loader, epoch)
                
                print_("Epoch {} validation-set results: ".format(epoch), "\t".join([f"{k}: {v}" if type(v) is np.ndarray else f"{k}: {v:.3f}" for k, v in val_eval_results.items()]))
                
                # Best model is one with minimum FDE + ADE
                if hvd.rank() == 0:
                    if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
                        if (val_eval_results["FDE"] + val_eval_results["ADE"]) < val_best:
                            val_best = val_eval_results["FDE"] + val_eval_results["ADE"]
                            ade_best = val_eval_results["ADE"]
                            fde_best = val_eval_results["FDE"]
                            self.save(epoch, optimizer, val_best, ade_best, fde_best)
                            print_("Validation FDE+ADE improved. Saving model. ")
                        print_("Best loss: {:.4f}".format(val_best), "Best ADE: {:.3f}".format(ade_best), "Best FDE: {:.3f}".format(fde_best))

                    else:
                        if val_eval_results["E-Acc"] > val_edge_acc_best:
                            print_("Validation Edge Accuracy improved.")  
                            val_edge_acc_best = val_eval_results["E-Acc"]  
                        
                        print_("Saving relation header")
                        self.save_relation_header(epoch, optimizer, val_edge_acc_best)    
                        print_("Best validation edge accuracy: {:.4f}".format(val_edge_acc_best))                
                
                    # save the current epoch
                    if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
                        self.save_current_epoch(epoch, optimizer, val_best, ade_best, fde_best)
            
            print_("Epoch {} time: {:.3f}s".format(epoch, time.time() - t_start_epoch))

    def _eval(self, val_loader, epoch):
        hvd.broadcast_parameters(self.state_dict(), root_rank=0)

        self.eval()
        # validation results
        results = {}
        loc_preds, gt_locs_all, batch_idxs_all, scene_idxs_all, has_preds_all, has_last_all = [], [], [], [], [], []
        feat_psirads_all, gt_psirads_all, gt_vels_all, shapes_all, agenttypes_all, gt_ctrs_all, is_connected_all = [], [], [], [], [], [], []
        theta_all = []
        is_scored_all = []

        if self.proposal_header:
            proposals_all = []
        
        if self.learned_relation_header:
            ig_preds = []
            ig_labels_all = []            

        tot = 0
        with torch.no_grad():
            tot_log = self.num_val_samples // (self.batch_size * hvd.size())            
            for i, data in enumerate(val_loader):
                dd = process_data(data, self.config)
                
                dgl_graph = self.init_dgl_graph(dd['batch_idxs'], dd['ctrs'], dd['orig'], dd['rot'], dd['agenttypes'], dd['world_locs'], dd['has_preds']).to(dev)
                dgl_graph = self.feature_encoder(dgl_graph, dd['feats'][:,:self.observation_steps], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])

                if self.two_stage_training and self.training_stage == 2:
                    stage_1_graph = self.build_stage_1_graph(dgl_graph, dd['feats'][:,:self.observation_steps], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])
                else:
                    stage_1_graph = None

                ig_dict = {}
                ig_dict["ig_labels"] = dd["ig_labels"]
                
                res = self.forward(dd["scene_idxs"], dgl_graph, stage_1_graph, ig_dict, dd['batch_idxs'], dd["batch_idxs_edges"], dd["actor_ctrs"], prop_ground_truth=0., eval=True)

                loss_dict = self.get_loss(dgl_graph, dd['batch_idxs'], res, dd['agenttypes'], dd['has_preds'], dd['gt_locs'], dd['batch_size'], dd["ig_labels"], epoch)
                
                if i % 50 == 0:
                    print_("Validation data: ", "{:.2f}%".format(i * 100 / tot_log), "\t".join([k + ":" + f"{v.item():.3f}" for k, v in loss_dict.items()]))

                if self.proposal_header:
                    proposals_all.append(res["proposals"].detach().cpu())
                
                if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
                    loc_preds.append(res["loc_pred"].detach().cpu())
                is_connected_all.append(dd["is_connected"].detach().cpu())

                if self.learned_relation_header:
                    ig_preds.append(res["edge_probs"].detach().cpu())
                    ig_labels_all.append(dd["ig_labels"].detach().cpu())                                            

                gt_locs_all.append(dd['gt_locs'].detach().cpu())
                batch_idxs_all.append(dd['batch_idxs'].detach().cpu() + tot)
                scene_idxs_all.append(dd['scene_idxs'].detach().cpu())
                has_preds_all.append(dd['has_preds'].detach().cpu())
                has_last_all.append(dd['has_last'].detach().cpu())
                gt_psirads_all.append(dd['gt_psirads'].detach().cpu())
                feat_psirads_all.append(dd['feat_psirads'].detach().cpu())
                gt_vels_all.append(dd['gt_vels'].detach().cpu())
                theta_all.append(dd['theta'].detach().cpu())
                
                if self.dataset == "argoverse2":
                    is_scored_all.append(dd['is_scored'].detach().cpu())
                
                if self.dataset == "interaction":
                    shapes_all.append(dd['shapes'][:,0,:].detach().cpu())
                
                agenttypes_all.append(dd['agenttypes'].detach().cpu())
                gt_ctrs_all.append((torch.matmul(dd['ctrs'].unsqueeze(1), dd["rot"]).squeeze(1) + dd['orig']).detach().cpu())
                tot += dd['batch_size']

        print_('Calculating validation metrics...')
        
        has_last_mask = np.concatenate(has_last_all, axis=0).astype(bool)
        if self.dataset=='interaction':
            # only evaluate vehicles
            eval_agent_mask = np.concatenate(agenttypes_all, axis=0)[:, 1].astype(bool)
        else:
            # evaluate all context agents
            eval_agent_mask = np.ones(np.concatenate(agenttypes_all, axis=0)[:, 1].shape).astype(bool)

        init_mask = has_last_mask * eval_agent_mask
        results['gt_locs_all'] = np.concatenate(gt_locs_all, axis=0)
        results['has_preds_all'] = np.concatenate(has_preds_all, axis=0)
        results['batch_idxs'] = np.concatenate(batch_idxs_all)       
        results['gt_psirads_all'] = np.concatenate(gt_psirads_all, axis=0)
        results['feat_psirads_all'] = np.concatenate(feat_psirads_all, axis=0)
        results['gt_vels_all'] = np.concatenate(gt_vels_all, axis=0)
        results['theta_all'] = np.concatenate(theta_all, axis=0)
        results['gt_ctrs_all'] = np.concatenate(gt_ctrs_all, axis=0)
        if self.dataset == "interaction":
            results['shapes_all'] = np.concatenate(shapes_all, axis=0)
        if self.proposal_header:
            results["proposals_all"] = np.concatenate(proposals_all, axis=0)
        if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
            results['loc_pred'] = np.concatenate(loc_preds, axis=0)    
        if self.learned_relation_header:
            results['ig_preds'] = np.concatenate(ig_preds, axis=0)
            results["ig_labels_all"] = np.concatenate(ig_labels_all, axis=0)    
        
        if self.mode == 'train':
            mask = init_mask
            
            all_val_eval_results = calc_metrics(results, self.config, mask, identifier='reg')
            all_val_eval_results = sync(all_val_eval_results, self.config, comm)
        else:      
            all_val_eval_results = {}
            
            mask = init_mask
            
            ### REGULAR FDE/ADE
            val_eval_results = calc_metrics(results, self.config, mask, identifier='reg')
            val_eval_results = sync(val_eval_results, self.config, comm)

            if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
                all_val_eval_results['FDE'] = val_eval_results['FDE']
                all_val_eval_results['ADE'] = val_eval_results['ADE']
                all_val_eval_results['SMR'] = val_eval_results['SMR']
                all_val_eval_results['SMR_AV2'] = val_eval_results['SMR_AV2']
                all_val_eval_results['SCR'] = val_eval_results['SCR']
            if self.learned_relation_header:
                all_val_eval_results['E-Acc'] = val_eval_results['E-Acc']
                all_val_eval_results['E-Acc 0'] = val_eval_results['E-Acc 0']
                all_val_eval_results['E-Acc 1'] = val_eval_results['E-Acc 1']
                all_val_eval_results['E-Acc 2'] = val_eval_results['E-Acc 2']
                all_val_eval_results['PropNoEdge'] = val_eval_results['PropNoEdge']
            if self.proposal_header:
                all_val_eval_results['pFDE'] = val_eval_results['pFDE']
                all_val_eval_results['pADE'] = val_eval_results['pADE']                

            if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
                ### INTERACTIVE FDE/ADE
                # mask out agents without an incident edge in the ground-truth sparse interaction graph
                connected_mask = np.concatenate(is_connected_all, axis=0).astype(bool)
                mask = init_mask * connected_mask
                val_eval_results = calc_metrics(results, self.config, mask, identifier='int')
                val_eval_results = sync(val_eval_results, self.config, comm)

                all_val_eval_results['iFDE'] = val_eval_results['FDE']
                all_val_eval_results['iADE'] = val_eval_results['ADE']

                ### INTERACTIVE 3 FDE/ADE
                hardness_3_mask = np.load("fde_3_{}.npy".format(self.dataset)).astype(bool)
                mask = init_mask * connected_mask * hardness_3_mask
                val_eval_results = calc_metrics(results, self.config, mask, identifier='int')
                val_eval_results = sync(val_eval_results, self.config, comm)

                all_val_eval_results['iFDE_3'] = val_eval_results['FDE']
                all_val_eval_results['iADE_3'] = val_eval_results['ADE']

                ### INTERACTIVE 5 FDE/ADE
                hardness_5_mask = np.load("fde_5_{}.npy".format(self.dataset)).astype(bool)
                mask = init_mask * connected_mask * hardness_5_mask
                val_eval_results = calc_metrics(results, self.config, mask, identifier='int')
                val_eval_results = sync(val_eval_results, self.config, comm)

                all_val_eval_results['iFDE_5'] = val_eval_results['FDE']
                all_val_eval_results['iADE_5'] = val_eval_results['ADE']

                if self.dataset == "argoverse2":
                    ### SCORED SPLIT, REGULAR FDE/ADE
                    scored_mask = np.concatenate(is_scored_all, axis=0).astype(bool)
                    mask = init_mask * scored_mask
                    val_eval_results = calc_metrics(results, self.config, mask, identifier='reg')
                    val_eval_results = sync(val_eval_results, self.config, comm)

                    all_val_eval_results['FDE_scored'] = val_eval_results['FDE']
                    all_val_eval_results['ADE_scored'] = val_eval_results['ADE']
                    all_val_eval_results['pFDE_scored'] = val_eval_results['pFDE']
                    all_val_eval_results['pADE_scored'] = val_eval_results['pADE']
                    all_val_eval_results['SMR_scored'] = val_eval_results['SMR']
                    all_val_eval_results['SMR_AV2_scored'] = val_eval_results['SMR_AV2']
                    all_val_eval_results['SCR_scored'] = val_eval_results['SCR']
                    
                    ### SCORED SPLIT, INTERACTIVE FDE/ADE
                    mask = init_mask * scored_mask * connected_mask
                    val_eval_results = calc_metrics(results, self.config, mask, identifier='int')
                    val_eval_results = sync(val_eval_results, self.config, comm)

                    all_val_eval_results['iFDE_scored'] = val_eval_results['FDE']
                    all_val_eval_results['iADE_scored'] = val_eval_results['ADE']

                    ### SCORED SPLIT, INTERACTIVE 3 FDE/ADE
                    hardness_3_mask = np.load("fde_3_{}.npy".format(self.dataset)).astype(bool)
                    mask = init_mask * scored_mask * connected_mask * hardness_3_mask
                    val_eval_results = calc_metrics(results, self.config, mask, identifier='int')
                    val_eval_results = sync(val_eval_results, self.config, comm)

                    all_val_eval_results['iFDE_3_scored'] = val_eval_results['FDE']
                    all_val_eval_results['iADE_3_scored'] = val_eval_results['ADE']

                    ### SCORED SPLIT, INTERACTIVE 5 FDE/ADE
                    hardness_5_mask = np.load("fde_5_{}.npy".format(self.dataset)).astype(bool)
                    mask = init_mask * scored_mask * connected_mask * hardness_5_mask
                    val_eval_results = calc_metrics(results, self.config, mask, identifier='int')
                    val_eval_results = sync(val_eval_results, self.config, comm)

                    all_val_eval_results['iFDE_5_scored'] = val_eval_results['FDE']
                    all_val_eval_results['iADE_5_scored'] = val_eval_results['ADE']

        return all_val_eval_results
   
    def init_dgl_graph(self, batch_idxs, ctrs, orig, rot, agenttypes, world_locs, has_preds):        
        n_scenarios = len(np.unique(batch_idxs))
        graphs, labels = [], []
        for ii in range(n_scenarios):
            label = None

            # number of agents in the scene (currently > 0)
            si = ctrs[batch_idxs == ii].shape[0]
            assert si > 0

            # start with a fully-connected graph
            if si > 1:
                off_diag = np.ones([si, si]) - np.eye(si)
                rel_src = np.where(off_diag)[0]
                rel_dst = np.where(off_diag)[1]

                graph = dgl.graph((rel_src, rel_dst))
            else:
                graph = dgl.graph(([], []), num_nodes=si)

            # separate graph for each scenario
            graph.ndata["ctrs"] = ctrs[batch_idxs == ii]
            graph.ndata["rot"] = rot[batch_idxs == ii]
            graph.ndata["orig"] = orig[batch_idxs == ii]
            graph.ndata["agenttypes"] = agenttypes[batch_idxs == ii].float()
            # ground truth future in SE(2)-transformed coordinates
            graph.ndata["ground_truth_futures"] = world_locs[batch_idxs == ii][:, self.observation_steps:]
            graph.ndata["has_preds"] = has_preds[batch_idxs == ii].float()
            
            graphs.append(graph)
            labels.append(label)
        
        graphs = dgl.batch(graphs)
        return graphs

    def build_stage_1_graph(self, graph, x, agenttypes, actor_idcs, actor_ctrs, lane_graph):
        all_edges = [x.unsqueeze(1) for x in graph.edges('uv')]
        all_edges = torch.cat(all_edges, 1)
        
        stage_1_graph = dgl.graph((all_edges[:, 0], all_edges[:, 1]), num_nodes = graph.num_nodes())
        stage_1_graph.ndata["ctrs"] = graph.ndata["ctrs"]
        stage_1_graph.ndata["rot"] = graph.ndata["rot"]
        stage_1_graph.ndata["orig"] = graph.ndata["orig"]
        stage_1_graph.ndata["agenttypes"] = graph.ndata["agenttypes"].float()

        stage_1_graph = self.pretrained_relation_header.feature_encoder(stage_1_graph, x, agenttypes, actor_idcs, actor_ctrs, lane_graph)

        return stage_1_graph

    def forward(self, scene_idxs, graph, stage_1_graph, ig_dict, batch_idxs, batch_idxs_edges, actor_ctrs, ks=None, prop_ground_truth = 0., eval=True):
        
        if self.learned_relation_header:
            edge_logits = self.relation_header(graph)
            graph.edata["edge_logits"] = edge_logits
        else:
            # use ground-truth interaction graph
            if not self.two_stage_training:
                edge_probs = torch.nn.functional.one_hot(ig_dict["ig_labels"].to(dev).long(), self.num_edge_types)
            elif self.two_stage_training and self.training_stage == 2:
                prh_logits = self.pretrained_relation_header.relation_header(stage_1_graph)
                graph.edata["edge_logits"] = prh_logits
        
        all_edges = [x.unsqueeze(1) for x in graph.edges('all')]
        all_edges = torch.cat(all_edges, 1)
        # remove half of the directed edges (effectively now an undirected graph)
        eids_remove = all_edges[torch.where(all_edges[:, 0] > all_edges[:, 1])[0], 2]
        graph.remove_edges(eids_remove)

        if self.learned_relation_header or (self.two_stage_training and self.training_stage == 2):
            edge_logits = graph.edata.pop("edge_logits")
            edge_probs = my_softmax(edge_logits, -1)

        graph.edata["edge_probs"] = edge_probs

        dag_graph = build_graph(graph, self.config)
        
        #if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
        #    dag_graph = prune_graph_johnson(dag_graph)
        
        if self.proposal_header:
            dag_graph, proposals = self.proposal_decoder(dag_graph, actor_ctrs)
        
        if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
            loc_pred = self.trajectory_decoder(dag_graph, prop_ground_truth, batch_idxs)
        
        # loc_pred: shape [N, prediction_steps, num_joint_modes, 2]
        res = {}

        if self.proposal_header:
            res["proposals"] = proposals # trajectory proposal future coordinates
        
        if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
            res["loc_pred"] = loc_pred # predicted future coordinates
        
        if self.learned_relation_header:
            res["edge_logits"] = edge_logits.float() # edge probabilities for computing BCE loss    
            res["edge_probs"] = edge_probs.float()     
        
        return res

    def get_loss(self, graph, batch_idxs, res, agenttypes, has_preds, gt_locs, batch_size, ig_labels, epoch):
        
        huber_loss = nn.HuberLoss(reduction='none')
        
        if self.proposal_header:
            ### Proposal Regression Loss
            has_preds_mask = has_preds.unsqueeze(-1).unsqueeze(-1)
            has_preds_mask = has_preds_mask.expand(has_preds_mask.shape[0], has_preds_mask.shape[1], self.num_proposals, 2).bool().to(dev)

            proposals = res["proposals"]
            
            if self.supervise_vehicles and self.dataset=='interaction':
                # only compute loss on vehicle trajectories
                vehicle_mask = agenttypes[:, 1].bool()
            else:
                # compute loss on all trajectories
                vehicle_mask = torch.ones(agenttypes[:, 1].shape).bool().to(dev)
            
            vehicle_mask = vehicle_mask.cpu()
            has_preds_mask = has_preds_mask[vehicle_mask]
            proposals = proposals[vehicle_mask]
            gt_locs = gt_locs[vehicle_mask]
            batch_idxs = batch_idxs[vehicle_mask]

            target = torch.stack([gt_locs] * self.num_proposals, dim=2).to(dev)

            # Regression loss
            loss_prop_reg = huber_loss(proposals, target)
            loss_prop_reg = loss_prop_reg * has_preds_mask

            b_s = torch.zeros((batch_size, self.num_proposals)).to(loss_prop_reg.device)
            count = 0
            for i, batch_num_nodes_i in enumerate(graph.batch_num_nodes()):
                batch_num_nodes_i = batch_num_nodes_i.item()
                
                batch_loss_prop_reg = loss_prop_reg[count:count+batch_num_nodes_i]    
                # divide by number of agents in the scene        
                b_s[i] = torch.sum(batch_loss_prop_reg, (0, 1, 3)) / batch_num_nodes_i

                count += batch_num_nodes_i

            # sanity check
            assert batch_size == (i + 1)

            loss_prop_reg = torch.min(b_s, dim=1)[0].mean()        
        
        if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
            ### Regression Loss
            # has_preds: [N, 30]
            # res["loc_pred"]: [N, 30, 6, 2]
            has_preds_mask = has_preds.unsqueeze(-1).unsqueeze(-1)
            has_preds_mask = has_preds_mask.expand(has_preds_mask.shape[0], has_preds_mask.shape[1], self.num_joint_modes, 2).bool().to(dev)
            
            loc_pred = res["loc_pred"]
            
            if not self.proposal_header:
                if self.supervise_vehicles and self.dataset=='interaction':
                    vehicle_mask = agenttypes[:, 1].bool()
                else:
                    vehicle_mask = torch.ones(agenttypes[:, 1].shape).bool().to(dev)
    
                gt_locs = gt_locs[vehicle_mask]
                batch_idxs = batch_idxs[vehicle_mask]
            
            has_preds_mask = has_preds_mask[vehicle_mask]
            loc_pred = loc_pred[vehicle_mask]
            
            target = torch.stack([gt_locs] * self.num_joint_modes, dim=2).to(dev)

            # Regression loss
            reg_loss = huber_loss(loc_pred, target)

            # 0 out loss for the indices that don't have a ground-truth prediction.
            reg_loss = reg_loss * has_preds_mask

            b_s = torch.zeros((batch_size, self.num_joint_modes)).to(reg_loss.device)
            count = 0
            for i, batch_num_nodes_i in enumerate(graph.batch_num_nodes()):
                batch_num_nodes_i = batch_num_nodes_i.item()
                
                batch_reg_loss = reg_loss[count:count+batch_num_nodes_i]    
                # divide by number of agents in the scene        
                b_s[i] = torch.sum(batch_reg_loss, (0, 1, 3)) / batch_num_nodes_i

                count += batch_num_nodes_i

            # sanity check
            assert batch_size == (i + 1)

            loss_reg = torch.min(b_s, dim=1)[0].mean()      

        # Relation Loss
        if self.learned_relation_header:
            if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 1):
                if self.focal_loss:
                    ce_loss = FocalLoss(weight=torch.Tensor([self.weight_0, self.weight_1, self.weight_2]).to(dev), gamma=self.gamma, reduction='mean')
                else:
                    ce_loss = nn.CrossEntropyLoss(weight=torch.Tensor([self.weight_0, self.weight_1, self.weight_2]).to(dev))

                # Now compute relation cross entropy loss
                relations_preds = res["edge_logits"]
                relations_gt = ig_labels.to(relations_preds.device).long()

                loss_rel = ce_loss(relations_preds, relations_gt)     
        
        if not self.two_stage_training:
            loss = loss_reg
            
            if self.proposal_header:
                loss = loss + self.proposal_coef * loss_prop_reg

            if self.learned_relation_header:
                loss = loss + self.rel_coef * loss_rel

            loss_dict = {"total_loss": loss,
                        "loss_reg": loss_reg
                        }

            if self.proposal_header:
                loss_dict["loss_prop_reg"] = loss_prop_reg * self.proposal_coef
            
            if self.learned_relation_header:
                loss_dict["loss_rel"] = self.rel_coef * loss_rel                   

        else:
            if self.training_stage == 1:
                loss = self.rel_coef * loss_rel
                if self.proposal_header:
                    loss = loss + loss_prop_reg * self.proposal_coef
                
                loss_dict = {"total_loss": loss,
                             "loss_rel": self.rel_coef * loss_rel} 

                if self.proposal_header:
                    loss_dict["loss_prop_reg"] = loss_prop_reg * self.proposal_coef

            else:
                loss = loss_reg
                
                if self.proposal_header:
                    loss = loss + loss_prop_reg * self.proposal_coef
                
                loss_dict = {"total_loss": loss,
                             "loss_reg": loss_reg} 
                             
                if self.proposal_header:
                    loss_dict["loss_prop_reg"] = loss_prop_reg * self.proposal_coef
        
        return loss_dict

    def save_current_epoch(self, epoch, optimizer, val_best, ade_best, fde_best):
        # save best model to pt file
        path = self.log_path / "current_model_{}.pt".format(epoch)
        state = {
            'epoch': epoch,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_best': val_best, 
            'ade_best': ade_best,
            'fde_best': fde_best
            }
        torch.save(state, path)

    def save(self, epoch, optimizer, val_best, ade_best, fde_best):
        # save best model to pt file
        path = self.log_path / "best_model.pt"
        state = {
            'epoch': epoch,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_best': val_best, 
            'ade_best': ade_best,
            'fde_best': fde_best
            }
        torch.save(state, path)

    def save_relation_header(self, epoch, optimizer, val_edge_acc_best):
        # save best model to pt file
        path = self.log_path / "best_model_relation_header.pt"
        state = {
            'epoch': epoch,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_edge_acc_best': val_edge_acc_best
            }
        torch.save(state, path)

    def load_relation_header(self):
        # load best model from pt file
        path = self.log_path / "best_model_relation_header.pt"
        state = torch.load(path, map_location=dev)
        self.load_state_dict(state['state_dict'])

    def load_for_train_stage_1(self, optimizer):
        path = self.log_path / "best_model_relation_header.pt"
        state = torch.load(path, map_location=dev)
        self.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])

        return optimizer, state['epoch'] + 1, state['val_edge_acc_best']
    
    def load_for_train(self, optimizer):
        # load best model from pt file
        path = self.log_path / "best_model.pt"
        state = torch.load(path, map_location=dev)
        self.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])

        return optimizer, state['epoch'] + 1, state['val_best'], state['ade_best'], state['fde_best']

    def prepare_for_stage_2(self, pretrained_relation_header):
        # first, load model from stage 1 and set weights for stage 2
        path = self.log_path / "best_model_relation_header.pt"
        state = torch.load(path, map_location=dev)
        pretrained_relation_header.load_state_dict(state['state_dict'])

        # second, freeze the weights of the network trained in stage 1
        for param in pretrained_relation_header.parameters():
            param.requires_grad = False

        self.pretrained_relation_header = pretrained_relation_header

    def load_for_eval(self):
        # load best model from pt file
        path = self.log_path / "best_model.pt"
        state = torch.load(path, map_location=dev)
        self.load_state_dict(state['state_dict'])

    def _eval_constant_velocity(self, val_loader, epoch):
        hvd.broadcast_parameters(self.state_dict(), root_rank=0)

        self.eval()
        
        fde = []
        n_counted_in_eval = 0
        n_counted_in_interactive_eval = 0
        
        with torch.no_grad():
            tot_log = self.num_val_samples // (self.batch_size * hvd.size())
            for i, data in enumerate(val_loader):
                dd = process_data(data, self.config)

                x = dd['world_locs'][:,:self.observation_steps]
                # transform into gt global coordinate frame
                x = torch.matmul(x, dd["rot"]) + dd["orig"].view(-1, 1, 2)
                N = x.shape[0]

                x = x.detach().cpu().numpy()
                loc_pred = np.zeros((N, self.prediction_steps, 2))
                
                for j in range(N):
                    counted_interaction = (dd['has_preds'][j, -1] == 1) and (dd['agenttypes'][j, 1] == 1)
                    counted_argoverse2 = (dd['has_preds'][j, -1] == 1)
                    
                    if (self.dataset == 'interaction' and counted_interaction) or (self.dataset == 'argoverse2' and counted_argoverse2):
                        # interactive case
                        if dd['is_connected'][j] == 1:
                            n_counted_in_interactive_eval += 1
                            n_counted_in_eval += 1
                        else:
                            n_counted_in_eval += 1                    
                    
                    final_x, final_y = estimate_constant_velocity(x[j], self.prediction_steps, dd['has_obs'][j, :self.observation_steps])
                    gt_final_x, gt_final_y = dd['gt_locs'][j, -1, 0], dd['gt_locs'][j, -1, 1]
                    fde.append(evaluate_fde(final_x, final_y, gt_final_x, gt_final_y))
                
                if i % 10 == 0:
                    print_("Validation data: ", "{:.2f}%".format(i * 100 / tot_log))
        
        fde = np.array(fde)
        
        print("FDE: ", np.mean(fde), n_counted_in_eval, n_counted_in_interactive_eval)
        
        print(np.mean(fde > 5))
        np.save("fde_5_{}.npy".format(self.dataset), fde > 5)
        print(np.mean(fde > 3))
        np.save("fde_3_{}.npy".format(self.dataset), fde > 3)

