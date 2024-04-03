from torch import nn
import torch
from dag_utils import build_graph
from graph_transformer import GTBackbone
from relation_header import FJMPRelationHeader
from utils import *
from lanegcn_modules import *
from modules import *
from mpi4py import MPI
import time 
from gmm import GMMDecoder

comm = MPI.COMM_WORLD
dev = 'cuda:{}'.format(0)


class GNNPipeline(nn.Module):
    def __init__(self, config):
        super(GNNPipeline, self).__init__()
        self.config = config
        self.model_path = config["model_path"]
        self.observation_steps = config["observation_steps"]
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
        self.supervise_vehicles = config["supervise_vehicles"]

        self.build()

    def build(self):

        model_dict = torch.load(self.model_path, map_location=dev)

        self.relation_feature_encoder = FJMPFeatureEncoder(self.config).to(dev)
        self.relation_feature_encoder.load_state_dict(model_dict['feature_state_dict'])

        self.relation_header = FJMPRelationHeader(self.config).to(dev)
        self.relation_header.load_state_dict(model_dict['relation_state_dict'])

        for param in self.relation_feature_encoder.parameters():
            param.requires_grad = False

        for param in self.relation_header.parameters():
            param.requires_grad = False

        self.feature_encoder = FJMPFeatureEncoder(self.config).to(dev)
        self.aux_prop_decoder = GMMDecoder(self.config).to(dev)
        #self.gnn_backbone = GTBackbone(self.config).to(dev)
        self.gmm_decoder = GMMDecoder(self.config).to(dev)  #LaneGCNHeader(self.config).to(dev)#GMMDecoder(self.config).to(dev) 


    def get_graph_logits(self, graph, x, agenttypes, actor_idcs, actor_ctrs, lane_graph):
        all_edges = [x.unsqueeze(1) for x in graph.edges('uv')]
        all_edges = torch.cat(all_edges, 1)
        
        base_graph = dgl.graph((all_edges[:, 0], all_edges[:, 1]), num_nodes = graph.num_nodes())
        base_graph.ndata["ctrs"] = graph.ndata["ctrs"]
        base_graph.ndata["rot"] = graph.ndata["rot"]
        base_graph.ndata["orig"] = graph.ndata["orig"]
        base_graph.ndata["agenttypes"] = graph.ndata["agenttypes"].float()

        base_graph = self.relation_feature_encoder(base_graph, x, agenttypes, actor_idcs, actor_ctrs, lane_graph)
        edge_logits = self.relation_header(base_graph)

        return edge_logits


    def forward(self, dd, training=False):

        dgl_graph = self.init_dgl_graph(dd['batch_idxs'], dd['ctrs'], dd['orig'], dd['rot'], dd['agenttypes'], dd['world_locs'], dd['has_preds']).to(dev)
        edge_logits = self.get_graph_logits(dgl_graph, dd['feats'][:,:self.observation_steps], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph']) 
        dgl_graph.edata["edge_logits"] = edge_logits

        dgl_graph = self.feature_encoder(dgl_graph, dd['feats'][:,:self.observation_steps], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])

        all_edges = [x.unsqueeze(1) for x in dgl_graph.edges('all')]
        all_edges = torch.cat(all_edges, 1)
        # remove half of the directed edges (effectively now an undirected graph)
        eids_remove = all_edges[torch.where(all_edges[:, 0] > all_edges[:, 1])[0], 2]
        dgl_graph.remove_edges(eids_remove)

        edge_logits = dgl_graph.edata.pop("edge_logits")

        edge_probs = my_softmax(edge_logits, -1)

        dgl_graph.edata["edge_probs"] = edge_probs

        new_graphs = [build_graph(gs, self.config) for gs in dgl.unbatch(dgl_graph)]
        pos_enc = [dgl.random_walk_pe(gs, k=2).to(dev) for gs in new_graphs]
        pos_enc = torch.cat(pos_enc, 0)

        dgl_graph = dgl.batch(new_graphs)
        dgl_graph.ndata["pos_enc"] = pos_enc

        if training:
            prop_gmm_params, prop_score = self.aux_prop_decoder(dgl_graph)

        #dgl_graph = self.gnn_backbone(dgl_graph)

        gmm_params, score = self.gmm_decoder(dgl_graph)

        if training:
            return gmm_params, score, prop_gmm_params, prop_score, dgl_graph

        return gmm_params, score

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


    def _train(self, train_loader, val_loader, optimizer, start_epoch, val_best, ade_best, fde_best):        
        hvd.broadcast_parameters(self.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        for epoch in range(start_epoch, self.max_epochs + 1):   
            # this shuffles the training set every epoch         
            train_loader.sampler.set_epoch(int(epoch))
            
            t_start_epoch = time.time()
            self.train()
            tot_log = self.num_train_samples // (self.batch_size * hvd.size())  

            for e, param_group in enumerate(optimizer.param_groups):
                if epoch == self.switch_lr_1 or epoch == self.switch_lr_2:
                    param_group["lr"] = param_group["lr"] * (self.lr_step)
                
                if e == 0:
                    cur_lr = param_group["lr"]  


            accum_gradients = {}
            for i, data in enumerate(train_loader):     
                dd = process_data(data, self.config)

                gmm_params, score, prop_gmm_params, prop_score, dgl_graph = self.forward(dd, training=True) 
    
                loss = self.get_loss(dgl_graph, dd['batch_idxs'], gmm_params, score, prop_gmm_params, prop_score, dd['agenttypes'], dd['has_preds'], dd['gt_locs'], dd['batch_size'])
                
                optimizer.zero_grad()
                loss.backward()
                accum_gradients = accumulate_gradients(accum_gradients, self.named_parameters())
                optimizer.step()
                
                if i % 100 == 0:
                    print_("Training data: ", "{:.2f}%".format(i * 100 / tot_log), "lr={:.3e}".format(cur_lr), "rel_coef={:.1f}".format(self.rel_coef),
                        "\t".join([k + ":" + f"{v.item():.3f}" for k, v in {"loss": loss}.items()]))

            self.eval()
            val_eval_results = self._eval(val_loader, epoch)
            
            print_("Epoch {} validation-set results: ".format(epoch), "\t".join([f"{k}: {v}" if type(v) is np.ndarray else f"{k}: {v:.3f}" for k, v in val_eval_results.items()]))
            
            # Best model is one with minimum FDE + ADE
            if hvd.rank() == 0:
                if (val_eval_results["FDE"] + val_eval_results["ADE"]) < val_best:
                    val_best = val_eval_results["FDE"] + val_eval_results["ADE"]
                    ade_best = val_eval_results["ADE"]
                    fde_best = val_eval_results["FDE"]
                    self.save(epoch, optimizer, val_best, ade_best, fde_best)
                    print_("Validation FDE+ADE improved. Saving model. ")
                print_("Best loss: {:.4f}".format(val_best), "Best ADE: {:.3f}".format(ade_best), "Best FDE: {:.3f}".format(fde_best))

            print_("Epoch {} time: {:.3f}s".format(epoch, time.time() - t_start_epoch))


    def save(self, epoch, optimizer, val_best, ade_best, fde_best):
        # save best model to pt file
        path = self.log_path / "best_gnn_model.pt"
        state = {
            'epoch': epoch,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_best': val_best, 
            'ade_best': ade_best,
            'fde_best': fde_best
            }
        torch.save(state, path)


    def _eval(self, val_loader, epoch):
        hvd.broadcast_parameters(self.state_dict(), root_rank=0)

        self.eval()

        loc_preds, gt_locs_all, agenttypes_all, has_last_all, has_preds_all, batch_idxs_all = [], [], [], [], [], []

        tot = 0
        with torch.no_grad():
            tot_log = self.num_val_samples // (self.batch_size * hvd.size())            
            for i, data in enumerate(val_loader):

                dd = process_data(data, self.config)
            
                gmm_params, score = self.forward(dd) 
                gmm_params = gmm_params[..., :2]

                loc_preds.append(gmm_params.detach().cpu())
                gt_locs_all.append(dd['gt_locs'].detach().cpu())
                has_last_all.append(dd['has_last'].detach().cpu())
                has_preds_all.append(dd['has_preds'].detach().cpu())
                batch_idxs_all.append(dd['batch_idxs'].detach().cpu() + tot)
                agenttypes_all.append(dd['agenttypes'].detach().cpu())

                tot += dd['batch_size']

            loc_preds = np.concatenate(loc_preds, axis=0)
            gt_locs_all = np.concatenate(gt_locs_all, axis=0)
            has_preds_all = np.concatenate(has_preds_all, axis=0)
            batch_idxs_all = np.concatenate(batch_idxs_all)
            has_last_mask = np.concatenate(has_last_all, axis=0).astype(bool)

        
        if self.dataset=='interaction':
            # only evaluate vehicles
            eval_agent_mask = np.concatenate(agenttypes_all, axis=0)[:, 1].astype(bool)
        else:
            # evaluate all context agents
            eval_agent_mask = np.ones(np.concatenate(agenttypes_all, axis=0)[:, 1].shape).astype(bool)

        mask = has_last_mask * eval_agent_mask

        gt_locs_masked = gt_locs_all[mask]
        has_preds_masked = has_preds_all[mask].astype(bool)
        batch_idxs_masked = batch_idxs_all[mask]
        loc_preds_masked = loc_preds[mask]

        n_scenarios = np.unique(batch_idxs_masked).shape[0]
        scenarios = np.unique(batch_idxs_masked).astype(int)

        has_preds_all_mask = np.reshape(has_preds_masked, has_preds_masked.shape + (1,))
        has_preds_all_mask = np.broadcast_to(has_preds_all_mask, has_preds_masked.shape[:2] + (self.config["num_joint_modes"],))  

        num_joint_modes = loc_preds_masked.shape[2]
        gt_locs_masked = np.stack([gt_locs_masked]*num_joint_modes, axis=2)

        mse_error = (loc_preds_masked - gt_locs_masked)**2

        euclidean_rmse = np.sqrt(mse_error.sum(-1))   
        
        euclidean_rmse_filtered = np.zeros(euclidean_rmse.shape)
        euclidean_rmse_filtered[has_preds_all_mask] = euclidean_rmse[has_preds_all_mask]
    
        # mean over the agents then min over the num_joint_modes samples then mean over the scenarios
        mean_FDE = np.zeros((n_scenarios, num_joint_modes))
        mean_ADE = np.zeros((n_scenarios, num_joint_modes))
        
        for j, i in enumerate(scenarios):
            i = int(i)
            has_preds_all_i = has_preds_masked[batch_idxs_masked == i]
            euclidean_rmse_filtered_i = euclidean_rmse_filtered[batch_idxs_masked == i]
            mean_FDE[j] = euclidean_rmse_filtered_i[:, -1].mean(0)
            mean_ADE[j] = euclidean_rmse_filtered_i.sum((0, 1)) / has_preds_all_i.sum()

        FDE = mean_FDE.min(1).mean()
        ADE = mean_ADE.min(1).mean()
  
        data = {
            "FDE": FDE,
            "ADE": ADE,
            "n_scenarios": n_scenarios}

        data_list = comm.allgather(data)

        FDE = 0
        ADE = 0
        n_scenarios = 0

        for i in range(len(data_list)):
            FDE += data_list[i]['FDE'] * data_list[i]['n_scenarios']
            ADE += data_list[i]['ADE'] * data_list[i]['n_scenarios']
            n_scenarios += data_list[i]['n_scenarios']

        FDE /= n_scenarios
        ADE /= n_scenarios

        results = {
            'FDE': FDE,
            'ADE': ADE,
        }

        return results
    
    def get_loss(self, graph, batch_idxs, gmm_params, score, prop_gmm_params, prop_score, agenttypes, has_preds, gt_locs, batch_size):
            #proposal loss
            has_preds_mask = has_preds.unsqueeze(-1).unsqueeze(-1).to(dev)
            
            if self.supervise_vehicles and self.dataset=='interaction':
                # only compute loss on vehicle trajectories
                vehicle_mask = agenttypes[:, 1].bool()
            else:
                # compute loss on all trajectories
                vehicle_mask = torch.ones(agenttypes[:, 1].shape).bool().to(dev)
            
            vehicle_mask = vehicle_mask.cpu()
            has_preds_mask = has_preds_mask[vehicle_mask]
            prop_gmm_params = prop_gmm_params[vehicle_mask]
            prop_score = prop_score[vehicle_mask]
            gt_locs = gt_locs[vehicle_mask]
            batch_idxs = batch_idxs[vehicle_mask]

            prop_gmm_loss, prop_score_loss = self.imitation_loss(prop_gmm_params, prop_score, gt_locs.to(dev))

            # Regression loss
            loss_prop_reg = (prop_gmm_loss * has_preds_mask).mean() + prop_score_loss

            # regression loss                                                
            gmm_loss, score_loss = self.imitation_loss(gmm_params, score, gt_locs.to(dev))

            # 0 out loss for the indices that don't have a ground-truth prediction.
            loss_reg = (gmm_loss * has_preds_mask).mean() + score_loss

            loss = loss_reg + loss_prop_reg * self.proposal_coef

            return loss
    
    def imitation_loss(self, gmm, scores, ground_truth):

        mu = gmm[..., :2]
        ground_truth = ground_truth[..., None, :]
        dx = ground_truth[..., 0] - mu[..., 0]
        dy = ground_truth[..., 1] - mu[..., 1]

        cov = gmm[..., 2:]
        log_std_x = torch.clamp(cov[..., 0], -2, 2)
        log_std_y = torch.clamp(cov[..., 1], -2, 2)
        std_x = torch.exp(log_std_x)
        std_y = torch.exp(log_std_y)


        gmm_loss = log_std_x + log_std_y + 0.5 * (torch.square(dx/std_x) + torch.square(dy/std_y))
        gmm_loss = gmm_loss.mean(1)

        best_mode = torch.argmin(gmm_loss, dim=-1)
        score_loss = F.cross_entropy(scores, best_mode, label_smoothing=0.2)

        gmm_loss = torch.min(gmm_loss, dim=-1).values
        
        return gmm_loss, score_loss
