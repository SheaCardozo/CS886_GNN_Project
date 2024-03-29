from torch import nn
import torch
from utils import *
from lanegcn_modules import *
from modules import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
dev = 'cuda:{}'.format(0)


class FJMPRelationHeader(nn.Module):
    def __init__(self, config):
        super(FJMPRelationHeader, self).__init__()
        self.config = config
        
        self.agenttype_enc = Linear(2 * self.config["num_agenttypes"], self.config["h_dim"])
        self.dist = Linear(2, self.config["h_dim"])
        # convert src/dst features to edge features
        self.f_1e = MLP(self.config['h_dim'] * 4, self.config['h_dim'])

        self.h_1e_out = nn.Sequential(
                LinearRes(self.config['h_dim'], self.config['h_dim']),
                nn.Linear(self.config['h_dim'], self.config["num_edge_types"]),
            )
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias, 0.1)
    
    def message_func1(self, edges):
        agenttype_enc = self.agenttype_enc(torch.cat([edges.src['agenttypes'], edges.dst['agenttypes']], dim=-1))
        dist = self.dist(edges.dst['ctrs'] - edges.src['ctrs'])
        h_1e = self.f_1e(torch.cat([edges.src['xt_enc'], edges.dst['xt_enc'], dist, agenttype_enc], dim=-1))
        
        edges.data['h_1e'] = h_1e
        return {'h_1e': h_1e}
    
    def node_to_edge(self, graph):
        # propagate edge features back to nodes
        graph.apply_edges(self.message_func1)

        return graph

    def forward(self, graph):
        graph = self.node_to_edge(graph)
        h_1e_out = self.h_1e_out(graph.edata["h_1e"])
        
        return h_1e_out

class FJMPHeaderEncoderTrainer(nn.Module):
    def __init__(self, config):
        super(FJMPHeaderEncoderTrainer, self).__init__()
        self.config = config
        self.num_train_samples = config["num_train_samples"]
        self.switch_lr_1 = config["switch_lr_1"]
        self.switch_lr_2 = config["switch_lr_2"]
        self.lr_step = config["lr_step"]
        self.observation_steps = config["observation_steps"]
        self.log_path = config["log_path"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["lr"]
        self.max_epochs = config["max_epochs"]
        self.gamma = config["gamma"]
        self.weight_0 = config["weight_0"]
        self.weight_1 = config["weight_1"]
        self.weight_2 = config["weight_2"]
        self.rel_coef = config["rel_coef"]
        self.supervise_vehicles = config["supervise_vehicles"]
        self.num_proposals = config["num_proposals"]

        self.build()
    
    def _train(self, train_loader, val_loader, optimizer, start_epoch):

        val_edge_acc_best = 0.

        hvd.broadcast_parameters(self.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        for epoch in range(start_epoch, self.max_epochs + 1):
            train_loader.sampler.set_epoch(int(epoch))
            
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

                dgl_graph = self.init_dgl_graph(dd['batch_idxs'], dd['ctrs'], dd['orig'], dd['rot'], dd["agenttypes"], dd['world_locs'], dd['has_preds']).to(dev)
                dgl_graph = self.feature_encoder(dgl_graph, dd['feats'][:,:self.observation_steps], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])

                edge_logits = self.relation_header(dgl_graph)


                dgl_graph.edata["edge_logits"] = edge_logits

                all_edges = [x.unsqueeze(1) for x in dgl_graph.edges('all')]
                all_edges = torch.cat(all_edges, 1)
                # remove half of the directed edges (effectively now an undirected graph)
                eids_remove = all_edges[torch.where(all_edges[:, 0] > all_edges[:, 1])[0], 2]
                dgl_graph.remove_edges(eids_remove)
                
                relations_preds = dgl_graph.edata.pop("edge_logits")
                relations_gt = dd["ig_labels"].to(relations_preds.device).long()

                ce_loss = FocalLoss(weight=torch.Tensor([self.weight_0, self.weight_1, self.weight_2]).to(dev), gamma=self.gamma, reduction='mean')
                loss_rel = ce_loss(relations_preds, relations_gt)   

                dgl_graph, proposals = self.proposal_decoder(dgl_graph, dd["actor_ctrs"])

                prop_loss = self.get_prop_loss(dgl_graph, dd['batch_idxs'], proposals, dd['agenttypes'], dd['has_preds'], dd['gt_locs'], dd['batch_size'])

                loss = loss_rel + self.rel_coef * prop_loss 
                optimizer.zero_grad()
                loss.backward()
                accum_gradients = accumulate_gradients(accum_gradients, self.named_parameters())
                optimizer.step()
                
                if i % 100 == 0:
                    print_("Training data: ", "{:.2f}%".format(i * 100 / tot_log), "lr={:.3e}".format(cur_lr),
                        "\t".join([k + ":" + f"{v.item():.3f}" for k, v in {"loss": loss}.items()]),)
                    
            self.eval()
            edge_acc, ea0, ea1, ea2 = self._eval(val_loader, epoch)

            val_dict = {"val acc": edge_acc, "val acc 0": ea0, "val acc 1": ea1, "val acc 2": ea2}
            print_("Epoch {} validation-set results: ".format(epoch), "\t".join([f"{k}: {v}" if type(v) is np.ndarray else f"{k}: {v:.3f}" for k, v in val_dict.items()]))

            if hvd.rank() == 0:

                if edge_acc > val_edge_acc_best:
                    print_("Validation Edge Accuracy improved.")  
                    val_edge_acc_best = edge_acc  
                
                    print_("Saving relation header")
                    self.save_relation_header(epoch, optimizer, val_edge_acc_best)    
                    print_("Best validation edge accuracy: {:.4f}".format(val_edge_acc_best))     

    def _eval(self, val_loader, epoch):

        ig_preds = []
        ig_labels_all = []            

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                dd = process_data(data, self.config)
                
                dgl_graph = self.init_dgl_graph(dd['batch_idxs'], dd['ctrs'], dd['orig'], dd['rot'], dd['agenttypes'], dd['world_locs'], dd['has_preds']).to(dev)
                dgl_graph = self.feature_encoder(dgl_graph, dd['feats'][:,:self.observation_steps], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])

                edge_logits = self.relation_header(dgl_graph)

                dgl_graph.edata["edge_logits"] = edge_logits

                all_edges = [x.unsqueeze(1) for x in dgl_graph.edges('all')]
                all_edges = torch.cat(all_edges, 1)
                # remove half of the directed edges (effectively now an undirected graph)
                eids_remove = all_edges[torch.where(all_edges[:, 0] > all_edges[:, 1])[0], 2]
                dgl_graph.remove_edges(eids_remove)
                
                relations_preds = dgl_graph.edata.pop("edge_logits")
                edge_probs = my_softmax(relations_preds, -1)

                ig_labels_all.append(dd["ig_labels"].detach().cpu())                                            
                ig_preds.append(edge_probs.detach().cpu())

        results_ig_preds = np.concatenate(ig_preds, axis=0)
        results_ig_labels_all = np.concatenate(ig_labels_all, axis=0)    

        ig_preds = np.argmax(results_ig_preds, axis=1)
        relation_accuracy = np.mean(ig_preds == results_ig_labels_all)

        edge_mask_0 = results_ig_labels_all == 0
        edge_mask_1 = results_ig_labels_all == 1
        edge_mask_2 = results_ig_labels_all == 2

        edge_accuracy_0 = np.mean(ig_preds[edge_mask_0] == 0)
        edge_accuracy_1 = np.mean(ig_preds[edge_mask_1] == 1)
        edge_accuracy_2 = np.mean(ig_preds[edge_mask_2] == 2)  

        return np.mean(comm.allgather(relation_accuracy)), np.mean(comm.allgather(edge_accuracy_0)), np.mean(comm.allgather(edge_accuracy_1)), np.mean(comm.allgather(edge_accuracy_2))
    
        
    def save_models(self, epoch, optimizer, val_edge_acc_best):
        # save best model to pt file
        path = self.log_path / "best_models.pt"
        state = {
            'epoch': epoch,
            'relation_state_dict': self.relation_header.state_dict(),
            'feature_state_dict': self.feature_encoder.state_dict(),
            'proposal_state_dict': self.proposal_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_edge_acc_best': val_edge_acc_best
            }
        torch.save(state, path)


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


    def build(self):
        self.relation_header = FJMPRelationHeader(self.config).to(dev) 
        self.feature_encoder = FJMPFeatureEncoder(self.config).to(dev)
        self.proposal_decoder = FJMPTrajectoryProposalDecoder(self.config).to(dev)
        

    def get_prop_loss(self, graph, batch_idxs, proposals, agenttypes, has_preds, gt_locs, batch_size):
        
        huber_loss = nn.HuberLoss(reduction='none')
        
        ### Proposal Regression Loss
        has_preds_mask = has_preds.unsqueeze(-1).unsqueeze(-1)
        has_preds_mask = has_preds_mask.expand(has_preds_mask.shape[0], has_preds_mask.shape[1], self.num_proposals, 2).bool().to(dev)
        
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
        return loss_prop_reg