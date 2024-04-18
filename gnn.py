from torch import nn
import torch
from dag_utils import build_graph
from graph_transformer import GTBackbone
from relation_header import FJMPRelationHeader
from utils import *
from lanegcn_modules import *
from modules import *
import time 
from gmm import GMMDecoder


class GNNPipeline(nn.Module):
    def __init__(self, config, device):
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

        self.device = device

        self.build()

    def build(self):

        model_dict = torch.load(self.model_path, map_location=self.device)

        self.relation_feature_encoder = FJMPFeatureEncoder(self.config).to(self.device)
        self.relation_feature_encoder.load_state_dict(model_dict['feature_state_dict'])

        self.relation_header = FJMPRelationHeader(self.config).to(self.device)
        self.relation_header.load_state_dict(model_dict['relation_state_dict'])

        for param in self.relation_feature_encoder.parameters():
            param.requires_grad = False

        for param in self.relation_header.parameters():
            param.requires_grad = False

        self.feature_encoder = FJMPFeatureEncoder(self.config).to(self.device)
        self.aux_prop_decoder = GMMDecoder(self.config, self.num_proposals).to(self.device) 
        self.gnn_backbone = GTBackbone(self.config).to(self.device)
        self.gmm_decoder = GMMDecoder(self.config, self.num_joint_modes).to(self.device)  #LaneGCNHeader(self.config).to(dev)#GMMDecoder(self.config).to(dev) 

        '''
        m = sum(p.numel() for p in self.relation_feature_encoder.parameters())
        print("Relation feature encoder: {} parameters".format(m))

        m = sum(p.numel() for p in self.relation_header.parameters())
        print("Relation header: {} parameters".format(m))

        m = sum(p.numel() for p in self.feature_encoder.parameters())
        print("Feature encoder: {} parameters".format(m))

        m = sum(p.numel() for p in self.aux_prop_decoder.parameters())
        print("Auxiliary proposal decoder: {} parameters".format(m))

        m = sum(p.numel() for p in self.gmm_decoder.parameters())
        print("GMM decoder: {} parameters".format(m))

        m = sum(p.numel() for p in self.gnn_backbone.parameters())
        print("GNN backbone: {} parameters".format(m))
        '''

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


    def forward(self, dd, train=False):

        dgl_graph = self.init_dgl_graph(dd['batch_idxs'], dd['ctrs'], dd['orig'], dd['rot'], dd['agenttypes'], dd['world_locs'], dd['has_preds']).to(self.device)
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
        pos_enc = [dgl.random_walk_pe(gs, k=2).to(self.device) for gs in new_graphs]
        pos_enc = torch.cat(pos_enc, 0)

        dgl_graph = dgl.batch(new_graphs)
        dgl_graph.ndata["pos_enc"] = pos_enc

        if train:
            prop_gmm_params = self.aux_prop_decoder(dgl_graph)

        dgl_graph = self.gnn_backbone(dgl_graph)

        gmm_params = self.gmm_decoder(dgl_graph)

        if train:
            return self.get_loss(dgl_graph, dd['batch_idxs'], gmm_params, prop_gmm_params, dd['agenttypes'], dd['has_preds'], dd['gt_locs'], dd['batch_size'])

        return gmm_params

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
    
    def imitation_loss(self, gmm, ground_truth):
        mu = gmm[..., :2]
        dx = ground_truth[..., 0] - mu[..., 0]
        dy = ground_truth[..., 1] - mu[..., 1]

        cov = gmm[..., 2:]
        log_std_x = torch.clamp(cov[..., 0], -2, 2)
        log_std_y = torch.clamp(cov[..., 1], -2, 2)
        std_x = torch.exp(log_std_x)
        std_y = torch.exp(log_std_y)

        gmm_loss = log_std_x + log_std_y + 0.5 * (torch.square(dx/std_x) + torch.square(dy/std_y)) + torch.ones_like(log_std_x) * 4

        return gmm_loss

    
    def get_loss(self, graph, batch_idxs, gmm_params, prop_gmm_params, agenttypes, has_preds, gt_locs, batch_size):

            loc_pred, proposals = gmm_params, prop_gmm_params
            #proposal loss
            has_preds_mask = has_preds.unsqueeze(-1).unsqueeze(-1)
            has_preds_mask = has_preds_mask.expand(has_preds_mask.shape[0], has_preds_mask.shape[1], self.num_proposals, 1).bool().squeeze().to(self.device)
            
            if self.supervise_vehicles and self.dataset=='interaction':
                # only compute loss on vehicle trajectories
                vehicle_mask = agenttypes[:, 1].bool()
            else:
                # compute loss on all trajectories
                vehicle_mask = torch.ones(agenttypes[:, 1].shape).bool().to(self.device)
            
            vehicle_mask = vehicle_mask.cpu()
            has_preds_mask = has_preds_mask[vehicle_mask]
            proposals = proposals[vehicle_mask]
            gt_locs = gt_locs[vehicle_mask]
            batch_idxs = batch_idxs[vehicle_mask]

            target = torch.stack([gt_locs] * self.num_proposals, dim=2).to(self.device)

            # Regression loss
            loss_prop_reg = self.imitation_loss(proposals, target)

            loss_prop_reg = loss_prop_reg * has_preds_mask

            b_s = torch.zeros((batch_size, self.num_proposals)).to(self.device)
            count = 0
            for i, batch_num_nodes_i in enumerate(graph.batch_num_nodes()):
                batch_num_nodes_i = batch_num_nodes_i.item()
                
                batch_loss_prop_reg = loss_prop_reg[count:count+batch_num_nodes_i]    
                # divide by number of agents in the scene        
                b_s[i] = torch.sum(batch_loss_prop_reg, (0, 1)) / batch_num_nodes_i

                count += batch_num_nodes_i

            # sanity check
            assert batch_size == (i + 1), (batch_size, i+1)

            loss_prop_reg, best_modes = torch.min(b_s, dim=1)   
            loss_prop_reg = loss_prop_reg.mean()     

            # regression loss
            has_preds_mask = has_preds.unsqueeze(-1).unsqueeze(-1)
            has_preds_mask = has_preds_mask.expand(has_preds_mask.shape[0], has_preds_mask.shape[1], self.num_joint_modes, 1).bool().squeeze().to(self.device)
                        
            has_preds_mask = has_preds_mask[vehicle_mask]
            loc_pred = loc_pred[vehicle_mask]
            
            target = torch.stack([gt_locs] * self.num_joint_modes, dim=2).to(self.device)

            # Regression loss
            reg_loss = self.imitation_loss(loc_pred, target)

            # 0 out loss for the indices that don't have a ground-truth prediction.
            reg_loss = reg_loss * has_preds_mask

            b_s = torch.zeros((batch_size, self.num_joint_modes)).to(self.device)
            count = 0
            for i, batch_num_nodes_i in enumerate(graph.batch_num_nodes()):
                batch_num_nodes_i = batch_num_nodes_i.item()
                
                batch_reg_loss = reg_loss[count:count+batch_num_nodes_i]    
                # divide by number of agents in the scene        
                b_s[i] = torch.sum(batch_reg_loss, (0, 1)) / batch_num_nodes_i

                count += batch_num_nodes_i

            # sanity check
            assert batch_size == (i + 1)

            loss_reg = torch.min(b_s, dim=1)[0].mean()      

            loss = loss_reg + loss_prop_reg * self.proposal_coef
            return loss
