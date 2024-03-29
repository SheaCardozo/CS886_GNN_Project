from torch import nn
import torch
from dag_utils import build_graph
from graph_transformer import GTBackbone
from relation_header import FJMPRelationHeader
from utils import *
from lanegcn_modules import *
from modules import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
dev = 'cuda:{}'.format(0)


class GNNPipeline(nn.Module):
    def __init__(self, config):
        super(GNNPipeline, self).__init__()
        self.config = config
        self.model_path = config["model_path"]
        self.observation_steps = config["observation_steps"]

        self.build()

    def build(self):

        model_dict = torch.load(self.model_path, map_location=dev)

        self.relation_feature_encoder = FJMPFeatureEncoder(self.config).to(dev).load_state_dict(model_dict['feature_state_dict'])
        #self.relation_aux_prop_decoder = FJMPTrajectoryProposalDecoder(self.config).to(dev).load_state_dict(model_dict['proposal_state_dict'])
        self.relation_header = FJMPRelationHeader(self.config).to(dev).load_state_dict(model_dict['relation_state_dict'])

        for param in self.relation_feature_encoder.parameters():
            param.requires_grad = False

        for param in self.relation_header.parameters():
            param.requires_grad = False

        self.feature_encoder = FJMPFeatureEncoder(self.config).to(dev)
        self.aux_prop_decoder = FJMPTrajectoryProposalDecoder(self.config).to(dev)
        self.gnn_backbone = GTBackbone(self.config).to(dev)
        self.gmm_decoder = GMMDecoder(self.config).to(dev)


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


    def forward(self, scenario_data, training=False):

        dd = process_data(scenario_data, self.config)

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
        dgl_graph = build_graph(dgl_graph, self.config)

        dgl_graph, aux_proposals = self.aux_prop_decoder(dgl_graph, dd['actor_ctrs'])

        dgl_graph = self.gnn_backbone(dgl_graph)

        gmm_params = self.gmm_decoder(dgl_graph)

        if training:
            return gmm_params, aux_proposals

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
