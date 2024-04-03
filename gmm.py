import torch.nn as nn
import torch
import dgl

class GMMDecoder(nn.Module):
    def __init__(self, config):
        super(GMMDecoder, self).__init__()
        self.config = config
        self._future_len = self.config["prediction_steps"]
        self.multi_modal_query_embedding = nn.Embedding(self.config["num_joint_modes"], 128)
        self.gaussian = nn.Sequential(nn.Linear(128, 256), nn.ELU(), nn.Dropout(0.1), nn.Linear(256, self._future_len*4))
        self.score = nn.Sequential(nn.Linear(128, 64), nn.ELU(), nn.Dropout(0.1), nn.Linear(64, 1))
        self.register_buffer('modal', torch.arange(self.config["num_joint_modes"]).long())

    def forward(self, graph):

        graph_embeddings = [gs.ndata['xt_enc'].unsqueeze(1).tile((1, self.config["num_joint_modes"], 1)) for gs in dgl.unbatch(graph)]
        multi_modal_query = self.multi_modal_query_embedding(self.modal).unsqueeze(0)

        graph_embeddings = [graph_embeddings[i] + multi_modal_query.tile((graph_embeddings[i].shape[0], 1, 1)) for i in range(len(graph_embeddings))]
        graph_embeddings = torch.cat(graph_embeddings, dim=0)

        score = self.score(graph_embeddings).squeeze(-1)

        B, M, _ = graph_embeddings.shape

        res = self.gaussian(graph_embeddings).view(B, M, self._future_len, 4) # mu_x, mu_y, log_sig_x, log_sig_y

        pred_loc = res[..., :2] + graph.ndata["ctrs"].view(-1, 1, 1, 2)
        pred_loc = torch.matmul(pred_loc, graph.ndata["rot"].unsqueeze(1)) + graph.ndata["orig"].view(-1, 1, 1, 2)

        pred_vars = (2*res[..., 2:]).exp()
        pred_stds = torch.matmul(pred_vars, graph.ndata["rot"].unsqueeze(1).pow(2)).log() * 0.5

        res[..., :2] = pred_loc
        res[..., 2:] = pred_stds

        res = res.permute(0, 2, 1, 3)

        return res, score

