import torch
import torch.nn.functional as F
from torch import nn
import numpy as np 
import random
import sys, math
import matplotlib.pyplot as plt
from scipy import sparse

def accumulate_gradients(grads, named_parameters):
    if grads == {}:
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                grads[n] = p.grad.abs().mean()
    else:
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                grads[n] += p.grad.abs().mean()
    
    return grads

def plot_grad_flow(grads, epoch, log_path):
    path = log_path / 'gradients_{}.png'.format(epoch)
    plt.rc('xtick', labelsize=4)
    plt.figure(figsize=(20, 20), dpi=200)

    to_plot = list(grads.values())
    to_plot = [x.detach().cpu() for x in to_plot]
    
    plt.plot(to_plot, alpha=0.3, color="b")
    plt.hlines(0, 0, len(grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(grads), 1), list(grads.keys()), rotation="vertical")
    plt.xlim(xmin=0, xmax=len(grads))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(path)
    print("Plotted gradient flow")

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

def sync3(data, comm):
    data_list = comm.allgather(data)
    
    final_grads = {}
    for i in range(len(data_list)):
        if i == 0:
            for key in data_list[i].keys():
                final_grads[key] = data_list[i][key]
        else:
            for key in data_list[i].keys():
                final_grads[key] += data_list[i][key]
    
    for key in final_grads.keys():
        final_grads[key] /= len(data_list)
    
    return final_grads

def sync(data, config, comm):
    data_list = comm.allgather(data)
    
    FDE = 0
    ADE = 0
    SCR = 0
    SMR = 0
    SMR_AV2 = 0
    pFDE = 0
    pADE = 0
    n_scenarios = 0
    for i in range(len(data_list)):
        FDE += data_list[i]['FDE'] * data_list[i]['n_scenarios']
        ADE += data_list[i]['ADE'] * data_list[i]['n_scenarios']
        SCR += data_list[i]['SCR'] * data_list[i]['n_scenarios']
        SMR += data_list[i]['SMR'] * data_list[i]['n_scenarios']
        SMR_AV2 += data_list[i]['SMR_AV2'] * data_list[i]['n_scenarios']
        pFDE += data_list[i]['pFDE'] * data_list[i]['n_scenarios']
        pADE += data_list[i]['pADE'] * data_list[i]['n_scenarios']
        n_scenarios += data_list[i]['n_scenarios']
    
    FDE /= n_scenarios
    ADE /= n_scenarios
    SCR /= n_scenarios
    SMR /= n_scenarios
    SMR_AV2 /= n_scenarios
    pFDE /= n_scenarios
    pADE /= n_scenarios

    if config['learned_relation_header']:
        n_gpus = 0
        edge_acc = 0
        edge_acc_0 = 0
        edge_acc_1 = 0
        edge_acc_2 = 0
        proportion_no_edge = 0
        for i in range(len(data_list)):
            n_gpus += 1
            edge_acc += data_list[i]['Edge Accuracy']
            edge_acc_0 += data_list[i]['Edge Accuracy 0']
            edge_acc_1 += data_list[i]['Edge Accuracy 1']
            edge_acc_2 += data_list[i]['Edge Accuracy 2']
            proportion_no_edge += data_list[i]['Proportion No Edge']
        
        edge_acc /= n_gpus
        edge_acc_0 /= n_gpus
        edge_acc_1 /= n_gpus
        edge_acc_2 /= n_gpus
        proportion_no_edge /= n_gpus
    
    return_dict = {
        'FDE': FDE,
        'ADE': ADE,
        'pFDE': pFDE,
        'pADE': pADE,
        'SCR': SCR,
        'SMR': SMR,
        'SMR_AV2': SMR_AV2
    }

    if config["learned_relation_header"]:
        return_dict['E-Acc'] = edge_acc
        return_dict['E-Acc 0'] = edge_acc_0 
        return_dict['E-Acc 1'] = edge_acc_1 
        return_dict['E-Acc 2'] = edge_acc_2 
        return_dict['PropNoEdge'] = proportion_no_edge
    
    return return_dict

class Logger(object):
    def __init__(self, log):
        self.terminal = sys.stdout
        self.log = open(log, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

### FROM LANE_GCN
def graph_gather(graphs, config):
    batch_size = len(graphs)
    node_idcs = []
    count = 0
    counts = []

    for i in range(batch_size):
        counts.append(count)
        idcs = torch.arange(count, count + graphs[i]["num_nodes"]).to(
            graphs[i]["feats"].device
        )
        node_idcs.append(idcs)
        count = count + graphs[i]["num_nodes"]

    graph = dict()
    graph["idcs"] = node_idcs
    graph["ctrs"] = [x["ctrs"] for x in graphs]

    for key in ["feats"]:
        graph[key] = torch.cat([x[key] for x in graphs], 0)

    for k1 in ["pre", "suc"]:
        graph[k1] = []
        for i in range(min(len(graphs[0]["pre"]), config["num_scales"])):
            graph[k1].append(dict())
            for k2 in ["u", "v"]:
                graph[k1][i][k2] = torch.cat(
                    [graphs[j][k1][i][k2] + counts[j] for j in range(batch_size)], 0
                )

    for k1 in ["left", "right"]:
        graph[k1] = dict()
        for k2 in ["u", "v"]:
            temp = [graphs[i][k1][k2] + counts[i] for i in range(batch_size)]
            temp = [
                x if x.dim() > 0 else graph["pre"][0]["u"].new().resize_(0)
                for x in temp
            ]
            graph[k1][k2] = torch.cat(temp)
    
    return graph

### FROM LANE_GCN
def dilated_nbrs(nbr, num_nodes, num_scales):
    data = np.ones(len(nbr['u']), bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, num_scales):
        mat = mat * mat

        nbr = dict()
        coo = mat.tocoo()
        nbr['u'] = coo.row.astype(np.int64)
        nbr['v'] = coo.col.astype(np.int64)
        nbrs.append(nbr)
    return nbrs

### FROM LANE_GCN
def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data

### FROM LANE_GCN
def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data

### FROM LANE_GCN
def cat(batch):
    if torch.is_tensor(batch[0]):
        batch = [x.unsqueeze(0) for x in batch]
        return_batch = torch.cat(batch, 0)
    elif isinstance(batch[0], list) or isinstance(batch[0], tuple):
        batch = zip(*batch)
        return_batch = [cat(x) for x in batch]
    elif isinstance(batch[0], dict):
        return_batch = dict()
        for key in batch[0].keys():
            return_batch[key] = cat([x[key] for x in batch])
    else:
        return_batch = batch
    return return_batch

### FROM LANE_GCN
def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch

### FROM LANE_GCN
def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data

### FROM LANE_GCN
def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data

### FROM NRI
def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=0)
    return soft_max_1d.transpose(axis, 0)

### FROM CONTRASTIVE FUTURE TRAJECTORY PREDICTION
def estimate_constant_velocity(history, prediction_horizon, has_obs):
    history = history[has_obs == 1]
    length_history = history.shape[0]
    z_x = history[:, 0] # these are the observations x
    z_y = history[:, 1] # these are the observations y
    
    if length_history == 1:
        v_x = 0
        v_y = 0
    else:
        v_x = 0
        v_y = 0
        for index in range(length_history - 1):
            v_x += z_x[index + 1] - z_x[index]
            v_y += z_y[index + 1] - z_y[index]
        v_x = v_x / (length_history - 1) # v_x is the average velocity x
        v_y = v_y / (length_history - 1) # v_y is the average velocity y
    
    x_pred = z_x[-1] + v_x * prediction_horizon 
    y_pred = z_y[-1] + v_y * prediction_horizon 

    return x_pred, y_pred

def evaluate_fde(x_pred, y_pred, x, y):
    return math.sqrt((x_pred - x) ** 2 + (y_pred - y) ** 2)

class FocalLoss(nn.Module):
    
    def __init__(self, weight=None, 
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )

def sign_func(x):
    if x > 0:
        return 1.
    elif x < 0:
        return -1.
    else:
        return 0.

def process_data(data, config):
    num_actors = [len(x) for x in data['feats']]
    num_edges = [int(n * (n-1) / 2) for n in num_actors]

    # LaneGCN processing 
    # ctrs gets copied once for each agent in scene, whereas actor_ctrs only contains one per scene
    # same data, but different format so that it is compatible with LaneGCN L2A/A2A function     
    actor_ctrs = gpu(data["ctrs"])
    lane_graph = graph_gather(to_long(gpu(data["graph"])), config)
    # unique index assigned to each scene
    scene_idxs = torch.Tensor([idx for idx in data['idx']])

    graph = data["graph"]

    world_locs = [x for x in data['feat_locs']]
    world_locs = torch.cat(world_locs, 0)

    has_obs = [x for x in data['has_obss']]
    has_obs = torch.cat(has_obs, 0)

    ig_labels = [x for x in data['ig_labels_{}'.format(config['ig'])]]
    ig_labels = torch.cat(ig_labels, 0)

    if config["dataset"] == "argoverse2":
        agentcategories = [x for x in data['feat_agentcategories']]
        # we know the agent category exists at the present timestep
        agentcategories = torch.cat(agentcategories, 0)[:, config["observation_steps"] - 1, 0]
        # we consider scored+focal tracks for evaluation in Argoverse 2
        is_scored = agentcategories >= 2

    locs = [x for x in data['feats']]
    locs = torch.cat(locs, 0)

    vels = [x for x in data['feat_vels']]
    vels = torch.cat(vels, 0)

    psirads = [x for x in data['feat_psirads']]
    psirads = torch.cat(psirads, 0)

    gt_psirads = [x for x in data['gt_psirads']]
    gt_psirads = torch.cat(gt_psirads, 0)

    gt_vels = [x for x in data['gt_vels']]
    gt_vels = torch.cat(gt_vels, 0)

    agenttypes = [x for x in data['feat_agenttypes']]
    agenttypes = torch.cat(agenttypes, 0)[:, config["observation_steps"] - 1, 0]
    agenttypes = torch.nn.functional.one_hot(agenttypes.long(), config["num_agenttypes"])

    # shape information is only available in INTERACTION dataset
    if config["dataset"] == "interaction":
        shapes = [x for x in data['feat_shapes']]
        shapes = torch.cat(shapes, 0)

    feats = torch.cat([locs, vels, psirads], dim=2)

    ctrs = [x for x in data['ctrs']]
    ctrs = torch.cat(ctrs, 0)

    orig = [x.view(1, 2) for j, x in enumerate(data['orig']) for i in range(num_actors[j])]
    orig = torch.cat(orig, 0)

    rot = [x.view(1, 2, 2) for j, x in enumerate(data['rot']) for i in range(num_actors[j])]
    rot = torch.cat(rot, 0)

    theta = torch.Tensor([x for j, x in enumerate(data['theta']) for i in range(num_actors[j])])

    gt_locs = [x for x in data['gt_preds']]
    gt_locs = torch.cat(gt_locs, 0)

    has_preds = [x for x in data['has_preds']]
    has_preds = torch.cat(has_preds, 0)

    # does a ground-truth waypoint exist at the last timestep?
    has_last = has_preds[:, -1] == 1
    
    batch_idxs = []
    batch_idxs_edges = []
    actor_idcs = []
    sceneidx_to_batchidx_mapping = {}
    count_batchidx = 0
    count = 0
    for i in range(len(num_actors)):            
        batch_idxs.append(torch.ones(num_actors[i]) * count_batchidx)
        batch_idxs_edges.append(torch.ones(num_edges[i]) * count_batchidx)
        sceneidx_to_batchidx_mapping[int(scene_idxs[i].item())] = count_batchidx
        idcs = torch.arange(count, count + num_actors[i]).to(locs.device)
        actor_idcs.append(idcs)
        
        count_batchidx += 1
        count += num_actors[i]
    
    batch_idxs = torch.cat(batch_idxs).to(locs.device)
    batch_idxs_edges = torch.cat(batch_idxs_edges).to(locs.device)
    batch_size = torch.unique(batch_idxs).shape[0]

    ig_labels_metrics = [x for x in data['ig_labels_sparse']]
    ig_labels_metrics = torch.cat(ig_labels_metrics, 0)

    # 1 if agent has out-or-ingoing edge in ground-truth sparse interaction graph
    # These are the agents we use to evaluate interactive metrics
    is_connected = torch.zeros(locs.shape[0])
    count = 0
    offset = 0
    for k in range(len(num_actors)):
        N = num_actors[k]
        for i in range(N):
            for j in range(N):
                if i >= j:
                    continue 
                
                # either an influencer or reactor in some DAG.
                if ig_labels_metrics[count] > 0:                      

                    is_connected[offset + i] += 1
                    is_connected[offset + j] += 1 

                count += 1
        offset += N

    is_connected = is_connected > 0     

    assert count == ig_labels_metrics.shape[0]

    dd = {
        'batch_size': batch_size,
        'batch_idxs': batch_idxs,
        'batch_idxs_edges': batch_idxs_edges, 
        'actor_idcs': actor_idcs,
        'actor_ctrs': actor_ctrs,
        'lane_graph': lane_graph,
        'feats': feats,
        'feat_psirads': psirads,
        'ctrs': ctrs,
        'orig': orig,
        'rot': rot,
        'theta': theta,
        'gt_locs': gt_locs,
        'has_preds': has_preds,
        'scene_idxs': scene_idxs,
        'sceneidx_to_batchidx_mapping': sceneidx_to_batchidx_mapping,
        'ig_labels': ig_labels,
        'gt_psirads': gt_psirads,
        'gt_vels': gt_vels,
        'agenttypes': agenttypes,
        'world_locs': world_locs,
        'has_obs': has_obs,
        'has_last': has_last,
        'graph': graph,
        'is_connected': is_connected
    }

    if config["dataset"] == "interaction":
        dd['shapes'] = shapes

    elif config["dataset"] == "argoverse2":
        dd['is_scored'] = is_scored

    # dd = data-dictionary
    return dd
