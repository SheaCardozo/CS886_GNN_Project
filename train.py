import numpy as np
import torch

import argparse
import os, sys
from pathlib import Path

from gnn import GNNPipeline
from modules import *
from utils import *
from dag_utils import *
from metrics import *
from get_dataloaders import get_dataloaders

import tqdm

from accelerate import Accelerator, DistributedDataParallelKwargs

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=['train'], help='running mode : (train)', default="train")
parser.add_argument("--dataset", choices=['interaction', 'argoverse2'], help='dataset : (interaction, argoverse2)', default="interaction")
parser.add_argument("--config_name", default="test", help="a name to indicate the log path and model save path")
parser.add_argument("--num_edge_types", default=3, type=int, help='3 types: no-interaction, a-influences-b, b-influences-a')
parser.add_argument("--h_dim", default=128, type=int, help='dimension for the hidden layers of MLPs. Note that the GRU always has h_dim=256')
parser.add_argument("--num_joint_modes", default=6, type=int, help='number of scene-level modes')
parser.add_argument("--num_proposals", default=15, type=int, help='number of proposal modes')
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_epochs", default=50, type=int, help='maximum number of epochs')
parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
parser.add_argument("--decoder", choices=['dagnn', 'lanegcn'], help='decoder architecture : (dagnn, lanegcn)', default="dagnn")
parser.add_argument("--num_heads", default=1, type=int, help='number of heads in multi-head attention for decoder attention.')
parser.add_argument("--learned_relation_header", action="store_true", help="if true, network learns+predicts interaction graph with interaction graph predictor. Otherwise, ground-truth pseudolabels are used.")
parser.add_argument("--gpu_start", default=0, type=int, help='gpu device i, where training will occupy gpu device i,i+1,...,i+n_gpus-1')
parser.add_argument("--n_mapnet_layers", default=2, type=int, help='number of MapNet blocks')
parser.add_argument("--n_l2a_layers", default=2, type=int, help='number of L2A attention blocks')
parser.add_argument("--n_a2a_layers", default=2, type=int, help='number of A2A attention blocks')
parser.add_argument("--resume_training", action="store_true", help="continue training from checkpoint")
parser.add_argument("--proposal_coef", default=1, type=float, help="coefficient for proposal losses")
parser.add_argument("--rel_coef", default=100, type=float, help="coefficient for interaction graph prediction losses.")
parser.add_argument("--proposal_header", action="store_true", help="add proposal multitask training objective?")
parser.add_argument("--two_stage_training", action="store_true", help="train relation predictor first?")
parser.add_argument("--training_stage", default=1, type=int, help='1 or 2. Which training stage in 2 stage training?')
parser.add_argument("--ig", choices=['sparse', 'dense', 'm2i'], help='which interaction graph pseudolabels to use', default="sparse")
parser.add_argument("--teacher_forcing", action="store_true", help="use teacher forcing of influencer future predictions?")
parser.add_argument("--scheduled_sampling", action="store_true", help="use linear schedule curriculum for teacher forcing of influencer future predictions?")
parser.add_argument("--eval_training", action="store_true", help="run evaluation on training set?")
parser.add_argument("--supervise_vehicles", action="store_true", help="supervise only vehicles in loss function (for INTERACTION)?")
parser.add_argument("--train_all", action="store_true", help="train on both the train and validation sets?")
parser.add_argument("--no_agenttype_encoder", action="store_true", help="encode agent type in FJMP encoder? Only done for Argoverse 2 as INTERACTION only predicts vehicle trajectories.")
parser.add_argument("--model_path", default="/home/sacardoz/FJMP/logs/real/best_models.pt", type=str, help='Path to Stage 1 Models')

args = parser.parse_args()

def val(model, config, val_loader):

    model.eval()

    loc_preds, gt_locs_all, agenttypes_all, has_last_all, has_preds_all, batch_idxs_all = [], [], [], [], [], []

    tot = 0
    with torch.no_grad():
        for data in tqdm.tqdm(val_loader, total=len(val_loader), desc=f"Validation"):
            dd = process_data(data, config)

            gmm_params = model(dd, train=False) 
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
    eval_agent_mask = np.concatenate(agenttypes_all, axis=0)[:, 1].astype(bool)

    mask = has_last_mask * eval_agent_mask

    gt_locs_masked = gt_locs_all[mask]
    has_preds_masked = has_preds_all[mask].astype(bool)
    batch_idxs_masked = batch_idxs_all[mask]
    loc_preds_masked = loc_preds[mask]

    n_scenarios = np.unique(batch_idxs_masked).shape[0]
    scenarios = np.unique(batch_idxs_masked).astype(int)

    has_preds_all_mask = np.reshape(has_preds_masked, has_preds_masked.shape + (1,))
    has_preds_all_mask = np.broadcast_to(has_preds_all_mask, has_preds_masked.shape[:2] + (config["num_joint_modes"],))  

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

    FDE = torch.Tensor([mean_FDE.min(1).mean()]).to(accelerator.device)
    ADE = torch.Tensor([mean_ADE.min(1).mean()]).to(accelerator.device)
    n_scenarios = torch.Tensor([n_scenarios]).to(accelerator.device)

    data_list = {
        "FDE": accelerator.gather(FDE),
        "ADE": accelerator.gather(ADE),
        "n_scenarios": accelerator.gather(n_scenarios)}

    FDE = 0
    ADE = 0
    n_scenarios = 0

    for iade, ifde, iscens in zip(data_list["ADE"], data_list["FDE"], data_list["n_scenarios"]):
        FDE += ifde * iscens
        ADE += iade * iscens
        n_scenarios += iscens

    FDE /= n_scenarios
    ADE /= n_scenarios

    results = {
        'FDE': FDE,
        'ADE': ADE,
    }

    model.train()

    return results

if __name__ == '__main__':
    config = {}
    config["mode"] = args.mode 
    config["dataset"] = args.dataset 
    config["config_name"] = args.config_name 
    config["num_edge_types"] = args.num_edge_types
    config["h_dim"] = args.h_dim 
    config["num_joint_modes"] = args.num_joint_modes
    config["num_proposals"] = args.num_proposals
    config["max_epochs"] = 100 
    config["log_path"] = Path('./logs') / config["config_name"]
    config["lr"] = args.lr 
    config["decoder"] = args.decoder
    config["num_heads"] = args.num_heads
    config["learned_relation_header"] = args.learned_relation_header
    config["n_mapnet_layers"] = args.n_mapnet_layers 
    config["n_l2a_layers"] = args.n_l2a_layers
    config["n_a2a_layers"] = args.n_a2a_layers
    config["resume_training"] = args.resume_training
    config["proposal_coef"] = args.proposal_coef
    config["rel_coef"] = args.rel_coef
    config["proposal_header"] = args.proposal_header
    config["two_stage_training"] = args.two_stage_training
    config["training_stage"] = args.training_stage
    config["ig"] = args.ig
    config["teacher_forcing"] = args.teacher_forcing
    config["scheduled_sampling"] = args.scheduled_sampling 
    config["eval_training"] = args.eval_training
    config["supervise_vehicles"] = args.supervise_vehicles
    config["no_agenttype_encoder"] = args.no_agenttype_encoder 
    config["train_all"] = args.train_all
    config["model_path"] = args.model_path

    config["log_path"].mkdir(exist_ok=True, parents=True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    log = os.path.join(config["log_path"], "log")
    # write stdout to log file
    sys.stdout = Logger(log)

    train_loader, val_loader = get_dataloaders(args, config)

    # Running training code
    model = GNNPipeline(config, device=accelerator.device)

    if accelerator.is_main_process:
        m = sum(p.numel() for p in model.parameters())
        
        print("Model: {} parameters".format(m))
        print("Training model...")

    # initialize optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold=0.0001, threshold_mode='rel')

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    val_best, ade_best, fde_best, val_edge_acc_best = np.inf, np.inf, np.inf, 0.

    for epoch in range(config["max_epochs"]):
        epoch_loss = torch.Tensor([0]).to(accelerator.device)
            
        model.train()

        for i, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            dd = process_data(data, config)

            with accelerator.autocast():
                loss = model(dd, train=True)
                epoch_loss += loss

                accelerator.backward(loss)

            optimizer.step()

        epoch_loss = accelerator.gather(epoch_loss)
            
        if accelerator.is_main_process:
            print(f"Training Epoch: {epoch}, lr={optimizer.param_groups[0]['lr']}, epoch_loss={epoch_loss.mean().item() / len(train_loader)}")
                
        val_eval_results = val(model, config, val_loader)

        if accelerator.is_main_process:
            #lr_scheduler.step(metrics=(val_eval_results["FDE"] + val_eval_results["ADE"]) / 2)
            print("Epoch {} validation-set results: ".format(epoch), "\t".join([f"{k}: {v}" if type(v) is np.ndarray else f"{k}: {v:.3f}" for k, v in val_eval_results.items()]))

            if (val_eval_results["FDE"] + val_eval_results["ADE"]) < val_best:
                val_best = val_eval_results["FDE"] + val_eval_results["ADE"]
                ade_best = val_eval_results["ADE"]
                fde_best = val_eval_results["FDE"]

                print("Validation FDE+ADE improved. Saving model. ")

                accelerator.unwrap_model(model).save(epoch, optimizer, val_best, ade_best, fde_best)    
                print("Best loss: {:.4f}".format(val_best), "Best ADE: {:.3f}".format(ade_best), "Best FDE: {:.3f}".format(fde_best))
