import numpy as np
import torch

import pickle
import argparse
import os, sys
from pathlib import Path

from modules import *
from utils import *
from dag_utils import *
from metrics import *
from get_dataloaders import get_dataloaders
from relation_header import FJMPHeaderEncoderTrainer

import tqdm

from accelerate import Accelerator, DistributedDataParallelKwargs

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=['interaction', 'argoverse2'], help='dataset : (interaction, argoverse2)', default="interaction")
parser.add_argument("--config_name", default="dev", help="a name to indicate the log path and model save path")
parser.add_argument("--h_dim", default=128, type=int, help='dimension for the hidden layers of MLPs. Note that the GRU always has h_dim=256')
parser.add_argument("--num_edge_types", default=3, type=int, help='3 types: no-interaction, a-influences-b, b-influences-a')
parser.add_argument("--ig", choices=['sparse', 'dense', 'm2i'], help='which interaction graph pseudolabels to use', default="sparse")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--max_epochs", default=50, type=int, help='maximum number of epochs')
parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
parser.add_argument("--gpu_start", default=0, type=int, help='gpu device i, where training will occupy gpu device i,i+1,...,i+n_gpus-1')
parser.add_argument("--focal_loss", action="store_true", help="use multiclass focal loss for relation header?")
parser.add_argument("--gamma", default=5, type=float, help="gamma parameter for focal loss.")
parser.add_argument("--weight_0", default=1., type=float, help="weight of class 0 for relation header.")
parser.add_argument("--weight_1", default=2., type=float, help="weight of class 1 for relation header.")
parser.add_argument("--weight_2", default=4., type=float, help="weight of class 2 for relation header.")
parser.add_argument("--train_all", action="store_true", help="train on both the train and validation sets?")
parser.add_argument("--no_agenttype_encoder", action="store_true", help="encode agent type in FJMP encoder? Only done for Argoverse 2 as INTERACTION only predicts vehicle trajectories.")
parser.add_argument("--n_mapnet_layers", default=2, type=int, help='number of MapNet blocks')
parser.add_argument("--n_l2a_layers", default=2, type=int, help='number of L2A attention blocks')
parser.add_argument("--n_a2a_layers", default=2, type=int, help='number of A2A attention blocks')
parser.add_argument("--num_proposals", default=15, type=int, help='number of proposal modes')
parser.add_argument("--rel_coef", default=100, type=float, help="coefficient for interaction graph prediction losses.")
parser.add_argument("--supervise_vehicles", action="store_true", help="supervise only vehicles in loss function (for INTERACTION)?")

args = parser.parse_args()

GPU_START = args.gpu_start

def val(model, config, val_loader):

    model.eval()

    ig_preds = []
    ig_labels_all = []            

    with torch.no_grad():
        for data in tqdm.tqdm(val_loader, total=len(val_loader), desc=f"Validation"):
            dd = process_data(data, config)

            relations_preds = model(dd, train=False)
            edge_probs = my_softmax(relations_preds, -1)

            ig_labels_all.append(dd["ig_labels"].detach().cpu())                                            
            ig_preds.append(edge_probs.detach().cpu())

    results_ig_preds = torch.concatenate(ig_preds, axis=0)
    results_ig_labels_all = torch.concatenate(ig_labels_all, axis=0)    

    ig_preds = torch.argmax(results_ig_preds, axis=1)
    relation_accuracy = torch.mean(torch.where(ig_preds == results_ig_labels_all, 1., 0.)).to(model.device)

    edge_mask_0 = results_ig_labels_all == 0
    edge_mask_1 = results_ig_labels_all == 1
    edge_mask_2 = results_ig_labels_all == 2

    edge_accuracy_0 = torch.mean(torch.where(ig_preds[edge_mask_0] == 0, 1., 0.)).to(model.device)
    edge_accuracy_1 = torch.mean(torch.where(ig_preds[edge_mask_1] == 1, 1., 0.)).to(model.device)
    edge_accuracy_2 = torch.mean(torch.where(ig_preds[edge_mask_2] == 2, 1., 0.)).to(model.device)

    model.train()

    return accelerator.gather(relation_accuracy).mean().item(), accelerator.gather(edge_accuracy_0).mean().item(), accelerator.gather(edge_accuracy_1).mean().item(), accelerator.gather(edge_accuracy_2).mean().item()


if __name__ == '__main__':
    config = {}
    config["dataset"] = args.dataset
    config["config_name"] = args.config_name
    config["h_dim"] = args.h_dim 
    config["max_epochs"] = args.max_epochs 
    config["log_path"] = Path('./logs') / config["config_name"]
    config["lr"] = args.lr 

    config["gamma"] = args.gamma
    config["weight_0"] = args.weight_0
    config["weight_1"] = args.weight_1
    config["weight_2"] = args.weight_2

    config["log_path"].mkdir(exist_ok=True, parents=True)

    config["train_all"] = args.train_all
    config["mode"] = "train"
    config["num_edge_types"] = args.num_edge_types
    config["ig"] = args.ig

    config["no_agenttype_encoder"] = args.no_agenttype_encoder
    config["n_mapnet_layers"] = args.n_mapnet_layers
    config["n_l2a_layers"] = args.n_l2a_layers
    config["n_a2a_layers"] = args.n_a2a_layers

    config["num_proposals"] = args.num_proposals
    config["rel_coef"] = args.rel_coef

    config["supervise_vehicles"] = args.supervise_vehicles

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    log = os.path.join(config["log_path"], "log")
    # write stdout to log file
    sys.stdout = Logger(log)

    train_loader, val_loader = get_dataloaders(args, config)

    # Running training code
    model = FJMPHeaderEncoderTrainer(config, device=accelerator.device)

    if accelerator.is_main_process:
        m = sum(p.numel() for p in model.parameters())
        
        print("Model: {} parameters".format(m))
        print("Training model...")

    # initialize optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)

    val_edge_acc_best = 0.

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
            print(f"Training Epoch: {epoch}, lr={optimizer.param_groups[0]['lr']}, epoch_loss={epoch_loss.mean().item()}")
                
        edge_acc, ea0, ea1, ea2 = val(model, config, val_loader)

        lr_scheduler.step(metrics=edge_acc)

        if accelerator.is_main_process:

            val_dict = {"val acc": edge_acc, "val acc 0": ea0, "val acc 1": ea1, "val acc 2": ea2}
            print("Epoch {} validation-set results: ".format(epoch), "\t".join([f"{k}: {v}" if type(v) is np.ndarray else f"{k}: {v:.3f}" for k, v in val_dict.items()]))

            if edge_acc > val_edge_acc_best:
                print("Validation Edge Accuracy improved.")  
                val_edge_acc_best = edge_acc  
            
                print("Saving relation header")
                accelerator.unwrap_model(model).save_models(epoch, val_edge_acc_best)    
                print("Best validation edge accuracy: {:.4f}".format(val_edge_acc_best))     
