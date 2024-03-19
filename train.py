import numpy as np
import torch
from torch.utils.data import DataLoader

import pickle
import argparse
import os, sys
from pathlib import Path

from fjmp_dataloader_interaction import InteractionDataset
from fjmp_dataloader_argoverse2 import Argoverse2Dataset
from fjmp_modules import *
from fjmp_utils import *
from dag_utils import *
from fjmp_metrics import *
from fjmp import FJMP

import horovod.torch as hvd 
from torch.utils.data.distributed import DistributedSampler
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=['train', 'eval', 'eval_constant_velocity'], help='running mode : (train, eval, eval_constant_velocity)', default="train")
parser.add_argument("--dataset", choices=['interaction', 'argoverse2'], help='dataset : (interaction, argoverse2)', default="interaction")
parser.add_argument("--config_name", default="dev", help="a name to indicate the log path and model save path")
parser.add_argument("--num_edge_types", default=3, type=int, help='3 types: no-interaction, a-influences-b, b-influences-a')
parser.add_argument("--h_dim", default=128, type=int, help='dimension for the hidden layers of MLPs. Note that the GRU always has h_dim=256')
parser.add_argument("--num_joint_modes", default=6, type=int, help='number of scene-level modes')
parser.add_argument("--num_proposals", default=15, type=int, help='number of proposal modes')
parser.add_argument("--batch_size", default=16, type=int)
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
parser.add_argument("--focal_loss", action="store_true", help="use multiclass focal loss for relation header?")
parser.add_argument("--gamma", default=5, type=float, help="gamma parameter for focal loss.")
parser.add_argument("--weight_0", default=1., type=float, help="weight of class 0 for relation header.")
parser.add_argument("--weight_1", default=2., type=float, help="weight of class 1 for relation header.")
parser.add_argument("--weight_2", default=4., type=float, help="weight of class 2 for relation header.")
parser.add_argument("--teacher_forcing", action="store_true", help="use teacher forcing of influencer future predictions?")
parser.add_argument("--scheduled_sampling", action="store_true", help="use linear schedule curriculum for teacher forcing of influencer future predictions?")
parser.add_argument("--eval_training", action="store_true", help="run evaluation on training set?")
parser.add_argument("--supervise_vehicles", action="store_true", help="supervise only vehicles in loss function (for INTERACTION)?")
parser.add_argument("--train_all", action="store_true", help="train on both the train and validation sets?")
parser.add_argument("--no_agenttype_encoder", action="store_true", help="encode agent type in FJMP encoder? Only done for Argoverse 2 as INTERACTION only predicts vehicle trajectories.")

args = parser.parse_args()

GPU_START = args.gpu_start


hvd.init()
os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank() + GPU_START)
dev = 'cuda:{}'.format(0)
torch.cuda.set_device(0)

seed = hvd.rank()
set_seeds(seed)


if __name__ == '__main__':
    config = {}
    config["mode"] = args.mode 
    config["dataset"] = args.dataset 
    config["config_name"] = args.config_name 
    config["num_edge_types"] = args.num_edge_types
    config["h_dim"] = args.h_dim 
    config["num_joint_modes"] = args.num_joint_modes
    config["num_proposals"] = args.num_proposals
    config["max_epochs"] = args.max_epochs 
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
    config["focal_loss"] = args.focal_loss 
    config["gamma"] = args.gamma
    config["weight_0"] = args.weight_0
    config["weight_1"] = args.weight_1
    config["weight_2"] = args.weight_2
    config["teacher_forcing"] = args.teacher_forcing
    config["scheduled_sampling"] = args.scheduled_sampling 
    config["eval_training"] = args.eval_training
    config["supervise_vehicles"] = args.supervise_vehicles
    config["no_agenttype_encoder"] = args.no_agenttype_encoder 
    config["train_all"] = args.train_all

    config["log_path"].mkdir(exist_ok=True, parents=True)
    log = os.path.join(config["log_path"], "log")
    # write stdout to log file
    sys.stdout = Logger(log)

    if args.dataset == 'interaction':
        if config["train_all"]:
            config["num_train_samples"] = 47584 + 11794
        else:
            config["num_train_samples"] = 47584
        config["num_val_samples"] = 11794
        config["switch_lr_1"] = 40
        config["switch_lr_2"] = 48
        config["lr_step"] = 1/5
        config["input_size"] = 5
        config["prediction_steps"] = 30 
        config["observation_steps"] = 10
        # two agent types: "car", and "pedestrian/bicyclist"
        config["num_agenttypes"] = 2
        config['dataset_path'] = 'dataset_INTERACTION'
        config['tracks_train_reformatted'] = os.path.join(config['dataset_path'], 'train_reformatted')
        config['tracks_val_reformatted'] = os.path.join(config['dataset_path'], 'val_reformatted')
        config['num_scales'] = 4
        config["map2actor_dist"] = 20.0
        config["actor2actor_dist"] = 100.0
        config['maps'] = os.path.join(config['dataset_path'], 'maps')
        config['cross_dist'] = 10
        config['cross_angle'] = 1 * np.pi
        config["preprocess"] = True
        config["val_workers"] = 0
        config["workers"] = 0
        if config["train_all"]:
            config["preprocess_train"] = os.path.join(config['dataset_path'], 'preprocess', 'train_all_interaction')
        else:
            config["preprocess_train"] = os.path.join(config['dataset_path'], 'preprocess', 'train_interaction')
        config["preprocess_val"] = os.path.join(config['dataset_path'], 'preprocess', 'val_interaction')
        config['batch_size'] = args.batch_size

        if config['mode'] == 'train':
            dataset = InteractionDataset(config, train=True, train_all=config["train_all"])
            print("Loaded preprocessed training data.")

            train_sampler = DistributedSampler(
                dataset, num_replicas=hvd.size(), rank=hvd.rank()
            )
            train_loader = DataLoader(
                dataset,
                batch_size=config["batch_size"],
                num_workers=config["workers"],
                sampler=train_sampler,
                collate_fn=collate_fn,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
                drop_last=True,
            )

        dataset = InteractionDataset(config, train=False)  
        print("Loaded preprocessed validation data.")  
        val_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
        val_loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            num_workers=config["val_workers"],
            sampler=val_sampler,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    elif args.dataset == "argoverse2":
        if config["train_all"]:
            config["num_train_samples"] = 199908 + 24988
        else:
            config["num_train_samples"] = 199908
        config["num_val_samples"] = 24988
        config["switch_lr_1"] = 32
        config["switch_lr_2"] = 36
        config["lr_step"] = 1/10
        config["input_size"] = 5
        config["prediction_steps"] = 60
        config["observation_steps"] = 50
        config["num_agenttypes"] = 5
        config['dataset_path'] = 'dataset_AV2'
        config['files_train'] = os.path.join(config['dataset_path'], 'train')
        config['files_val'] = os.path.join(config['dataset_path'], 'val')
        config['num_scales'] = 6
        config["map2actor_dist"] = 10.0
        config["actor2actor_dist"] = 100.0
        config['cross_dist'] = 6
        config['cross_angle'] = 0.5 * np.pi
        config["preprocess"] = True
        config["val_workers"] = 0
        config["workers"] = 0
        if config["train_all"]:
            config["preprocess_train"] = os.path.join(config['dataset_path'], 'preprocess', 'train_all_argoverse2')
        else:
            config["preprocess_train"] = os.path.join(config['dataset_path'], 'preprocess', 'train_argoverse2')
        config["preprocess_val"] = os.path.join(config['dataset_path'], 'preprocess', 'val_argoverse2')
        config['batch_size'] = args.batch_size

        if config['mode'] == 'train':
            dataset = Argoverse2Dataset(config, train=True, train_all=config["train_all"])
            print("Loaded preprocessed training data.")

            train_sampler = DistributedSampler(
                dataset, num_replicas=hvd.size(), rank=hvd.rank()
            )
            train_loader = DataLoader(
                dataset,
                batch_size=config["batch_size"],
                num_workers=config["workers"],
                sampler=train_sampler,
                collate_fn=collate_fn,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
                drop_last=True,
            )

        dataset = Argoverse2Dataset(config, train=False)  
        print("Loaded preprocessed validation data.")  
        val_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
        val_loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            num_workers=config["val_workers"],
            sampler=val_sampler,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # Running training code
    if args.mode == 'train':
        model = FJMP(config)
        m = sum(p.numel() for p in model.parameters())
        print_("Command line arguments:")
        for it in sys.argv:
            print_(it)
        
        print_("Model: {} parameters".format(m))
        print_("Training model...")

        # save stage 1 config
        if model.two_stage_training and model.training_stage == 1:
            if hvd.rank() == 0:
                with open(os.path.join(config["log_path"], "config_stage_1.pkl"), "wb") as f:
                    pickle.dump(config, f)

        # load model for stage 1 and freeze weights
        if model.two_stage_training and model.training_stage == 2:
            with open(os.path.join(config["log_path"], "config_stage_1.pkl"), "rb") as f:
                config_stage_1 = pickle.load(f) 
            
            pretrained_relation_header = FJMP(config_stage_1)
            model.prepare_for_stage_2(pretrained_relation_header)
        
        # initialize optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=model.learning_rate)
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters()
        ) 
        
        starting_epoch = 1 
        val_best, ade_best, fde_best, val_edge_acc_best = np.inf, np.inf, np.inf, 0.
        # resume training from checkpoint
        if config["resume_training"]:
            if (not model.two_stage_training) or (model.two_stage_training and model.training_stage == 2):
                optimizer, starting_epoch, val_best, ade_best, fde_best = model.load_for_train(optimizer)
            else:
                optimizer, starting_epoch, val_edge_acc_best = model.load_for_train_stage_1(optimizer)

        # train model
        model._train(train_loader, val_loader, optimizer, starting_epoch, val_best, ade_best, fde_best, val_edge_acc_best)
    
    # Run evaluation code
    elif args.mode == 'eval':
        model = FJMP(config)
        m = sum(p.numel() for p in model.parameters())
        print_("Model: {} parameters".format(m))
        print_("Evaluating model...")

        # load model from stage 1 and freeze weights
        if model.two_stage_training and model.training_stage == 2:
            with open(os.path.join(config["log_path"], "config_stage_1.pkl"), "rb") as f:
                config_stage_1 = pickle.load(f) 

            pretrained_relation_header = FJMP(config_stage_1)
            model.prepare_for_stage_2(pretrained_relation_header)
        
        if model.two_stage_training and model.training_stage == 1:
            model.load_relation_header()
        else:
            model.load_for_eval()
        # evaluate model
        results = model._eval(val_loader, 1)
        print_("Model Results: ", "\t".join([f"{k}: {v}" if type(v) is np.ndarray else f"{k}: {v:.3f}" for k, v in results.items()]))
    
    # Evaluate FDE of interactive agents in validation set using constant velocity model
    elif args.mode == 'eval_constant_velocity':
        model = FJMP(config)
        m = sum(p.numel() for p in model.parameters())
        print_("Evaluating interactive agents on validation set with constant velocity model...")
        model._eval_constant_velocity(val_loader, config["max_epochs"])