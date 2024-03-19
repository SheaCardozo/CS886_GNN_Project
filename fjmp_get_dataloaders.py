import numpy as np 
import horovod.torch as hvd 
import os

from torch.utils.data import DataLoader
from fjmp_dataloader_interaction import InteractionDataset
from fjmp_dataloader_argoverse2 import Argoverse2Dataset
from torch.utils.data.distributed import DistributedSampler

from fjmp_utils import *

def get_dataloaders(args, config):
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

    if config['mode'] != 'train':
        train_loader = None
        
    return train_loader, val_loader