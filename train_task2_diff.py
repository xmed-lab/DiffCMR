import os
from mpi4py import MPI

from improved_diffusion import dist_util, logger
# from datasets.city import load_data, create_dataset
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.utils import set_random_seed, set_random_seed_for_iterations
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data.distributed import DistributedSampler

import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from CMRxRecon import CMRxReconDataset
from torch.utils.tensorboard import SummaryWriter
from time import gmtime, strftime
current_time = strftime("%m%d_%H_%M", gmtime())
current_day = strftime("%m%d", gmtime())

############################
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
logdir = "./log/t2_04_128/"
trainpairfile = "/home/txiang/CMRxRecon/CMRxRecon_Repo/dataset/train_pair_file/Task2_acc_04_train_pair_file_npy_clean.txt"
############################


def main():
    dist_util.setup_dist()
    logger.configure(dir=logdir)
    arg_dict = model_and_diffusion_defaults()
    arg_dict["image_size"]=128
    # arg_dict["num_channels"]=128
    # arg_dict["rrdb_blocks"]=2
    print(arg_dict)
    model, diffusion = create_model_and_diffusion(**arg_dict)
    logger.log("creating model and diffusion...")
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    dataset = CMRxReconDataset(trainpairfile, transform=tsfm, length=-1)

    train_set = dataset
    print(len(train_set))

    # 3. Create data loaders
    loader_args = dict(num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=6, **loader_args, drop_last=True)

    def load_gen(loader):
        while True:
            yield from loader
    train_gen = load_gen(train_loader)

    TrainLoop(
            model=model,
            diffusion=diffusion,
            data=train_gen,
            batch_size=6,
            microbatch=-1,
            lr=1e-5,
            ema_rate="0.9999",
            log_interval=200,
            save_interval=2000,
            resume_checkpoint="",
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=schedule_sampler,
            weight_decay=0.0,
            lr_anneal_steps=0,
            clip_denoised=False,
            logger=logger,
            image_size=128,
            val_dataset=None,
            run_without_test=True,
        ).run_loop(start_print_iter=1000000)

if __name__ == "__main__":
    main()
