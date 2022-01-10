import argparse
import logging
import os
from datetime import time, timedelta
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from config.config import params
from data_tools.data_prep import stratified_data_gen
from loss_functions.dice_losses import DiceLoss
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from unet import UNet


def setup(params):
    """"
    CLI parser is used instead of config file here, due to the possibility of
    several unique specifications across nodes. Config will only hold main addr/port.
    Thanks to yangkky and shen li for their great learning materials on DDP.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', default=1, type=int)  # total nodes
    parser.add_argument('--gpn', default=1, type=int)  # gpus/node
    parser.add_argument('--nrank', default=0, type=int)  # node rank
    parser.add_argument('--epochs', default=10, type=int)  # num epochs
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA supported GPUs must be available for DDP"
    args.wsize = args.nodes * args.gpn  # nodes * gpu/node = #gpus
    os.environ['MASTER_ADDR'] = params['master_addr']
    os.environ['MASTER_PORT'] = params['master_port']

    mp.spawn(train_val_loop, nprocs=args.gpn,
             args=(args,))  # ensure args is passed as tuple


def train_val_loop(grank, args):  # mp calls function(i, args), i = proc idx
    """
    The training and validation loop is packaged into a function to delay
    execution until argparser fetches command line arguments
    """

    proc_rank = grank + args.gpn * args.nrank  # gpu rank out of all nodes

    # Setup logging
    logging.basicConfig(format=f'{proc_rank}: %(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO, handlers=[
                            logging.FileHandler(params['log_path']),
                            logging.StreamHandler()])

    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=args.wsize, rank=proc_rank, timeout=timedelta(minutes=30))

    torch.manual_seed(params['random_state'])  # ensure uniformitiy among procs

    # DataLoader generation
    train_dset, val_dset = stratified_data_gen(
        params['meta_path'], params['img_root'], params['train_pct'],
        params['batch_size'],  params['num_workers'], params['random_state'], dset_only=True)

    tsampler = DistributedSampler(
        train_dset, num_replicas=args.wsize, rank=proc_rank)
    vsampler = DistributedSampler(
        val_dset, num_replicas=args.wsize, rank=proc_rank)

    train_loader = DataLoader(
        dataset=train_dset, batch_size=params['batch_size'], shuffle=False,
        num_workers=0, pin_memory=True, sampler=tsampler)  # sampler instead of shuffle

    val_loader = DataLoader(
        dataset=val_dset, batch_size=params['batch_size'], shuffle=False,
        num_workers=0, pin_memory=True, sampler=vsampler)

    # Define model
    torch.cuda.device(grank)

    # Use .cuda explicitly for error catching
    model = UNet(3, 32, 1, padding=1, bn=False).cuda(grank)

    # DDP model wrapping
    model = DistributedDataParallel(model, device_ids=[grank])

    # Optimization params
    criterion = DiceLoss()
    learning_rate = params['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_train_losses = []
    epoch_val_losses = []

    # Main loop
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")

        model.train()
        train_losses = []
        for sample in tqdm(train_loader):
            # Train loop
            image, mask = sample['image'], sample['mask']
            image = image.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            preds = model.forward(image)
            loss = criterion(preds, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        else:
            # Val loop
            model.eval()
            val_losses = []
            with torch.no_grad():
                for sample in tqdm(val_loader):
                    image, mask = sample['image'], sample['mask']
                    image = image.cuda(non_blocking=True)
                    mask = mask.cuda(non_blocking=True)
                    preds = model.forward(image)
                    loss = criterion(preds, mask)
                    val_losses.append(loss.item())

        epoch_train_loss, epoch_val_loss = np.mean(
            train_losses), np.mean(val_losses)

        # log metrics
        logging.info(
            f"Train loss: {epoch_train_loss} | Train Accuracy: {100*(1 - epoch_train_loss)}")
        logging.info(
            f"Validation loss: {epoch_val_loss} | Validation Accuracy: {100*(1-epoch_val_loss)}")

        epoch_train_losses.append(epoch_train_loss)
        epoch_val_losses.append(epoch_val_loss)

        # download model
        if grank == 0:
            if epoch_val_loss == min(epoch_val_losses):
                logging.info("Saving model...")
                torch.save({
                    'proc_rank': proc_rank,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                }, join(params['model_dir'], f'/epoch{epoch}_val_{int(100*(1-epoch_val_loss))}'))

    if grank == 0:
        with torch.no_grad():
            # Plot losses
            fig, axs = plt.subplots(1, 2, figsize=(10, 7))
            axs[0].plot(epoch_train_losses, label='train')
            axs[0].plot(epoch_val_losses, label='val')
            axs[0].legend()
            axs[0].set(xlabel='Epoch', ylabel='Dice Loss',
                       title='Loss vs. Epoch')

            axs[1].plot(100*(1-epoch_train_losses), label='train')
            axs[1].plot(100*(1-epoch_val_losses), label='val')
            axs[1].legend()
            axs[0].set(xlabel='Epoch', ylabel='Accuracy',
                       title='Accuracy vs. Epoch')

            fig.savefig(f'images/loss_curves.png')


if __name__ == "__main__":
    setup(params)
