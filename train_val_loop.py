import torch
import pandas as pd
from tqdm import tqdm
from config import params
from data_prep import stratified_data_gen
from losses import DiceLoss
from unet import UNet
import logging
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO, handlers=[
                        logging.FileHandler("run.log"),
                        logging.StreamHandler()
                    ])

# DataLoader creation
train_loader, val_loader = stratified_data_gen(
    params['meta_path'], params['img_root'], params['train_pct'],
    params['batch_size'],  params['num_workers'], params['random_state'])

# Define model
model = UNet(3, 32, 1, padding=1).to(params['device'])

# Optimization params
criterion = DiceLoss()
learning_rate = params['learning_rate']
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch_train_losses = []
epoch_val_losses = []

# Main loop
for epoch in range(params['epochs']):
    logging.info(f"Epoch {epoch+1}/{params['epochs']}")

    model.train()
    train_losses = []
    for sample in tqdm(train_loader):
        # Train loop
        image, mask = sample['image'], sample['mask']
        image = image.to(params['device'], dtype=torch.float)
        mask = mask.to(params['device'], dtype=torch.float)

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
                image = image.to(params['device'], dtype=torch.float)
                mask = mask.to(params['device'], dtype=torch.float)
                preds = model.forward(image)
                loss = criterion(preds, mask)
                val_losses.append(loss.item())

    epoch_train_loss, epoch_val_loss = np.mean(
        train_losses), np.mean(val_losses)

    # log metrics
    logging.info(
        f"Train loss: {epoch_train_loss} | Train Accuracy: {100*(1 - epoch_train_loss)}%")
    logging.info(
        f"Validation loss: {epoch_val_loss} | Validation Accuracy: {100*(1-epoch_val_loss)}%")

    epoch_train_losses.append(epoch_train_loss)
    epoch_val_losses.append(epoch_val_loss)

    # download model
    if epoch_val_loss == min(epoch_val_losses):
        logging.info("Saving model...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
        }, join(params['model_dir'], f'/epoch{epoch}_val_{int(100*(1-epoch_val_loss))}'))

# Plot losses
fig, axs = plt.subplots(1, 2, figsize=(10, 7))
axs[0].plot(epoch_train_losses, label='train')
axs[0].plot(epoch_val_losses, label='val')
axs[0].legend()
axs[0].set(xlabel='Epoch', ylabel='Dice Loss', title='Loss vs. Epoch')

axs[1].plot(100*(1-epoch_train_losses), label='train')
axs[1].plot(100*(1-epoch_val_losses), label='val')
axs[1].legend()
axs[0].set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy vs. Epoch')

fig.savefig('loss_curves.png')
