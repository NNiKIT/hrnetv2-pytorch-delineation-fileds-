import logging
import argparse
from tqdm import tqdm
import os
import sys

import torch
import torch.nn as nn
from torch import optim

from torch.utils.data import DataLoader, random_split
from core.dataset import CustomDataset
from core.hrnetv2 import initialize_weights, HRNetv2
from core.loss import DiceLoss, DiceBCELoss, BCELoss
from core.eval import eval_dice

from torch.utils.tensorboard import SummaryWriter

import neptune.new as neptune
from neptune.new.types import File

epochs = 50
batch_size = 15
lr = 0.0005
val_percent = 0.1

checkpoints_to_save = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
31, 32, 33, 34, 35, 36, 37, 38, 39 ,40,
41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_boundaries = 'data/boundaries/'
dir_checkpoint = 'checkpoints/'

loss_fn = DiceBCELoss()
optimizer = optim.Adam

# NEPTUNE.AI TRACKING
track_hparams_dict = {'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 'val_percent': val_percent}

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--resume', '-r', default='None',
                        metavar='resume training from checkpoint', required=False,
                        help="Specify checkpoint path to load and resume trainig from")

    return parser.parse_args()

def train_net(net,
              loss_fn,
              optimizer,
              device,
              epochs,
              batch_size,
              lr,
              dir_img,
              dir_mask,
              dir_boundaries,
              dir_checkpoint,
              val_percent,
              checkpoints_to_save,
              last_epoch=None):

    dataset = CustomDataset(dir_img, dir_mask, dir_boundaries)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_OPT{optimizer}')

    iteration = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoint to save:     {checkpoints_to_save}
        Device:          {device.type}
    ''')

    optim = optimizer(net.parameters(), weight_decay=0.0005, lr=lr)

    if last_epoch:
        train_epoch = last_epoch
    else:
        train_epoch = 0
    while train_epoch < epochs:
        net.train()
        epoch_loss_tot = 0
        epoch_loss_mask = 0
        epoch_loss_boundary = 0
        
        with tqdm(total=n_train, desc=f'Epoch {train_epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['transformed_image']
                true_masks_and_boundaries = batch['transformed_mask_and_boundary']
                true_masks = true_masks_and_boundaries[:, 0, :, :]
                true_boundaries = true_masks_and_boundaries[:, 1, :, :]

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                true_boundaries = true_boundaries.to(device=device, dtype=torch.float32)

                preds_mask_and_boundary = net(imgs)

                loss_mask = loss_fn(preds_mask_and_boundary[:,0,:,:], true_masks)
                loss_boundary = loss_fn(preds_mask_and_boundary[:,1,:,:], true_boundaries)
                loss_tot = loss_mask + loss_boundary

                epoch_loss_mask += loss_mask.item()
                epoch_loss_boundary += loss_boundary.item()
                epoch_loss_tot += loss_tot.item()
                
                writer.add_scalar('loss_tot/train', loss_tot.item(), iteration)
                pbar.set_postfix(**{'loss tot (batch)': loss_tot.item()})
                writer.add_scalar('loss_mask/train', loss_mask.item(), iteration)
                writer.add_scalar('loss_boundary/train', loss_boundary.item(), iteration)

                optim.zero_grad()
                loss_tot.backward()
                #nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optim.step()

                pbar.update(imgs.shape[0])
                iteration += 1

                mask_score, boundary_score = eval_dice(net, val_loader, device)

                logging.info('/n')
                logging.info('Iteration: {}'.format(iteration))
                logging.info('Mask Dice eval: {}'.format(mask_score))
                logging.info('Boundary Dice eval: {}'.format(boundary_score))

                # NEPTUNE.AI TRACKING
                # Log batch tot loss
                run["training/batch/loss_tot"].log(loss_tot)
                # Log batch mask loss
                run["training/batch/loss_mask"].log(loss_mask)
                # Log batch boundary loss
                run["training/batch/loss_boundary"].log(loss_boundary)
                # Log batch mask dice
                run["training/batch/mask_dice"].log(mask_score)
                # Log batch boundary dice
                run["training/batch/boundary_dice"].log(boundary_score)

                # TENSORBOARD VIZ
                writer.add_scalar('Dice/mask_test', mask_score, iteration)
                writer.add_scalar('Dice/boundary_test', boundary_score, iteration)

                #writer.add_images('images', imgs, global_step)
                #writer.add_images('masks/true', true_masks, global_step)
                mask_viz = torch.unsqueeze(torch.sigmoid(preds_mask_and_boundary[:,0,:,:]) > 0.5, dim=1)
                boundary_viz = torch.unsqueeze(torch.sigmoid(preds_mask_and_boundary[:,1,:,:]) > 0.5, dim=1)
                writer.add_images('masks/pred', mask_viz, iteration)
                writer.add_images('boundaries/pred', boundary_viz, iteration)

        if not os.path.exists(dir_checkpoint):
          try:
            os.mkdir(dir_checkpoint)
            logging.info('Created checkpoint directory')
          except OSError:
            pass
          if train_epoch+1 in checkpoints_to_save:
            model_dict = {'epoch': train_epoch+1, 
            'model_state_dict': net.state_dict(), 'optimizer_state_dict': optim.state_dict(), 
            'loss_tot': loss_tot, 'loss_mask': loss_mask, 'loss_boundary': loss_boundary}
            
            torch.save(model_dict, dir_checkpoint + f'CP_epoch{train_epoch+1}.pth')
            logging.info(f'Checkpoint {train_epoch+1} saved !')
        else:
          if train_epoch+1 in checkpoints_to_save:
            model_dict = {'epoch': train_epoch+1, 
            'model_state_dict': net.state_dict(), 'optimizer_state_dict': optim.state_dict(), 
            'loss_tot': loss_tot, 'loss_mask': loss_mask, 'loss_boundary': loss_boundary,}
            
            torch.save(model_dict, dir_checkpoint + f'CP_epoch{train_epoch+1}.pth')
            logging.info(f'Checkpoint {train_epoch+1} saved !')
        
        train_epoch += 1

    writer.close()
if __name__ == '__main__':
    args = get_args()
    run = neptune.init(
        project="vilgefortz2300/PYTORCH-HRNetv2-CROPS-DELINEATION",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkODRmMDMxYS1jMjZjLTQ0MzgtYTcwNS1jNzc5NzkzYzgzODEifQ==",
    )  # your credentials

    run['config/criterion'] = loss_fn
    run['config/optimizer'] = optimizer
    run['config/params'] = track_hparams_dict  # dict() object
    run['source_code/files'].upload_files("core/*.py")
    run['source_code/files'].upload_files("*.py")

    net = HRNetv2(img_channels=4)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net.to(device=device)

    if args.resume != 'None':
        checkpoint = torch.load(args.resume, map_location=device)
        print(checkpoint.keys())
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        try:
            last_epoch = checkpoint['epoch']
            try:
                train_net(net=net,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        device=device,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                        dir_img=dir_img,
                        dir_mask=dir_mask,
                        dir_boundaries=dir_boundaries,
                        dir_checkpoint=dir_checkpoint,
                        val_percent=val_percent,
                        checkpoints_to_save=checkpoints_to_save,
                        last_epoch=last_epoch)
            except KeyboardInterrupt:
                torch.save(net.state_dict(), 'INTERRUPTED.pth')
                logging.info('Saved interrupt')
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)
        except KeyError:
            try:
                train_net(net=net,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        device=device,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                        dir_img=dir_img,
                        dir_mask=dir_mask,
                        dir_boundaries=dir_boundaries,
                        dir_checkpoint=dir_checkpoint,
                        val_percent=val_percent,
                        checkpoints_to_save=checkpoints_to_save)
            except KeyboardInterrupt:
                torch.save(net.state_dict(), 'INTERRUPTED.pth')
                logging.info('Saved interrupt')
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)
    else:
        net.apply(initialize_weights)
        try:
            train_net(net=net,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    device=device,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    dir_img=dir_img,
                    dir_mask=dir_mask,
                    dir_boundaries=dir_boundaries,
                    dir_checkpoint=dir_checkpoint,
                    val_percent=val_percent,
                    checkpoints_to_save=checkpoints_to_save)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)