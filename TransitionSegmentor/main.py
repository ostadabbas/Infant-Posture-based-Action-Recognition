import torch
import torch.nn as nn
import torch.optim as optim
import tensorboardX
import os
import random
import numpy as np

from train import train_epoch
from seqdataset import SequenceDataset, collate_fn
from torch.utils.data import DataLoader
from validation import val_epoch
from opts import parse_opts
from model import generate_model
from torch.optim import lr_scheduler
from dataset import get_training_set, get_validation_set
from mean import get_mean, get_std
from spatial_transforms import (
	Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
	MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from models import custom_loss
from utils import loss_function

def resume_model(opt, model, optimizer):
    """ Resume model 
    """
    checkpoint = torch.load(opt.resume_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model Restored from Epoch {}".format(checkpoint['epoch']))
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch


def main_worker():
    opt = parse_opts()
    print(opt)

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA for PyTorch
    device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")

    # tensorboard
    summary_writer = tensorboardX.SummaryWriter(log_dir='ts_logs')

    # defining model
    model =  generate_model(opt, device)

    # get data loaders
    # Create dataset
    inputs_2d_path = '/work/aclab/xiaof.huang/fu.n/InfantAction/ft_posture_model_outputs/posture_2d_res'
    inputs_3d_path = '/work/aclab/xiaof.huang/fu.n/InfantAction/ft_posture_model_outputs/posture_3d_res'

    root_path = '/work/aclab/xiaof.huang/fu.n/InfantAction/InfAct_images_dataset'
    test_dataset = SequenceDataset(inputs_2d_path, root_path, is_trainset=False)
    train_dataset = SequenceDataset(inputs_2d_path, root_path, is_trainset=True)

    # Create data loader with collate_fn
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    '''
    # Iterate over batches
    for batch in test_loader:
        seqs, labels = batch
        # Do something with signals and labels
        print(labels)
        print(seqs)
        pass
    '''

    # optimizer
    lstm_params = list(model.parameters())
    optimizer = torch.optim.Adam(lstm_params, lr=opt.lr_rate, weight_decay=opt.weight_decay)

    # scheduler = lr_scheduler.ReduceLROnPlateau(
    # 	optimizer, 'min', patience=opt.lr_patience)
    #criterion = nn.MSELoss()   
    criterion = custom_loss.CustomLoss()     
    #criterion = loss_function
    # resume model
    if opt.resume_path:
        start_epoch = resume_model(opt, model, optimizer)
    else:
        start_epoch = 1

    # start training
    for epoch in range(start_epoch, opt.n_epochs + 1):
        train_loss, train_acc = train_epoch(
			model, train_loader, criterion, optimizer, epoch, opt.log_interval, opt.eps, device)
        val_loss, val_acc = val_epoch(
			model, test_loader, criterion, opt.eps, device)

        # saving weights to checkpoint
        if (epoch) % opt.save_interval == 0:
            # scheduler.step(val_loss)
            # write summary
            summary_writer.add_scalar(
				'losses/train_loss', train_loss, global_step=epoch)
            summary_writer.add_scalar(
				'losses/val_loss', val_loss, global_step=epoch)
            summary_writer.add_scalar(
				'acc/train_acc', train_acc * 100, global_step=epoch)
            summary_writer.add_scalar(
				'acc/val_acc', val_acc * 100, global_step=epoch)

            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
            torch.save(state, os.path.join('snapshots', f'{opt.model}-Epoch-{epoch}-Loss-{val_loss}.pth'))
            print("Epoch {} model saved!\n".format(epoch))
    summary_writer.close()


if __name__ == "__main__":
    main_worker()