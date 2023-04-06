import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import argparse
import tensorboardX
import os
import random
import numpy as np
from utils import AverageMeter, calculate_accuracy

def train_epoch(model, data_loader, criterion, optimizer, epoch, log_interval, eps, device):
    model.train()
 
    train_loss = 0.0
    losses = AverageMeter()
    accuracies = AverageMeter()
    for batch_idx, (data, targets, lengths) in enumerate(data_loader):
        print(torch.tensor(lengths))
        print(data.shape)
        print(targets)

        sorted_lengths, sorted_idx = torch.sort(torch.tensor(lengths), descending=True)
        sorted_data = data[sorted_idx]
        sorted_targets = torch.tensor(targets)[sorted_idx]
        print('qqqqqqqqqqqqq')
        print(sorted_lengths)
        print(sorted_targets)
        output= model(sorted_data, sorted_lengths)  # rnn
        #print(w_phi.requires_grad)
        #w_phi = structured(phi)  # structured
        #score, vot, onset_pred, offset_pred = structured.predict(w_phi)
        #score, onset_pred, offset_pred = hingeloss(w_phi)
        #pred_vots += [(onset + 1, offset + 1)]  # see Note at the bottom for the '+1'

       # print(list(model.parameters())[1].size())


        #onset_pred, offset_pred = model(data, lengths)
        #print(onset_pred)
        #print(offset_pred)
        #outputs = torch.stack((onset_pred+1, offset_pred+1), -1)
        #print(outputs[0])
        print('dddddddddddddd')
        print(sorted_targets.dtype)
        loss = criterion(output, sorted_targets)
        score, onset, offset = criterion.predict(output)
        print(score)
        print(onset)
        print(offset)
        #loss = criterion(output, sorted_targets, 5)

        #loss = StructuredHingeLoss(targets, onset_pred+1, offset_pred+1, eps)
        #acc = calculate_accuracy(outputs, targets)
        acc = 0

        #print(loss)
        losses.update(loss, data.size(0))
        accuracies.update(acc, data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = loss / log_interval        
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(data_loader.dataset), 100. * (batch_idx + 1) / len(data_loader), avg_loss))
            train_loss = 0.0

    print('Train set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
        len(data_loader.dataset), losses.avg, accuracies.avg * 100))

    return losses.avg, accuracies.avg  