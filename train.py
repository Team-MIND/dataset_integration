"""" This will contain the code for training the model"""

import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
#from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from time import gmtime, strftime
import datetime

from fmri_adhd_autism_ds import *
from transforms import *
from model import *

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

parser = argparse.ArgumentParser(description="MRI Training")
parser.add_argument('--root_dir', default='.', help='path to root of dataset')
parser.add_argument('--tsv_path', default='.', help='path to tsv')

parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.0005,
                    help='weight decay value')
parser.add_argument('--gpu_ids', default=[0], help='a list of gpus')
parser.add_argument('--num_worker', default=4, help='numbers of worker')
parser.add_argument('--batch_size', type=int, default=1, help='bach size')
parser.add_argument('--epochs', default=1, type=int, help='epochs')

parser.add_argument('--load_previous_model', default=None,
                    help='Load with last checkpoint')
parser.add_argument('--log_dir', default=None,
                    help='Logging and model directory')

parser.add_argument('--labels', '--list', nargs='+',
                    required=True, help='classes for training')
parser.add_argument('--use_cpu', default=False, action="store_true")
parser.add_argument('--healthy_vs_nonhealthy',
                    default=False, action="store_true")
parser.add_argument('--triplet_loss', default=False,
                    action="store_true", help="turns on triplet loss")

args = parser.parse_args()

writer = SummaryWriter(args.log_dir + '/')

# =================================================================
# Training Classes
# =================================================================


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:

    def __init__(self):

        self.args = args

        # Initialize model
        if torch.cuda.is_available() and not self.args.use_cpu:
            print("run with gpu")
            self.net = FullModel(n_classes=len(self.args.labels)).cuda()
            self.net = nn.DataParallel(self.net, device_ids=self.args.gpu_ids)
        else:
            print("run with cpu")
            self.net = FullModel(n_classes=len(self.args.labels))

        # create datasets
        self.trainset = FMRI_AA(root_dir=self.args.root_dir, search_path=self.args.tsv_path, is_train=True, transforms=[RandomFlips(), ToTensor()])
        self.testset = FMRI_AA(root_dir=self.args.root_dir, search_path=self.args.tsv_path, is_train=False, transforms=[ ToTensor()])

        # create dataloader
        self.train_data_loader = DataLoader(self.trainset, num_workers=self.args.num_worker,
                                            batch_size=self.args.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(self.testset, num_workers=self.args.num_worker,
                                           batch_size=self.args.batch_size)

        # Create optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Load in previous checkpoints if there was an error
        if self.args.load_previous_model:
            checkpoint = torch.load(self.args.load_previous_model)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Create loss criterion
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self, test):
        # Use AverageMeter for tracking preformance statistics
        losses = AverageMeter()

        preds_list = []
        labels_list = []

        acc = 0
        prec = 0
        recall = 0
        f1 = 0

        # Run model through validation dataset with no gradient calulation
        self.net.eval()
        with torch.no_grad():
            data = self.test_data_loader if test else self.train_data_loader

            for i, (inputs, labels) in tqdm(enumerate(data)):
                if torch.cuda.is_available() and not self.args.use_cpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                output = self.net(inputs)
                loss = self.criterion(output, labels)
                preds = torch.argmax(output, dim=1)

                preds = np.array(preds.cpu())
                labels = np.array(labels.cpu())

                preds_list.extend([self.args.labels[i] for i in preds])
                labels_list.extend([self.args.labels[i] for i in labels])

                losses.update(loss.item(), inputs.size(0))

            # print(preds_list, labels_list)

            acc = accuracy_score(preds_list, labels_list)
            prec = precision_score(preds_list, labels_list, average='macro')
            recall = recall_score(preds_list, labels_list, average='macro')
            f1 = f1_score(preds_list, labels_list, average='macro')

        self.net.train()
        return losses.avg, acc, prec, recall, f1

    def train(self):
        # metrics storage
        losses = AverageMeter()

        # do i need to add the time on these or just call it best?
        bestCheckPoint = "checkpoint" + \
            strftime("-%Y-%m-%d-%H00", gmtime()) + ".tar"
        print("---------------------")
        print("begining training.")
        print("---------------------")

        for epoch in tqdm(range(self.args.epochs)):
            for i, (inputs, labels) in tqdm(enumerate(self.train_data_loader)):

                if torch.cuda.is_available() and not self.args.use_cpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                self.optimizer.zero_grad()
                outputs = self.net(inputs)

                #print(outputs, labels)
                t1 = datetime.datetime.now()

                loss = self.criterion(outputs, labels)
                if self.args.triplet_loss:
                    trip_loss = nn.TripletMarginLoss()
                    pos = self.net(self.trainset.getScan(labels, True))
                    neg = self.net(self.trainset.getScan(labels, False))
                    loss = loss + trip_loss(outputs, pos, neg)
                loss.backward()
                self.optimizer.step()
                losses.update(loss.data.cpu().numpy())

                t2 = datetime.datetime.now()
                print("training time: %f" % (t2 - t1).total_seconds())
                # disply stats, save, and evaluate
                if i % 10 == 0:

                    print('testing......')
                    
                    t3 = datetime.datetime.now()
                    # _, train_acc, train_prec, train_rec, train_fscore = self.evaluate(False)
                    val_loss, val_acc, val_prec, val_rec, val_fscore = self.evaluate(True)

                    """ 
                    Training scoring metrics

                    writer.add_scalar('train_f1', train_fscore, epoch*len(self.train_data_loader) + i)
                    writer.add_scalar('train_prec', train_prec, epoch*len(self.train_data_loader) + i)
                    writer.add_scalar('train_rec', train_rec, epoch*len(self.train_data_loader) + i)
                    writer.add_scalar('train_acc', train_acc, epoch*len(self.train_data_loader) + i)
                    """

                    writer.add_scalar('train_loss', losses.avg, epoch*len(self.train_data_loader) + i)

                    writer.add_scalar('val_loss', val_loss, epoch*len(self.train_data_loader) + i)
                    writer.add_scalar('val_f1', val_fscore, epoch*len(self.train_data_loader) + i)
                    writer.add_scalar('val_prec', val_prec, epoch*len(self.train_data_loader) + i)
                    writer.add_scalar('val_rec', val_rec, epoch*len(self.train_data_loader) + i)
                    writer.add_scalar('val_acc', val_acc, epoch*len(self.train_data_loader) + i)

                    t4 = datetime.datetime.now()
                    print("validation time: %f" % (t4 - t3).total_seconds())

                    print("epoch{}, iter {}/{} | loss: {}, val_loss: {}, acc: {}, prec: {}, recall: {}, f1: {}".format(epoch, i, len(self.train_data_loader),
                                                                                                                       losses.avg, val_loss, val_acc, val_prec, val_rec, val_fscore))

                    if i != 0:
                        torch.save({'epoch': self.args.epochs,
                                    'model_state_dict': self.net.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'loss': losses.avg}, self.args.log_dir +
                                    '/epoch{}iter{}.tar'.format(epoch, i))


if __name__ == "__main__":
    """Run the training script with args provided at cmd line"""

    t = Trainer()
    t.train()
