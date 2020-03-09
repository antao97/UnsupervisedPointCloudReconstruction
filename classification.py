#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: reconstruction.py
@Time: 2020/1/2 10:26 AM
"""

import os
import sys
import time
import shutil
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics

from tensorboardX import SummaryWriter

from model import ClassificationNet
from dataset import Dataset
from utils import Logger


class Classification(object):
    def __init__(self, args):
        self.dataset_name = args.dataset
        if args.epochs != None:
            self.epochs = args.epochs
        else:
            self.epochs = 250
        self.batch_size = args.batch_size
        self.snapshot_interval = args.snapshot_interval
        self.no_cuda = args.no_cuda
        self.model_path = args.model_path

        # create exp directory
        file = [f for f in args.model_path.split('/')]
        if args.exp_name != None:
            self.experiment_id = "Classify_" + args.exp_name
        elif file[-2] == 'models':
            self.experiment_id = file[-3]
        else:
            self.experiment_id = "Classify" + time.strftime('%m%d%H%M%S')
        snapshot_root = 'snapshot/%s' % self.experiment_id
        tensorboard_root = 'tensorboard/%s' % self.experiment_id
        self.save_dir = os.path.join(snapshot_root, 'models/')
        self.tboard_dir = tensorboard_root

        # check arguments
        if self.model_path == '':
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            else:
                choose = input("Remove " + self.save_dir + " ? (y/n)")
                if choose == "y":
                    shutil.rmtree(self.save_dir)
                    os.makedirs(self.save_dir)
                else:
                    sys.exit(0)
            if not os.path.exists(self.tboard_dir):
                os.makedirs(self.tboard_dir)
            else:
                shutil.rmtree(self.tboard_dir)
                os.makedirs(self.tboard_dir)
        sys.stdout = Logger(os.path.join(snapshot_root, 'log.txt'))
        self.writer = SummaryWriter(log_dir=self.tboard_dir)

        # print args
        print(str(args))

        # get gpu id
        gids = ''.join(args.gpu.split())
        self.gpu_ids = [int(gid) for gid in gids.split(',')]
        self.first_gpu = self.gpu_ids[0]

        # generate dataset
        self.train_dataset = Dataset(
                root=args.dataset_root,
                dataset_name=args.dataset,
                split='all',
                num_points=args.num_points,
                random_translate=True,
                random_rotate=args.use_rotate,
                random_jitter=args.use_jitter
            )
        self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers
            )
        print("Training set size:", self.train_loader.dataset.__len__())

        # initialize model
        self.model = ClassificationNet(args)
        if self.model_path != '':
            self._load_pretrain(args.model_path)

        # load model to gpu
        if not self.no_cuda:
            if len(self.gpu_ids) != 1:  # multiple gpus
                self.model = torch.nn.DataParallel(self.model.cuda(self.first_gpu), self.gpu_ids)
            else:
                self.model = self.model.cuda(self.gpu_ids[0])
        
        # initialize optimizer
        self.parameter = self.model.parameters()
        self.optimizer = optim.SGD(self.parameter, lr=0.1, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, self.epochs, eta_min=1e-3)


    def run(self):
        self.train_hist = {
            'loss': [],
            'per_epoch_time': [],
            'total_time': []
        }
        best_loss = 1000000000
        print('Training start!!')
        start_time = time.time()
        self.model.train()
        if self.model_path != '':
            start_epoch = self.model_path[-7:-4]
            if start_epoch[0] == '_':
                start_epoch = start_epoch[1:]
            start_epoch = int(start_epoch)
        else:
            start_epoch = 0
        for epoch in range(start_epoch, self.epochs):
            loss = self.train_epoch(epoch)
            
            # save snapeshot
            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)
                if loss < best_loss:
                    best_loss = loss
                    self._snapshot('best')
            
            # save tensorboard
            if self.writer:
                self.writer.add_scalar('Train Loss', self.train_hist['loss'][-1], epoch)
                self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)
        
        # finish all epoch
        self._snapshot(epoch + 1)
        if loss < best_loss:
            best_loss = loss
            self._snapshot('best')
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epochs, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")


    def train_epoch(self, epoch):
        epoch_start_time = time.time()
        loss_buf = []
        train_pred = []
        train_true = []
        num_batch = int(len(self.train_loader.dataset) / self.batch_size)
        for iter, (pts, label) in enumerate(self.train_loader):
            if not self.no_cuda:
                pts = pts.cuda(self.first_gpu)
                label = label.cuda(self.first_gpu)

            # forward
            self.optimizer.zero_grad()
            output, _ = self.model(pts)

            # loss
            if len(self.gpu_ids) != 1:  # multiple gpus
                loss = self.model.module.get_loss(output, label)
            else:
                loss = self.model.get_loss(output, label)

            # backward
            loss.backward()
            self.optimizer.step()
            loss_buf.append(loss.detach().cpu().numpy())

            preds = output.max(dim=1)[1]
            train_true.append(label.view(-1).cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        # finish one epoch
        self.scheduler.step()
        epoch_time = time.time() - epoch_start_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['loss'].append(np.mean(loss_buf))
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        print("Epoch %d: Loss %.6f, train acc: %.6f, train avg acc: %.6f, time %.4fs" % (epoch+1,
                                                                                         np.mean(loss_buf),
                                                                                         metrics.accuracy_score(
                                                                                             train_true, train_pred),
                                                                                         metrics.balanced_accuracy_score(
                                                                                             train_true, train_pred),
                                                                                         epoch_time))
        return np.mean(loss_buf)


    def _snapshot(self, epoch):
        state_dict = self.model.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
        save_dir = os.path.join(self.save_dir, self.dataset_name)
        torch.save(new_state_dict, save_dir + "_" + str(epoch) + '.pkl')
        print(f"Save model to {save_dir}_{str(epoch)}.pkl")


    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
        self.model.load_state_dict(new_state_dict)
        print(f"Load model from {pretrain}")


    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
