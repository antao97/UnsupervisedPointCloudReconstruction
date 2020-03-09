#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: inference.py
@Time: 2020/1/2 10:26 AM
"""

import os
import sys
import time
import shutil
import torch
import numpy as np
import h5py

from tensorboardX import SummaryWriter

from model import ReconstructionNet, ClassificationNet
from dataset import Dataset
from utils import Logger


class Inference(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.no_cuda = args.no_cuda
        self.task = args.task

        # create exp directory
        file = [f for f in args.model_path.split('/')]
        if args.exp_name != None:
            self.experiment_id = args.exp_name
        else:
            self.experiment_id = time.strftime('%m%d%H%M%S')
        cache_root = 'cache/%s' % self.experiment_id
        os.makedirs(cache_root, exist_ok=True)
        self.feature_dir = os.path.join(cache_root, 'features/')
        sys.stdout = Logger(os.path.join(cache_root, 'log.txt'))

        # check directory
        if not os.path.exists(self.feature_dir):
            os.makedirs(self.feature_dir)
        else:
            shutil.rmtree(self.feature_dir)
            os.makedirs(self.feature_dir)

        # print args
        print(str(args))

        # get gpu id
        gids = ''.join(args.gpu.split())
        self.gpu_ids = [int(gid) for gid in gids.split(',')]
        self.first_gpu = self.gpu_ids[0]

        # generate dataset
        self.infer_dataset_train = Dataset(
            root=args.dataset_root,
            dataset_name=args.dataset,
            split='train',
            num_points=args.num_points,
        )
        self.infer_dataset_test = Dataset(
            root=args.dataset_root,
            dataset_name=args.dataset,
            split='test',
            num_points=args.num_points,
        )
        self.infer_loader_train = torch.utils.data.DataLoader(
            self.infer_dataset_train,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers
        )
        self.infer_loader_test = torch.utils.data.DataLoader(
            self.infer_dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers
        )
        print("Inference set size (train):", self.infer_loader_train.dataset.__len__())
        print("Inference set size (test):", self.infer_loader_test.dataset.__len__())

        # initialize model
        if args.task == "reconstruct":
            self.model = ReconstructionNet(args)
        elif args.task == "classify":
            self.model = ClassificationNet(args)
        if args.model_path != '':
            self._load_pretrain(args.model_path)

        # load model to gpu
        if not args.no_cuda:
            if len(self.gpu_ids) != 1:  # multiple gpus
                self.model = torch.nn.DataParallel(self.model.cuda(self.first_gpu), self.gpu_ids)
            else:
                self.model = self.model.cuda(self.gpu_ids[0])
        
    def run(self):
        self.model.eval()
        
        # generate train set for SVM
        loss_buf = []
        feature_train = []
        lbs_train = []
        n = 0
        for iter, (pts, lbs) in enumerate(self.infer_loader_train):
            if not self.no_cuda:
                pts = pts.cuda(self.first_gpu)
                lbs = lbs.cuda(self.first_gpu)
            if self.task == "reconstruct":
                output, feature = self.model(pts)
            elif self.task == "classify":
                feature = self.model(pts)
            feature_train.append(feature.detach().cpu().numpy().squeeze(1))
            lbs_train.append(lbs.cpu().numpy().squeeze(1))
            if ((iter+1) * self.batch_size % 2048) == 0 \
                or (iter+1) == len(self.infer_loader_train):
                feature_train = np.concatenate(feature_train, axis=0)
                lbs_train = np.concatenate(lbs_train, axis=0)
                f = h5py.File(os.path.join(self.feature_dir, 'train' + str(n) + '.h5'),'w') 
                f['data'] = feature_train                 
                f['label'] = lbs_train
                f.close()
                print("Train set {} for SVM saved.".format(n))
                feature_train = []
                lbs_train = []
                n += 1
            if self.task == "reconstruct":
                if len(self.gpu_ids) != 1:  # multiple gpus
                    loss = self.model.module.get_loss(pts, output)
                else:
                    loss = self.model.get_loss(pts, output)
                loss_buf.append(loss.detach().cpu().numpy())
        if self.task == "reconstruct":
            print(f'Avg loss {np.mean(loss_buf)}')
        print("Finish generating train set for SVM.")

        # generate test set for SVM
        loss_buf = []
        feature_test = []
        lbs_test = []
        n = 0
        for iter, (pts, lbs) in enumerate(self.infer_loader_test):
            if not self.no_cuda:
                pts = pts.cuda(self.first_gpu)
                lbs = lbs.cuda(self.first_gpu)
            if self.task == "reconstruct":
                output, feature = self.model(pts)
            elif self.task == "classify":
                feature = self.model(pts)
            feature_test.append(feature.detach().cpu().numpy().squeeze(1))
            lbs_test.append(lbs.cpu().numpy().squeeze(1))
            if ((iter+1) * self.batch_size % 2048) == 0 \
                or (iter+1) == len(self.infer_loader_test):
                feature_test = np.concatenate(feature_test, axis=0)
                lbs_test = np.concatenate(lbs_test, axis=0)
                f = h5py.File(os.path.join(self.feature_dir, 'test' + str(n) + '.h5'),'w') 
                f['data'] = feature_test                 
                f['label'] = lbs_test
                f.close()
                print("Test set {} for SVM saved.".format(n))
                feature_test = []
                lbs_test = []
                n += 1
            if self.task == "reconstruct":
                if len(self.gpu_ids) != 1:  # multiple gpus
                    loss = self.model.module.get_loss(pts, output)
                else:
                    loss = self.model.get_loss(pts, output)
                loss_buf.append(loss.detach().cpu().numpy())
        if self.task == "reconstruct":
            print(f'Avg loss {np.mean(loss_buf)}')
        print("Finish generating test set for SVM.")

        return self.feature_dir


    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            if key[:10] == 'classifier':
                continue
            new_state_dict[name] = val
        self.model.load_state_dict(new_state_dict)
        print(f"Load model from {pretrain}")
