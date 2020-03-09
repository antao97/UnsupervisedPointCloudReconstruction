#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: svm.py
@Time: 2020/1/2 10:26 AM
"""

import os
import h5py
import numpy as np
from glob import glob
from sklearn.svm import LinearSVC


class SVM(object):
    def __init__(self, feature_dir):
        self.feature_dir = feature_dir

        self.train_path = glob(os.path.join(self.feature_dir, 'train*.h5'))
        self.test_path = glob(os.path.join(self.feature_dir, 'test*.h5'))

        print("Loading feature dataset...")
        train_data = []
        train_label = []
        for path in self.train_path:
            f = h5py.File(path, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            train_data.append(data)
            train_label.append(label)
        self.train_data = np.concatenate(train_data, axis=0)
        self.train_label = np.concatenate(train_label, axis=0)
        print("Training set size:", np.size(self.train_data, 0))

        test_data = []
        test_label = []
        for path in self.test_path:
            f = h5py.File(path, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            test_data.append(data)
            test_label.append(label)
        self.test_data = np.concatenate(test_data, axis=0)
        self.test_label = np.concatenate(test_label, axis=0)
        print("Testing set size:", np.size(self.test_data, 0))

    def run(self):
        clf = LinearSVC(random_state=0) 
        clf.fit(self.train_data, self.train_label)  
        result = clf.predict(self.test_data)  
        accuracy = np.sum(result==self.test_label).astype(float) / np.size(self.test_label)
        print("Transfer linear SVM accuracy: {:.2f}%".format(accuracy*100))


