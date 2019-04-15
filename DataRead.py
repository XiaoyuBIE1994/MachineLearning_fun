#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:01:36 2019

@author: Xiaoyu BIE
"""

import numpy as np
import pickle
import os
 
class Cifar10Reader():
    def __init__(self, cifar10_dir):
        self.data_dir = cifar10_dir

    def unpickle(self, file_path):
        with open(file_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def data_extraction(self):
        # train data
        train_x, train_y = [], []
        for i in range(1, 6):
            data_path = self.data_dir + "/data_batch_" + str(i)
            data_dict = self.unpickle(data_path)
            train_x = train_x + (data_dict[b'data'].tolist())
            train_y = train_y + (data_dict[b'labels'])
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        
        # test data
        data_path = self.data_dir + "/test_batch"
        data_dict = self.unpickle(data_path)
        test_x = data_dict[b'data']
        test_y = np.array(data_dict[b'labels'])
        
        return train_x, train_y, test_x, test_y

    def get_image(self, pixel):
        assert len(pixel) == 3072 # images in cifar10 32*32*3
        r = (pixel[0:1024]).reshape((32,32,1))
        g = (pixel[1024:2048]).reshape((32,32,1))
        b = (pixel[2048:3072]).reshape((32,32,1))
        image = np.concatenate([r, g, b], -1)
        return image