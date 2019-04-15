#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:41:41 2019

@author: Xiaoyu BIE
"""
import numpy as np

class NeuralNetWork():
    def __init__ (self,layer_dims):
        assert(isinstance(layer_dims, list) and len(layer_dims > 1))
        self.layer_dims = layer_dims

    def init_params(self):
        self.parameters = {}
        L = len(self.layer_dims)
        for l in range(1, L):
            self.parameters["W" + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            self.parameters["b" + str(l)] = np.zeros(self.layer_dims[l], 1)
        return



