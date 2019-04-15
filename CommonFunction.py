#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:42:35 2019

@author: xb257963
"""

import numpy as np

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z
    assert(A.shape == Z.shape)
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z<=0] = 0
    assert (dZ.shape == Z.shape)
    return dZ
    
    
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    assert(A.shape == Z.shape)
    return A, cache

def sigmoid_back(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert(dZ.shape == Z.shape)
    return dZ