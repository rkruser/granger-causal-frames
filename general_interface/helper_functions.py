import torch 
import torch.nn as nn
import os
import sys
import numpy as np
import time
import argparse
from argparse import Namespace


# Normalize first input
def postprocess_1(args):
    return [torch.from_numpy(args[0]).float()/255.0] + [torch.from_numpy(np.array(val)).float() for val in args[1:-1]]+[torch.from_numpy(np.array(args[-1]))]

# No normalizing of input here
def postprocess_2(args):
    return [torch.from_numpy(args[0]).float()] + [torch.from_numpy(np.array(val)).float() for val in args[1:-1]]+[torch.from_numpy(np.array(args[-1]))]

postprocess_test = postprocess_2

def nullfunc_1(numpy_obj):
    return np.zeros(numpy_obj.shape, dtype=numpy_obj.dtype)

'''
Collating functions
'''
def collatefunc_1(args):
    data = args[0]
#    if len(data[0].shape) == 4:
    data = torch.stack(data).transpose(0,1)
#    else:
#        data = torch.stack(data)
    return [data] + [torch.stack(a) for a in args[1:]]

def collatefunc_2(args):
    return [torch.stack(a) for a in args]

