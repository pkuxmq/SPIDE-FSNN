import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np
import pickle
import sys
import os
from scipy.optimize import root
import time
import copy
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)


class SNNSPIDEModule(nn.Module):

    """ 
    SNN module with implicit differentiation on the equilibrium point in the inner 'Backward' class.
    """

    def __init__(self, snn_func):
        super(SNNSPIDEModule, self).__init__()
        self.snn_func = snn_func

    def forward(self, z1, u, **kwargs):
        time_step = kwargs.get('time_step', 30)
        time_step_back = kwargs.get('time_step_back', 250)
        input_type = kwargs.get('input_type', 'constant')
        leaky = kwargs.get('leaky', None)

        with torch.no_grad():
            if input_type != 'constant':
                if len(u.size()) == 3:
                    u = u.permute(2, 0, 1)
                else:
                    u = u.permute(4, 0, 1, 2, 3)

            z1_out = self.snn_func.snn_forward(u, time_step, 'last', input_type)
            torch.cuda.empty_cache()

        if self.training:
            z1_out.requires_grad_(True)
            z1_out = self.Backward.apply(self.snn_func, z1_out, time_step_back)

        return z1_out

    class Backward(Function):
        """
        A 'dummy' function that does nothing in the forward pass and perform implicit differentiation
        in the backward pass. Essentially a wrapper that provides backprop for the `DEQModule` class.
        You should use this inner class in DEQModule's forward() function by calling:
        
            self.Backward.apply(self.func_copy, ...)
            
        """
        @staticmethod
        def forward(ctx, snn_func, z1, *args):
            ctx.save_for_backward(z1)
            ctx.snn_func = snn_func
            ctx.args = args
            return z1.clone()

        @staticmethod
        def backward(ctx, grad):
            torch.cuda.empty_cache()
            z1, = ctx.saved_tensors
            args = ctx.args
            time_step_back = args[-1]
            snn_func = ctx.snn_func
            snn_func.snn_backward(grad, time_step_back)
            torch.cuda.empty_cache()

            grad_args = [None for _ in range(len(args))]
            return (None, None, None, *grad_args)
