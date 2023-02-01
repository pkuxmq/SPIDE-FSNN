import os
import sys
import logging
import functools

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import copy
sys.path.append("../")
from modules.snn_spide_module import SNNSPIDEModule
from modules.snn_modules import SNNIFFuncMultiLayer, SNNLIFFuncMultiLayer, SNNConv, SNNConvTranspose

logger = logging.getLogger(__name__)


class SNNSPIDEConvMultiLayerNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(SNNSPIDEConvMultiLayerNet, self).__init__()
        self.parse_cfg(cfg)

        self.network_x = SNNConv(self.c_in, self.c_hidden, self.kernel_size_x, bias=True, stride=self.stride_x, padding=self.padding_x, dropout=self.dropout, init='kaiming')

        self.network_s1 = SNNConv(self.c_hidden, self.c_s1, self.kernel_size_s, bias=True, stride=2, dropout=self.dropout, init='kaiming')
        self.network_s2 = SNNConv(self.c_s1, self.c_s1, self.kernel_size_s, bias=True, stride=1, dropout=self.dropout, init='kaiming')

        self.network_s3 = SNNConvTranspose(self.c_s1, self.c_hidden, bias=False, kernel_size=3, stride=2, padding=1, output_padding=1, dropout=self.dropout, init='kaiming')

        if self.leaky == None:
            self.snn_func = SNNIFFuncMultiLayer(nn.ModuleList([self.network_s1, self.network_s2, self.network_s3]), self.network_x, vth=self.vth, vth_back=self.vth_back, u_rest=self.u_rest, u_rest_back=self.u_rest_back)
        else:
            self.snn_func = SNNLIFFuncMultiLayer(nn.ModuleList([self.network_s1, self.network_s2, self.network_s3]), self.network_x, vth=self.vth, leaky=self.leaky, vth_back=self.vth_back, u_rest=self.u_rest, u_rest_back=self.u_rest_back)

        self.network_s3._wnorm(norm_range=2.)

        self.snn_spide_conv = SNNSPIDEModule(self.snn_func)

        self.classifier = nn.Linear(self.c_s1 * self.h_hidden * self.w_hidden, self.num_classes, bias=True)

    def parse_cfg(self, cfg):
        self.c_in = cfg['MODEL']['c_in']
        self.c_hidden = cfg['MODEL']['c_hidden']
        self.c_s1 = cfg['MODEL']['c_s1']
        self.h_hidden = cfg['MODEL']['h_hidden']
        self.w_hidden = cfg['MODEL']['w_hidden']
        self.num_classes = cfg['MODEL']['num_classes']
        self.kernel_size_x = cfg['MODEL']['kernel_size_x']
        self.stride_x = cfg['MODEL']['stride_x']
        self.padding_x = cfg['MODEL']['padding_x']
        self.pooling_x = cfg['MODEL']['pooling_x'] if 'pooling_x' in cfg['MODEL'].keys() else False
        self.kernel_size_s = cfg['MODEL']['kernel_size_s']
        self.time_step = cfg['MODEL']['time_step']
        self.time_step_back = cfg['MODEL']['time_step_back']
        self.vth = cfg['MODEL']['vth']
        self.vth_back = cfg['MODEL']['vth_back'] if 'vth_back' in cfg['MODEL'].keys() else self.vth
        self.u_rest = cfg['MODEL']['u_rest'] if 'u_rest' in cfg['MODEL'].keys() else None
        self.u_rest_back = cfg['MODEL']['u_rest_back'] if 'u_rest_back' in cfg['MODEL'].keys() else None
        self.dropout = cfg['MODEL']['dropout'] if 'dropout' in cfg['MODEL'].keys() else 0.0
        self.leaky = cfg['MODEL']['leaky'] if 'leaky' in cfg['MODEL'].keys() else None

    def _forward(self, x, **kwargs):
        time_step = kwargs.get('time_step', self.time_step)
        time_step_back = kwargs.get('time_step_back', self.time_step_back)
        input_type = kwargs.get('input_type', 'constant')
        leaky = kwargs.get('leaky', self.leaky)
        dev = x.device

        if input_type == 'constant':
            B, C, H, W = x.size()
        else:
            B, C, H, W, _ = x.size()

        x1 = torch.zeros([B, self.c_hidden, self.h_hidden*2, self.w_hidden*2]).to(x.device)
        self.snn_func.network_x._reset(x1)
        self.network_s3._reset(x1)

        x1 = torch.zeros([B, self.c_s1, self.h_hidden, self.w_hidden]).to(x.device)
        self.network_s1._reset(x1)
        self.network_s2._reset(x1)

        z = torch.zeros([B, self.c_hidden, self.h_hidden, self.w_hidden]).to(x.device)
        z = self.snn_spide_conv(z, x, time_step=time_step, time_step_back=time_step_back, input_type=input_type, leaky=leaky)

        return z

    def forward(self, x, **kwargs):
        B = x.size(0)
        z = self._forward(x, **kwargs)
        z = z.reshape(B, -1)
        y = self.classifier(z)

        return y

    def get_forward_firing_rate(self):
        firing_rates = self.snn_spide_conv.snn_func.last_forward_firing_rate
        num = 0.
        rate = 0.
        B = firing_rates[0].shape[0]
        for firing_rate in firing_rates:
            firing_rate = firing_rate.reshape(B, -1)
            num += firing_rate.shape[1]
            rate += torch.sum(firing_rate)

        return rate / num

    def get_backward_firing_rate(self):
        firing_rates = self.snn_spide_conv.snn_func.last_backward_firing_rate
        num = 0.
        rate = 0.
        B = firing_rates[0].shape[0]
        for firing_rate in firing_rates:
            firing_rate = firing_rate.reshape(B, -1)
            num += firing_rate.shape[1]
            rate += torch.sum(firing_rate)

        return rate / num
