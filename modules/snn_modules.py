import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import pickle
import sys
import os
sys.path.append('../')
from modules.optimizations import VariationalHidDropout2d, weight_frob_norm_restriction


class SNNFuncMultiLayer(nn.Module):

    def __init__(self, network_s_list, network_x, vth, fb_num=1, vth_back=None, u_rest=None, u_rest_back=None):
        # network_s_list is a list of networks, the last fb_num are the feedback while previous are feed-forward
        super(SNNFuncMultiLayer, self).__init__()
        self.network_s_list = network_s_list
        self.network_x = network_x
        self.vth = torch.tensor(vth, requires_grad=False)
        self.fb_num = fb_num
        if vth_back == None:
            self.vth_back = self.vth
        else:
            self.vth_back = torch.tensor(vth_back, requires_grad=False)
        if u_rest == None:
            self.u_rest = -self.vth
        else:
            self.u_rest = torch.tensor(u_rest, requires_grad=False)
        if u_rest_back == None:
            self.u_rest_back = -self.vth_back
        else:
            self.u_rest_back = torch.tensor(u_rest_back, requires_grad=False)

    def snn_forward(self, x, time_step, f_type='first', input_type='constant'):
        pass

    def snn_backward(self, grad, time_step):
        pass

    def forward(self, x, time_step):
        return self.snn_forward(x, time_step)

    def copy(self, target):
        for i in range(len(self.network_s_list)):
            self.network_s_list[i].copy(target.network_s_list[i])
        self.network_x.copy(target.network_x)
        self.vth = target.vth
        self.fb_num = target.fb_num
        self.vth_back = target.vth_back
        self.u_rest = target.u_rest
        self.u_rest_back = target.u_rest_back


class SNNIFFuncMultiLayer(SNNFuncMultiLayer):

    def __init__(self, network_s_list, network_x, vth, fb_num=1, vth_back=None, u_rest=None, u_rest_back=None):
        super(SNNIFFuncMultiLayer, self).__init__(network_s_list, network_x, vth, fb_num, vth_back, u_rest, u_rest_back)

    def snn_forward(self, x, time_step, f_type='all', input_type='constant'):
        with torch.no_grad():
            if input_type == 'constant':
                x1 = self.network_x(x)
            else:
                x1 = self.network_x(x[0])
            u_list = []
            s_list = []
            a_list = []
            u1 = x1
            s1 = (u1 >= self.vth).float()
            u1 = u1 - (self.vth - self.u_rest) * s1
            u_list.append(u1)
            s_list.append(s1)
            a = s1
            a_list.append(a)
            for i in range(len(self.network_s_list) - 1):
                ui = self.network_s_list[i](s_list[-1])
                si = (ui >= self.vth).float()
                ui = ui - (self.vth - self.u_rest) * si
                u_list.append(ui)
                s_list.append(si)
                ai = si
                a_list.append(ai)

            for t in range(time_step - 1):
                if input_type == 'constant':
                    u_list[0] = u_list[0] + self.network_s_list[-1](s_list[-1]) + x1
                else:
                    u_list[0] = u_list[0] + self.network_s_list[-1](s_list[-1]) + self.network_x(x[t+1])
                s_list[0] = (u_list[0] >= self.vth).float()
                u_list[0] = u_list[0] - (self.vth - self.u_rest) * s_list[0]

                for i in range(len(self.network_s_list) - 1):
                    u_list[i + 1] = u_list[i + 1] + self.network_s_list[i](s_list[i])
                    s_list[i + 1] = (u_list[i + 1] >= self.vth).float()
                    u_list[i + 1] = u_list[i + 1] - (self.vth - self.u_rest) * s_list[i + 1]

                for i in range(len(a_list)):
                    a_list[i] += s_list[i]

            # save for backward
            self.a_list = []
            self.m_list = []
            for i in range(len(a_list)):
                self.a_list.append(a_list[i] * 1.0 / time_step)
                self.m_list.append((self.a_list[i] > 0.).float() * (self.a_list[i] < 1.).float())
            if input_type == 'constant':
                self.x = x
            else:
                self.x = torch.mean(x, dim=0)

            self.last_forward_firing_rate = []
            for a in self.a_list:
                self.last_forward_firing_rate.append(a.cpu())

            if f_type == 'first':
                return self.a_list[0]
            elif f_type == 'last':
                return self.a_list[-self.fb_num]
            else:
                return self.a_list[0], self.a_list[-self.fb_num]

    def snn_backward(self, grad, time_step):
        with torch.no_grad():
            # the list is in the inverse order, from the last layer to the first layer
            u_list_b = []
            s_list_b = []
            beta_list = []
            spike_num_list = []
            u_b = grad
            # deal with the negative gradient
            s_b = ((u_b >= self.vth_back).float() - (u_b <= -self.vth_back).float()) * (self.vth_back - self.u_rest_back)
            u_b = u_b - s_b

            u_list_b.append(u_b)
            s_list_b.append(s_b)
            beta = s_b
            beta_list.append(beta)
            spike_num = torch.abs(s_b / (self.vth_back - self.u_rest_back))
            spike_num_list.append(spike_num)
            network_num = len(self.network_s_list)
            for i in range(network_num - 1):
                ui_b = self.network_s_list[network_num - self.fb_num - 1 - i].back_compute(s_list_b[-1] * self.m_list[network_num - self.fb_num - i], 1. / (self.vth - self.u_rest))
                si_b = ((ui_b >= self.vth_back).float() - (ui_b <= -self.vth_back).float()) * (self.vth_back - self.u_rest_back)
                ui_b = ui_b - si_b
                u_list_b.append(ui_b)
                s_list_b.append(si_b)
                betai = si_b
                beta_list.append(betai)
                spike_numi = torch.abs(si_b / (self.vth_back - self.u_rest_back))
                spike_num_list.append(spike_numi)

            for t in range(time_step - 1):
                u_list_b[0] = u_list_b[0] + self.network_s_list[-self.fb_num].back_compute(s_list_b[-1] * self.m_list[-self.fb_num + 1], 1. / (self.vth - self.u_rest)) + grad
                s_list_b[0] = ((u_list_b[0] >= self.vth_back).float() - (u_list_b[0] <= -self.vth_back).float()) * (self.vth_back - self.u_rest_back)
                u_list_b[0] = u_list_b[0] - s_list_b[0]
                beta_list[0] = beta_list[0] + s_list_b[0]
                spike_num_list[0] = spike_num_list[0] + torch.abs(s_list_b[0] / (self.vth_back - self.u_rest_back))

                for i in range(network_num - 1):
                    u_list_b[i + 1] = u_list_b[i + 1] + self.network_s_list[network_num - self.fb_num - 1 - i].back_compute(s_list_b[i] * self.m_list[network_num - self.fb_num - i], 1. / (self.vth - self.u_rest))
                    s_list_b[i + 1] = ((u_list_b[i + 1] >= self.vth_back).float() - (u_list_b[i + 1] <= -self.vth_back).float()) * (self.vth_back - self.u_rest_back)
                    u_list_b[i + 1] = u_list_b[i + 1] - s_list_b[i + 1]
                    beta_list[i + 1] = beta_list[i + 1] + s_list_b[i + 1]
                    spike_num_list[i + 1] = spike_num_list[i + 1] + torch.abs(s_list_b[i + 1] / (self.vth_back - self.u_rest_back))

            for i in range(network_num):
                beta_list[i] = beta_list[i] * 1. / time_step
                spike_num_list[i] = (spike_num_list[i] * 1. / time_step).cpu()

        for i in range(network_num):
            self.network_s_list[i].set_grad(self.vth - self.u_rest, beta_list[network_num - self.fb_num - i - 1] * self.m_list[(i + 1) % network_num], self.a_list[i])
        self.network_x.set_grad(self.vth - self.u_rest, beta_list[-self.fb_num] * self.m_list[0], self.x)

        self.last_backward_firing_rate = []
        self.last_backward_firing_rate = spike_num_list


class SNNLIFFuncMultiLayer(SNNFuncMultiLayer):

    def __init__(self, network_s_list, network_x, vth, leaky, fb_num=1, vth_back=None, u_rest=None, u_rest_back=None):
        super(SNNLIFFuncMultiLayer, self).__init__(network_s_list, network_x, vth, fb_num, vth_back, u_rest, u_rest_back)
        self.leaky = torch.tensor(leaky, requires_grad=False)

    def snn_forward(self, x, time_step, f_type='all', input_type='constant'):
        with torch.no_grad():
            if input_type == 'constant':
                x1 = self.network_x(x)
            else:
                x1 = self.network_x(x[0])
            u_list = []
            s_list = []
            a_list = []
            spike_num_list = []
            u1 = x1
            s1 = (u1 >= self.vth).float()
            u1 = u1 - (self.vth - self.u_rest) * s1
            # add leaky term here
            u1 = u1 * self.leaky

            u_list.append(u1)
            s_list.append(s1)
            a = s1
            a_list.append(a)
            spike_num = s1
            spike_num_list.append(spike_num)
            for i in range(len(self.network_s_list) - 1):
                ui = self.network_s_list[i](s_list[-1])
                si = (ui >= self.vth).float()
                ui = ui - (self.vth - self.u_rest) * si
                # add leaky term here
                ui = ui * self.leaky

                u_list.append(ui)
                s_list.append(si)
                ai = si
                a_list.append(ai)
                spike_numi = si
                spike_num_list.append(spike_numi)

            for t in range(time_step - 1):
                if input_type == 'constant':
                    u_list[0] = u_list[0] + self.network_s_list[-1](s_list[-1]) + x1
                else:
                    u_list[0] = u_list[0] + self.network_s_list[-1](s_list[-1]) + self.network_x(x[t+1])
                s_list[0] = (u_list[0] >= self.vth).float()
                u_list[0] = u_list[0] - (self.vth - self.u_rest) * s_list[0]
                # add leaky term here
                u_list[0] = u_list[0] * self.leaky

                for i in range(len(self.network_s_list) - 1):
                    u_list[i + 1] = u_list[i + 1] + self.network_s_list[i](s_list[i])
                    s_list[i + 1] = (u_list[i + 1] >= self.vth).float()
                    u_list[i + 1] = u_list[i + 1] - (self.vth - self.u_rest) * s_list[i + 1]
                    # add leaky term here
                    u_list[i + 1] = u_list[i + 1] * self.leaky

                for i in range(len(a_list)):
                    a_list[i] = a_list[i] * self.leaky + s_list[i]
                    spike_num_list[i] = spike_num_list[i] + s_list[i]

            weighted = ((1. - self.leaky ** time_step) / (1. - self.leaky))
            # save for backward
            self.a_list = []
            self.m_list = []
            for i in range(len(a_list)):
                self.a_list.append(a_list[i] / weighted)
                self.m_list.append((self.a_list[i] > 0.).float() * (self.a_list[i] < 1.).float())
                spike_num_list[i] = (spike_num_list[i] * 1. / time_step).cpu()
            if input_type == 'constant':
                self.x = x
            else:
                self.x = torch.mean(x, dim=0)

            self.last_forward_firing_rate = []
            self.last_forward_firing_rate = spike_num_list

            if f_type == 'first':
                return self.a_list[0]
            elif f_type == 'last':
                return self.a_list[-self.fb_num]
            else:
                return self.a_list[0], self.a_list[-self.fb_num]

    def snn_backward(self, grad, time_step, neuron_type='LIF'):
        with torch.no_grad():
            # the list is in the inverse order, from the last layer to the first layer
            u_list_b = []
            s_list_b = []
            beta_list = []
            spike_num_list = []
            u_b = grad
            # deal with the negative gradient
            s_b = ((u_b >= self.vth_back).float() - (u_b <= -self.vth_back).float()) * (self.vth_back - self.u_rest_back)
            u_b = u_b - s_b
            if neuron_type == 'LIF':
                # add leaky term here
                u_b = u_b * self.leaky

            u_list_b.append(u_b)
            s_list_b.append(s_b)
            beta = s_b
            beta_list.append(beta)
            spike_num = torch.abs(s_b / (self.vth_back - self.u_rest_back))
            spike_num_list.append(spike_num)
            network_num = len(self.network_s_list)
            for i in range(network_num - 1):
                ui_b = self.network_s_list[network_num - self.fb_num - 1 - i].back_compute(s_list_b[-1] * self.m_list[network_num - self.fb_num - i], 1. / (self.vth - self.u_rest))
                si_b = ((ui_b >= self.vth_back).float() - (ui_b <= -self.vth_back).float()) * (self.vth_back - self.u_rest_back)
                ui_b = ui_b - si_b
                if neuron_type == 'LIF':
                    # add leaky term here
                    ui_b = ui_b * self.leaky

                u_list_b.append(ui_b)
                s_list_b.append(si_b)
                betai = si_b
                beta_list.append(betai)
                spike_numi = torch.abs(si_b / (self.vth_back - self.u_rest_back))
                spike_num_list.append(spike_numi)

            for t in range(time_step - 1):
                u_list_b[0] = u_list_b[0] + self.network_s_list[-self.fb_num].back_compute(s_list_b[-1] * self.m_list[-self.fb_num + 1], 1. / (self.vth - self.u_rest)) + grad
                s_list_b[0] = ((u_list_b[0] >= self.vth_back).float() - (u_list_b[0] <= -self.vth_back).float()) * (self.vth_back - self.u_rest_back)
                u_list_b[0] = u_list_b[0] - s_list_b[0]
                if neuron_type == 'LIF':
                    # add leaky term here
                    u_list_b[0] = u_list_b[0] * self.leaky
                    beta_list[0] = beta_list[0] * self.leaky + s_list_b[0]
                else:
                    beta_list[0] = beta_list[0] + s_list_b[0]
                spike_num_list[0] = spike_num_list[0] + torch.abs(s_list_b[0] / (self.vth_back - self.u_rest_back))

                for i in range(network_num - 1):
                    u_list_b[i + 1] = u_list_b[i + 1] + self.network_s_list[network_num - self.fb_num - 1 - i].back_compute(s_list_b[i] * self.m_list[network_num - self.fb_num - i], 1. / (self.vth - self.u_rest))
                    s_list_b[i + 1] = ((u_list_b[i + 1] >= self.vth_back).float() - (u_list_b[i + 1] <= -self.vth_back).float()) * (self.vth_back - self.u_rest_back)
                    u_list_b[i + 1] = u_list_b[i + 1] - s_list_b[i + 1]
                    if neuron_type == 'LIF':
                        # add leaky term here
                        u_list_b[i + 1] = u_list_b[i + 1] * self.leaky
                        beta_list[i + 1] = beta_list[i + 1] * self.leaky + s_list_b[i + 1]
                    else:
                        beta_list[i + 1] = beta_list[i + 1] + s_list_b[i + 1]
                    spike_num_list[i + 1] = spike_num_list[i + 1] + torch.abs(s_list_b[i + 1] / (self.vth_back - self.u_rest_back))

            if neuron_type == 'LIF':
                weighted = ((1. - self.leaky ** time_step) / (1. - self.leaky))
            else:
                weighted = 1. * time_step
            for i in range(network_num):
                beta_list[i] = beta_list[i] / weighted
                spike_num_list[i] = (spike_num_list[i] * 1. / time_step).cpu()

        for i in range(network_num):
            self.network_s_list[i].set_grad(self.vth - self.u_rest, beta_list[network_num - self.fb_num - i - 1] * self.m_list[(i + 1) % network_num], self.a_list[i])
        self.network_x.set_grad(self.vth - self.u_rest, beta_list[-self.fb_num] * self.m_list[0], self.x)

        self.last_backward_firing_rate = []
        self.last_backward_firing_rate = spike_num_list


class SNNFC(nn.Module):

    def __init__(self, d_in, d_out, bias=False, need_resize=False, sizes=None, dropout=0.0, init='default'):
        super(SNNFC, self).__init__()
        self.fc = nn.Linear(d_in, d_out, bias=bias)
        self.need_resize = need_resize
        self.sizes = sizes
        if dropout > 0:
            self.drop = VariationalHidDropout2d(dropout, spatial=False)
        else:
            self.drop = None

        self._initialize_weights(init)

    def forward(self, x):
        if self.need_resize:
            self.forward_sizes = x.size()
            if self.sizes == None:
                sizes = x.size()
                B = sizes[0]
                x = torch.reshape(self.fc(x.reshape(B, -1)), sizes)
            else:
                B = x.size(0)
                self.sizes[0] = B
                x = torch.reshape(self.fc(x.reshape(B, -1)), self.sizes)
        else:
            x = self.fc(x)
        if self.drop:
            return self.drop(x)
        return x

    def back_compute(self, x, factor):
        B = x.shape[0]
        if self.drop:
            x = self.drop(x)
        if self.need_resize:
            x = x.reshape(B, -1)
            x = x.matmul(self.fc.weight) * factor
            x = torch.reshape(x, self.forward_sizes)
        else:
            x = x.matmul(self.fc.weight) * factor
        return x

    def set_grad(self, factor, beta, a):
        B = beta.shape[0]
        beta = beta.reshape(B, -1)
        a = a.reshape(B, -1)
        self.fc.weight.grad = 1. / factor * beta.t().matmul(a)
        if self.fc.bias != None:
            self.fc.bias.grad = 1. / factor * torch.sum(beta, dim=0)


    def _wnorm(self, norm_range=1.):
        self.fc, self.fc_fn = weight_frob_norm_restriction(self.fc, names=['weight'], dim=0, norm_range=norm_range)

    def _reset(self, x):
        if 'fc_fn' in self.__dict__:
            self.fc_fn.reset(self.fc)
        if self.drop:
            self.drop.reset_mask(x)

    def _initialize_weights(self, init):
        m = self.fc
        if init == 'default':
            m.weight.data.uniform_(-1, 1)
            for i in range(m.weight.size(0)):
                m.weight.data[i] /= torch.norm(m.weight.data[i])
        elif init == 'kaiming':
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            m.bias.data.zero_()

    def copy(self, target):
        self.fc.weight.data = target.fc.weight.data.clone()
        if self.fc.bias is not None:
            self.fc.bias.data = target.fc.bias.data.clone()

        self.need_resize = target.need_resize
        self.sizes = target.sizes

# weight standardization
class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, gamma=None):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if gamma == None:
            gamma = 1.71
        self.gamma = gamma / np.sqrt(self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3])
        self.scale = 1.#nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        weight = self.weight
        mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight) * self.gamma * self.scale

        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class SNNConv(nn.Module):

    def __init__(self, d_in, d_out, kernel_size, bias=False, stride=1, padding=None, dropout=0.0, init='default', wsconv=False):
        super(SNNConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        if padding == None:
            padding = (kernel_size - 1) // 2
        self.padding = padding
        if wsconv:
            self.conv = WSConv2d(d_in, d_out, kernel_size, stride, padding, bias=bias, gamma=1.)
        else:
            self.conv = nn.Conv2d(d_in, d_out, kernel_size, stride, padding, bias=bias)
        self.wsconv = wsconv
        if dropout > 0:
            self.drop = VariationalHidDropout2d(dropout, spatial=False)
        else:
            self.drop = None

        self._initialize_weights(init)

    def forward(self, x, BN_mean_var=None):
        x = self.conv(x)
        if self.drop:
            return self.drop(x)
        return x

    def back_compute(self, x, factor):
        if self.drop:
            x = self.drop(x)
        if not self.wsconv:
            x = F.conv_transpose2d(x, self.conv.weight, bias=None, stride=self.stride, padding=self.padding, output_padding=self.stride-1) * factor
        else:
            weight = self.conv.weight
            mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            weight = weight - mean
            std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
            weight = weight / std.expand_as(weight) * self.conv.gamma * self.conv.scale
            x = F.conv_transpose2d(x, weight, bias=None, stride=self.stride, padding=self.padding, output_padding=self.stride-1) * factor
        return x

    def set_grad(self, factor, beta, a):
        with torch.enable_grad():
            a_ = a.clone().detach().requires_grad_()
            tmp = self.conv(a_)
        tmp.backward(1. / factor * beta)

    def _wnorm(self, norm_range=1.):
        self.conv, self.conv_fn = weight_frob_norm_restriction(self.conv, names=['weight'], dim=0, norm_range=norm_range)

    def _reset(self, x):
        if 'conv_fn' in self.__dict__:
            self.conv_fn.reset(self.conv)
        if self.drop:
            self.drop.reset_mask(x)

    def _initialize_weights(self, init):
        m = self.conv
        if init == 'default':
            m.weight.data.uniform_(-1, 1)
            for i in range(m.out_channels):
                m.weight.data[i] /= torch.norm(m.weight.data[i])
        elif init == 'kaiming':
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            m.bias.data.zero_()

    def copy(self, target):
        self.conv.weight.data = target.conv.weight.data.clone()
        if self.conv.bias is not None:
            self.conv.bias.data = target.conv.bias.data.clone()


class SNNConvTranspose(nn.Module):

    def __init__(self, d_in, d_out, kernel_size=3, bias=False, stride=2, padding=1, output_padding=1, dropout=0.0, init='default'):
        super(SNNConvTranspose, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.convT = nn.ConvTranspose2d(d_in, d_out, kernel_size, stride, padding, output_padding, bias=bias)
        if dropout > 0:
            self.drop = VariationalHidDropout2d(dropout, spatial=False)
        else:
            self.drop = None

        self._initialize_weights(init)

    def forward(self, x):
        x = self.convT(x)
        if self.drop:
            return self.drop(x)
        return x

    def back_compute(self, x, factor):
        if self.drop:
            x = self.drop(x)
        x = F.conv2d(x, self.convT.weight, bias=None, stride=self.stride, padding=self.padding) * factor
        return x

    def set_grad(self, factor, beta, a):
        with torch.enable_grad():
            a_ = a.clone().detach().requires_grad_()
            tmp = self.convT(a_)
        tmp.backward(1. / factor * beta)

    def _wnorm(self, norm_range=1.):
        self.convT, self.convT_fn = weight_frob_norm_restriction(self.convT, names=['weight'], dim=0, norm_range=norm_range)

    def _reset(self, x):
        if 'convT_fn' in self.__dict__:
            self.convT_fn.reset(self.convT)
        if self.drop:
            self.drop.reset_mask(x)

    def _initialize_weights(self, init):
        m = self.convT
        if init == 'default':
            m.weight.data.uniform_(-1, 1)
            for i in range(m.out_channels):
                m.weight.data[:, i] /= torch.norm(m.weight.data[:, i])
        elif init == 'kaiming':
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            m.bias.data.zero_()
        m.weight.data = m.weight.data * 0.1

    def copy(self, target):
        self.convT.weight.data = target.convT.weight.data.clone()
        if self.convT.bias is not None:
            self.convT.bias.data = target.convT.bias.data.clone()


class SNNZero(nn.Module):

    def __init__(self, input_size, output_size):
        super(SNNZero, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        return torch.zeros(self.output_size).to(x.device)

    def back_compute(self, x, factor):
        return torch.zeros(self.input_size).to(x.device)

    def set_grad(self, factor, beta, a):
        return

    def _wnorm(self, norm_range=1.):
        return

    def _reset(self, x):
        return

    def copy(self, target):
        self.input_size = target.input_size
        self.output_size = target.output_size


class SNNPooling(nn.Module):

    def __init__(self):
        super(SNNPooling, self).__init__()
        self.pool = nn.AvgPool2d((2, 2), stride=(2, 2))

    def forward(self, x):
        return self.pool(x)

    def back_compute(self, x, factor):
        tmp = torch.zeros(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3]*2).to(x.device).requires_grad_(True)
        with torch.enable_grad():
            y = self.pool(tmp)
            y.backward(x)
        return tmp.grad * factor

    def set_grad(self, factor, beta, a):
        return

    def _wnorm(self, norm_range=1.):
        return

    def _reset(self, x):
        return

    def copy(self, target):
        return
