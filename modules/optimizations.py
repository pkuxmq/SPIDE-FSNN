# Modified based on the DEQ repo.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.functional import normalize


class VariationalHidDropout2d(nn.Module):
    def __init__(self, dropout=0.0, spatial=True):
        """
        Hidden-to-hidden (VD-based) dropout that applies the same mask at every pixel and every layer
        :param dropout: The dropout rate (0 means no dropout is applied)
        """
        super(VariationalHidDropout2d, self).__init__()
        self.dropout = dropout
        self.mask = None
        self.spatial = spatial

    def reset_mask(self, x):
        dropout = self.dropout
        spatial = self.spatial

        # x has dimension (N, C, H, W)
        if spatial:
            m = torch.zeros_like(x[:,:,:1,:1]).bernoulli_(1 - dropout)
        else:
            m = torch.zeros_like(x).bernoulli_(1 - dropout)
        mask = m.requires_grad_(False) / (1 - dropout)
        self.mask = mask
        return mask

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        assert self.mask is not None, "You need to reset mask before using VariationalHidDropout"
        return self.mask.expand_as(x) * x 

    
class VariationalHidDropout2dList(nn.Module):
    def __init__(self, dropout=0.0, spatial=True):
        """
        Hidden-to-hidden (VD-based) dropout that applies the same mask at every pixel and every layer
        :param dropout: The dropout rate (0 means no dropout is applied)
        """
        super(VariationalHidDropout2dList, self).__init__()
        self.dropout = dropout
        self.mask = None
        self.spatial = spatial

    def reset_mask(self, xs):
        dropout = self.dropout
        spatial = self.spatial
        
        self.mask = []
        for x in xs:
            # x has dimension (N, C, H, W)
            if spatial:
                m = torch.zeros_like(x[:,:,:1,:1]).bernoulli_(1 - dropout)
            else:
                m = torch.zeros_like(x).bernoulli_(1 - dropout)
            mask = m.requires_grad_(False) / (1 - dropout)
            self.mask.append(mask)
        return self.mask

    def forward(self, xs):
        if not self.training or self.dropout == 0:
            return xs
        assert self.mask is not None and len(self.mask) > 0, "You need to reset mask before using VariationalHidDropoutList"
        return [self.mask[i].expand_as(x) * x for i, x in enumerate(xs)]


class WeightFrobNormRestriction(object):

    _version: int = 1
    
    def __init__(self, names, sample_batch=64, dim=0, eps=1e-12, norm_range=1.):
        """
        Weight frob norm restriction module
        :param names: The list of weight names to apply weightnorm restriction on
        """
        self.names = names
        self.sample_batch = sample_batch
        self.dim = dim
        self.eps = eps
        self.norm_range = norm_range

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            weight_mat = weight_mat.permute(self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module, name):
        w = getattr(module, name + '_w')
        w_mat = self.reshape_weight_to_matrix(w)

        # estimate frob norm for w
        with torch.no_grad():
            h_, w_ = w_mat.size()
            v = torch.randn(self.sample_batch, h_).to(w.device)
            frob = torch.sqrt(torch.mean(torch.sum(torch.mm(v, w_mat)**2, dim=1)) + self.eps)
            frob_restrict = torch.min(frob, torch.tensor(self.norm_range))

        # restrict the frob norm
        w = w / frob * frob_restrict

        return w

    @staticmethod
    def apply(module, names, sample_batch=64, dim=0, eps=1e-12, norm_range=1.):
        fn = WeightFrobNormRestriction(names, sample_batch, dim, eps, norm_range)

        for name in names:
            weight = getattr(module, name)

            with torch.no_grad():
                weight_mat = fn.reshape_weight_to_matrix(weight)
                h, w = weight_mat.size()
                # first compute the frob norm of weight and restrict
                v = torch.randn(sample_batch, h).to(weight.device)
                frob = torch.sqrt(torch.mean(torch.sum(torch.mm(v, weight_mat), dim=1)) + eps)
                if frob > norm_range:
                    weight.data = weight.data / frob * norm_range

            # remove w from parameter list
            del module._parameters[name]

            module.register_parameter(name + '_w', Parameter(weight.data))
            setattr(module, name, fn.compute_weight(module, name))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(FrobNormRestrictionStateDictHook(fn))
        module._register_load_state_dict_pre_hook(FrobNormRestrictionLoadStateDictPreHook(fn))
        return fn

    def remove(self, module):
        for name in self.names:
            with torch.no_grad():
                weight = self.compute_weight(module, name)
            delattr(module, name)
            del module._parameters[name + '_w']
            module.register_parameter(name, Parameter(weight.data))

    def reset(self, module):
        for name in self.names:
            setattr(module, name, self.compute_weight(module, name))

    def __call__(self, module, inputs):
        pass


class FrobNormRestrictionLoadStateDictPreHook:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        fn = self.fn
        for name in fn.names:
            version = local_metadata.get('weight_frob_norm_restriction', {}).get(name + '.version', None)
            if version is None or version < 1:
                weight_key = prefix + name
                if version is None and all(weight_key + s in state_dict for s in ('_w')) and weight_key not in state_dict:
                    return
                has_missing_keys = False
                for suffix in ('_w'):
                    key = weight_key + suffix
                    if key not in state_dict:
                        has_missing_keys = True
                        if strict:
                            missing_keys.append(key)
                if has_missing_keys:
                    return


class FrobNormRestrictionStateDictHook:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        if 'weight_frob_norm_restriction' not in local_metadata:
            local_metadata['weight_frob_norm_restriction'] = {}
        for name in self.fn.names:
            key = name + '.version'
            if key in local_metadata['weight_frob_norm_restriction']:
                raise RuntimeError("Unexpected key in metadata['weight_frob_norm_restriction']: {}".format(key))
            local_metadata['weight_frob_norm_restriction'][key] = self.fn._version


def weight_frob_norm_restriction(module, names, sample_batch=64, dim=0, eps=1e-12, norm_range=1.):
    fn = WeightFrobNormRestriction.apply(module, names, sample_batch, dim, eps, norm_range)
    return module, fn
