import torch
import common.torch


def get_activation(activation):
    activation_layer = None
    if activation == 'relu':
        activation_layer = torch.nn.ReLU
    elif activation == 'sigmoid':
        activation_layer = torch.nn.Sigmoid
    elif activation == 'tanh':
        activation_layer = torch.nn.Tanh
    elif activation == 'leaky_relu':
        activation_layer = common.torch.LeakyReLU
    elif activation == 'leaky_tanh':
        activation_layer = common.torch.LeakyTanh
    elif activation == 'softsign':
        activation_layer = torch.nn.Softsign
    elif activation == 'silu':
        activation_layer = common.torch.SiLU
    elif activation == 'softplus':
        activation_layer = torch.nn.Softplus
    elif activation == 'mish':
        activation_layer = common.torch.Mish
    elif activation == 'gelu':
        activation_layer = torch.nn.GELU
    return activation_layer


def get_normalization2d(normalization, planes):
    assert normalization in [
        '',
        'rebn',
        'bn',
        'fixedbn',
        'nregn',
        'fixednregn',
        'regn',
        'gn',
        'fixedgn',
    ]

    num_group_alternatives = [32, 24, 16, 8]
    for i in range(len(num_group_alternatives)):
        num_groups = min(num_group_alternatives[i], planes // 2)
        if planes % num_groups == 0:
            break
    assert planes % num_groups == 0

    norm = torch.nn.Identity()
    if normalization == 'bn':
        norm = torch.nn.BatchNorm2d(planes)
        torch.nn.init.constant_(norm.weight, 1)
        torch.nn.init.constant_(norm.bias, 0)

    elif normalization == 'fixedbn':
        norm = torch.nn.BatchNorm2d(planes, affine=False)

    elif normalization == 'rebn':
        norm = common.torch.ReparameterizedBatchNorm2d(planes)
        torch.nn.init.constant_(norm.weight, 0)
        torch.nn.init.constant_(norm.bias, 0)

    elif normalization == 'gn':
        norm = torch.nn.GroupNorm(num_groups, planes)
        torch.nn.init.constant_(norm.weight, 1)
        torch.nn.init.constant_(norm.bias, 0)

    elif normalization == 'fixedgn':
        norm = torch.nn.GroupNorm(num_groups, planes, affine=False)

    elif normalization == 'regn':
        norm = common.torch.ReparameterizedGroupNorm(num_groups, planes)
        torch.nn.init.constant_(norm.weight, 0)
        torch.nn.init.constant_(norm.bias, 0)

    assert isinstance(norm, torch.nn.Identity) or norm != ''
    return norm


def get_normalization1d(normalization, out_features):
    assert normalization in [
        '',
        'rebn',
        'bn',
        'fixedbn',
    ]

    norm = torch.nn.Identity()
    if normalization == 'bn':
        norm = torch.nn.BatchNorm1d(out_features)
        torch.nn.init.constant_(norm.weight, 1)
        torch.nn.init.constant_(norm.bias, 0)

    elif normalization == 'fixedbn':
        norm = torch.nn.BatchNorm1d(out_features, affine=False)

    elif normalization == 'rebn':
        norm = common.torch.ReparameterizedBatchNorm1d(out_features)
        torch.nn.init.constant_(norm.weight, 0)
        torch.nn.init.constant_(norm.bias, 0)

    assert isinstance(norm, torch.nn.Identity) or norm != ''
    return norm