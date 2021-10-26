import torch
import common.torch
from .utils import get_normalization2d, get_activation


class ResNetBlock(torch.nn.Module):
    """
    ResNet block taken from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, normalization='bn', init_scale=1, include_bias=False, zero_residuals=True, activation='relu', dropout=False):
        """
        Constructor.

        :param inplanes: input channels
        :type inplanes: int
        :param planes: output channels
        :type planes: int
        :param stride: stride
        :type stride: int
        :param downsample: whether to downsample
        :type downsample: bool
        :param normalization: whether to use normalization
        :type normalization: bool
        """

        super(ResNetBlock, self).__init__()

        self.activation = activation
        """ (str) Activation. """

        self.dropout = dropout
        """ (bool) Dropout. """

        activation_layer = get_activation(self.activation)
        assert activation_layer is not None

        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        common.torch.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity=activation, scale=init_scale)
        self.norm1 = get_normalization2d(normalization, planes)
        self.relu1 = activation_layer()

        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        common.torch.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity=activation, scale=init_scale)

        if self.dropout:
            self.drop2 = torch.nn.Dropout(p=0.1)

        self.norm2 = get_normalization2d(normalization, planes)
        if zero_residuals and not isinstance(self.norm2, torch.nn.Identity):
            if self.norm2.weight is not None:
                torch.nn.init.constant_(self.norm2.weight, 0)
        self.relu2 = activation_layer()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass.

        :param x: input
        :type x: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)

        if self.dropout:
            out = self.drop2(out)

        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out += identity
        out = self.relu2(out)

        return out


class ResNetBottleneckBlock(torch.nn.Module):
    """
    ResNet block taken from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, normalization='bn', init_scale=1, include_bias=False, zero_residuals=True, activation='relu', dropout=False):
        super(ResNetBottleneckBlock, self).__init__()

        self.activation = activation
        """ (str) Activation. """

        assert dropout is False

        activation_layer = get_activation(self.activation)
        assert activation_layer is not None

        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        common.torch.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity=activation, scale=init_scale)
        self.norm1 = get_normalization2d(normalization, planes)
        self.relu1 = activation_layer()

        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        common.torch.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity=activation, scale=init_scale)
        self.norm2 = get_normalization2d(normalization, planes)
        self.relu2 = activation_layer()

        self.conv3 = torch.nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        common.torch.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity=activation, scale=init_scale)
        self.norm3 = get_normalization2d(normalization, self.expansion * planes)
        if zero_residuals and not isinstance(self.norm2, torch.nn.Identity):
            if self.norm2.weight is not None:
                torch.nn.init.constant_(self.norm3.weight, 0)
        self.relu3 = activation_layer()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out