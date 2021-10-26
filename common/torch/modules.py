import torch
from torch.distributions.chi2 import Chi2
from .utils import expand_as, softmax, is_cuda


# https://github.com/ultralytics/yolov5/issues/2136
class SiLU(torch.nn.Module):  # export-friendly version of nn.SiLU()
    def __init__(self, inplace=False):
        super(SiLU, self).__init__()  # init the base class
        # jsut dummy to accep tinplace argument
        pass
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


#Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
#https://arxiv.org/abs/1908.08681v1
#implemented for PyTorch / FastAI by lessw2020
#github: https://github.com/lessw2020/mish

class Mish(torch.nn.Module):
    def __init__(self, inplace=False):
        super(Mish, self).__init__()

    @staticmethod
    def forward(x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(torch.nn.functional.softplus(x)))


class View(torch.nn.Module):
    """
    Simple view layer.
    """

    def __init__(self, *args):
        """
        Constructor.

        :param args: shape
        :type args: [int]
        """

        super(View, self).__init__()

        self.shape = args

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        assert input.is_contiguous()

        return input.view(self.shape)


class ViewOrReshape(torch.nn.Module):
    """
    Simple view layer.
    """

    def __init__(self, *args):
        """
        Constructor.

        :param args: shape
        :type args: [int]
        """

        super(ViewOrReshape, self).__init__()

        self.shape = args

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        if input.is_contiguous():
            return input.view(self.shape)
        else:
            return input.reshape(self.shape)


class Clamp(torch.nn.Module):
    """
    Wrapper for clamp.
    """

    def __init__(self, min=0, max=1):
        """
        Constructor.
        """

        super(Clamp, self).__init__()

        self.min = min
        """ (float) Min value. """

        self.max = max
        """ (float) Max value. """

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return torch.clamp(torch.clamp(input, min=self.min), max=self.max)


class Scale(torch.nn.Module):
    """
    Simply scaling layer, mainly to allow simple saving and loading.
    """

    def __init__(self, shape, min=-1, max=1):
        """
        Constructor.

        :param shape: shape
        :type shape: [int]
        """

        super(Scale, self).__init__()

        self.register_buffer('scale_min', torch.ones(shape)*min)
        self.register_buffer('scale_max', torch.ones(shape)*max)

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return expand_as(self.scale_min, input) + torch.mul(expand_as(self.scale_max, input) - expand_as(self.scale_min, input), input)


class Entropy(torch.nn.Module):
    """
    Entropy computation based on logits.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(Entropy, self).__init__()

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return -1.*torch.sum(softmax(input, dim=1) * torch.nn.functional.log_softmax(input, dim=1))


class Normalize(torch.nn.Module):
    """
    Normalization layer to be learned.
    """

    def __init__(self, n_channels):
        """
        Constructor.

        :param n_channels: number of channels
        :type n_channels: int
        """

        super(Normalize, self).__init__()

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = torch.nn.Parameter(torch.ones(n_channels))
        self.bias = torch.nn.Parameter(torch.zeros(n_channels))
        # buffers are not saved in state dict!
        #self.register_buffer('std', torch.ones(n_channels))
        #self.register_buffer('mean', torch.zeros(n_channels))

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return (input - self.bias.view(1, -1, 1, 1))/self.weight.view(1, -1, 1, 1)


class Flatten(torch.nn.Module):
    """
    Flatten vector, allows to flatten without knowing batch_size and flattening size.
    """

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return input.view(input.size(0), -1)


class LeakyReLU(torch.nn.Module):
    """
    Just to redefine default negative slope.
    """

    def __init__(self, negative_slope=0.25, inplace=False):
        """
        Constructor.

        :param negative_slope: slope on negative side
        :type negative_slope: float
        :param inplace: in place operation
        :type inplace: bool
        """

        super(LeakyReLU, self).__init__()

        self.negative_slope = negative_slope
        """ (float) Negative slope. """

        self.inplace = inplace
        """ (bool) In place operation. """

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return torch.nn.functional.leaky_relu(input, self.negative_slope, self.inplace)


class LeakyTanh(torch.nn.Module):
    """
    Just to redefine default negative slope.
    """

    def __init__(self, negative_slope=0.25, inplace=False):
        """
        Constructor.

        :param negative_slope: slope on negative side
        :type negative_slope: float
        :param inplace: in place operation
        :type inplace: bool
        """

        super(LeakyTanh, self).__init__()

        self.negative_slope = negative_slope
        """ (float) Negative slope. """

        self.inplace = inplace
        """ (bool) In place operation. """

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return torch.min(input, torch.zeros_like(input))*self.negative_slope*torch.tanh(input) + torch.max(input, torch.zeros_like(input))*torch.tanh(input)


class ReparameterizedBatchNorm(torch.nn.Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps',
                     'num_features', 'affine']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ReparameterizedBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            torch.nn.init.zeros_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(ReparameterizedBatchNorm, self)._load_from_state_dict( state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return torch.nn.functional.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            1 + self.weight if self.affine else self.weight,
            self.bias, bn_training, exponential_average_factor, self.eps)


class ReparameterizedBatchNorm1d(ReparameterizedBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class ReparameterizedBatchNorm2d(ReparameterizedBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
