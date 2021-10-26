import torch
import common.torch
from .classifier import Classifier
from .utils import get_normalization2d, get_activation


class LeNet(Classifier):
    """
    LeNet classifier.
    """

    def __init__(self, N_class, resolution=(1, 32, 32), channels=64, activation='relu', normalization='bn', linear=1024, dropout=False, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param channels: channels to start with
        :type channels: int
        :param units: units per layer
        :type units: [int]
        :param activation: activation function
        :type activation: str
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param bias: whether to use bias
        :type bias: bool
        """

        super(LeNet, self).__init__(N_class, resolution, **kwargs)

        # the constructor parameters must be available as attributes for state to work
        self.channels = channels
        """ (int) Channels. """

        self.activation = activation
        """ (str) Activation. """

        self.linear = linear
        """ (int) Additional linear layer. """

        self.dropout = dropout
        """ (str) Dropout. """

        activation_layer = get_activation(self.activation)
        assert activation_layer is not None

        self.normalization = normalization
        """ (bool) Normalization. """

        layer = 0
        layers = []
        resolutions = []

        while True:
            input_channels = self.resolution[0] if layer == 0 else layers[layer - 1]
            output_channels = self.channels if layer == 0 else layers[layer - 1] * 2

            conv = torch.nn.Conv2d(input_channels, output_channels, kernel_size=5, stride=1, padding=2, bias=self.include_bias)
            #torch.nn.init.normal_(conv.weight, mean=0, std=0.1)
            common.torch.kaiming_normal_(conv.weight, nonlinearity=activation, scale=self.init_scale)
            if self.include_bias:
                torch.nn.init.constant_(conv.bias, 0)
            self.append_layer('conv%d' % layer, conv)

            self.append_layer('%s%d' % (self.normalization, layer), get_normalization2d(self.normalization, output_channels))

            if self.activation:
                relu = activation_layer()
                self.append_layer('act%d' % layer, relu)

            pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.append_layer('pool%d' % layer, pool)

            layers.append(output_channels)
            resolutions.append([
                self.resolution[1] // 2 if layer == 0 else resolutions[layer - 1][0] // 2,
                self.resolution[2] // 2 if layer == 0 else resolutions[layer - 1][1] // 2,
            ])

            if resolutions[-1][0] // 2 < 3 or resolutions[-1][0] % 2 == 1 or resolutions[-1][1] // 2 < 3 or resolutions[-1][1] % 2 == 1:
                break

            layer += 1

        representation = int(resolutions[-1][0] * resolutions[-1][1] * layers[-1])
        assert representation > 0
        view = common.torch.ViewOrReshape(-1, representation)
        self.append_layer('view', view)

        if self.linear > 0:
            fc = torch.nn.Linear(representation, self.linear, bias=self.include_bias)
            common.torch.kaiming_normal_(fc.weight, nonlinearity=activation, scale=self.init_scale)
            if self.include_bias:
                torch.nn.init.constant_(fc.bias, 0)
            self.append_layer('fc%d' % layer, fc)

            if self.activation:
                relu = activation_layer()
                self.append_layer('act%d' % layer, relu)

        if self.dropout:
            drop = torch.nn.Dropout(p=0.5)
            self.append_layer('drop', drop)

        logits = torch.nn.Linear(self.linear if self.linear > 0 else representation, self._N_output, bias=self.include_bias)
        common.torch.kaiming_normal_(logits.weight, nonlinearity=activation, scale=self.init_scale)
        if self.include_bias:
            torch.nn.init.constant_(logits.bias, 0)
        self.append_layer('logits', logits)

    def __str__(self):
        """
        Print network.
        """

        string = super(LeNet, self).__str__()
        string += '(channels: %d)\n' % self.channels
        string += '(activation: %s)\n' % self.activation
        string += '(normalization: %s)\n' % self.normalization
        string += '(linear: %d)\n' % self.linear
        string += '(dropout: %s)\n' % self.dropout

        return string