import torch
import common.torch
from .classifier import Classifier
from operator import mul
from functools import reduce
from .utils import get_normalization1d, get_activation


class MLP(Classifier):
    """
    MLP classifier.
    """

    def __init__(self, N_class, resolution=(1, 32, 32), units=[64, 64, 64], activation='relu', normalization='bn', dropout=False, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param activation: activation function
        :type activation: str
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param bias: whether to use bias
        :type bias: bool
        """

        super(MLP, self).__init__(N_class, resolution, **kwargs)

        self.units = units
        """ ([int]) Units. """

        self.normalization = normalization
        """ (callable) Normalization. """

        self.activation = activation
        """ (str) Activation. """

        self.dropout = dropout
        """ (str) Dropout. """

        activation_layer = get_activation(self.activation)
        assert activation_layer is not None

        # not overwriting self.units!
        units = [reduce(mul, self.resolution, 1)] + self.units
        view = common.torch.ViewOrReshape(-1, units[0])
        self.append_layer('view0', view)

        for layer in range(1, len(units)):
            in_features = units[layer - 1]
            out_features = units[layer]

            lin = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=self.include_bias)
            common.torch.kaiming_normal_(lin.weight, nonlinearity=activation, scale=self.init_scale)
            if self.include_bias:
                torch.nn.init.constant_(lin.bias, 0)
            self.append_layer('lin%d' % layer, lin)

            if activation_layer is not None:
                act = activation_layer()
                self.append_layer('act%d' % layer, act)

            self.append_layer('%s%d' % (self.normalization, layer), get_normalization1d(self.normalization, out_features))

        if self.dropout:
            drop = torch.nn.Dropout(p=0.1)
            self.append_layer('drop', drop)

        logits = torch.nn.Linear(units[-1], self._N_output, bias=self.include_bias)
        common.torch.kaiming_normal_(logits.weight, nonlinearity=activation, scale=self.init_scale)
        if self.include_bias:
            torch.nn.init.constant_(logits.bias, 0)
        self.append_layer('logits', logits)

    def __str__(self):
        """
        Print network.
        """

        string = super(MLP, self).__str__()
        string += '(units: %s)\n' % '-'.join(list(map(str, self.units)))
        string += '(activation: %s)\n' % self.activation
        string += '(normalization: %s)\n' % self.normalization
        string += '(dropout: %s)\n' % self.dropout

        return string