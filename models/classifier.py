import torch
from .resnet_block import ResNetBlock
import common.torch
from common.log import log, LogLevel


class Classifier(torch.nn.Module):
    """
    Simple classifier.
    """

    def __init__(self, N_class, resolution, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution
        :type resolution: [int]
        """

        super(Classifier, self).__init__()

        assert N_class > 0, 'positive N_class expected'
        assert len(resolution) <= 3

        self.N_class = int(N_class)  # Having strange bug where torch complaints about numpy.in64 being passed to nn.Linear.
        """ (int) Number of classes. """

        self.resolution = list(resolution)
        """ ([int]) Resolution as (channels, height, width) """

        # __ attributes are private, which is important for the State to work properly.
        self.__layers = []
        """ ([str]) Will hold layer names. """

        self.kwargs = kwargs
        """ (dict) Kwargs. """

        self.include_bias = self.kwargs.get('bias', True)
        """ (bool) Bias. """

        self.include_clamp = self.kwargs.get('clamp', True)
        """ (bool) Clamp. """

        self.include_whiten = self.kwargs.get('whiten', False)
        """ (bool) Whiten. """

        self.include_rescale = self.kwargs.get('rescale', False)
        """ (bool) Re-Scale. """

        self.init_scale = self.kwargs.get('init_scale', 1)
        """ (float) Init scale. """

        self.auxiliary = self.kwargs.get('auxiliary', None)
        """ (torch.nn.Module) Auxiliary model. """

        self._N_output = self.N_class if self.N_class > 2 else 1
        """ (int) Number of outputs. """

        if self.include_clamp:
            self.append_layer('clamp', common.torch.Clamp())

        assert not (self.include_whiten and self.include_rescale)

        if self.include_whiten:
            whiten = common.torch.Normalize(resolution[0])
            self.append_layer('whiten', whiten)
            whiten.weight.requires_grad = False
            whiten.bias.requires_grad = False
            log('[Warning] normalization added, freeze model.whiten weight and bias if training')

        if self.include_rescale:
            rescale = common.torch.Scale(1, min=-1, max=1)
            self.append_layer('rescale', rescale)

        self.scale = self.kwargs.get('scale', 1)
        """ (float) Logit scaling. """

        self.operators = kwargs.get('operators', [])
        """ ([attacks.activations.operator.Operator]) Operators on activations. """

    def append_layer(self, name, layer):
        """
        Add a layer.

        :param name: layer name
        :type name: str
        :param layer: layer
        :type layer: torch.nn.Module
        """

        setattr(self, name, layer)
        self.__layers.append(name)

    def prepend_layer(self, name, layer):
        """
        Add a layer.

        :param name: layer name
        :type name: str
        :param layer: layer
        :type layer: torch.nn.Module
        """

        self.insert_layer(0, name, layer)

    def insert_layer(self, index, name, layer):
        """
        Add a layer.

        :param index: index
        :type index: int
        :param name: layer name
        :type name: str
        :param layer: layer
        :type layer: torch.nn.Module
        """

        setattr(self, name, layer)
        self.__layers.insert(index, name)

    def forward(self, image, return_features=False, operators=None, auxiliary=False):
        """
        Forward pass, takes an image and outputs the predictions.

        :param image: input image
        :type image: torch.autograd.Variable
        :param return_features: whether to also return representation layer
        :type return_features: bool
        :return: logits
        :rtype: torch.autograd.Variable
        """

        if auxiliary:
            assert self.auxiliary is not None

        def has_parameters(model, name):
            return getattr(getattr(model, name), 'weight', None) is not None or getattr(getattr(model, name), 'bias', None) is not None

        features = []
        output = image

        if self.include_whiten:
            assert self.whiten.weight.requires_grad is False
            assert self.whiten.bias.requires_grad is False

        auxiliary_output = None
        if return_features:
            for i in range(len(self.__layers)):
                name = self.__layers[i]

                if name == 'logits':
                    auxiliary_output = None
                    if auxiliary and self.auxiliary is not None:
                        auxiliary_output = self.auxiliary(output)

                output = getattr(self, name)(output)
                for operator in self.operators:
                    output = operator(output, i)
                if operators is not None:
                    for operator in operators:
                        output = operator(output, i)
                features.append(output)

            if auxiliary:
                return self.scale * output, features, auxiliary_output
            else:
                return self.scale * output, features
        else:
            for i in range(len(self.__layers)):
                name = self.__layers[i]

                if name == 'logits':
                    if auxiliary and self.auxiliary is not None:
                        auxiliary_output = self.auxiliary(output)

                output = getattr(self, name)(output)

                for operator in self.operators:
                    output = operator(output, i)
                if operators is not None:
                    for operator in operators:
                        output = operator(output, i)

            if auxiliary:
                return self.scale * output, auxiliary_output
            else:
                return self.scale * output

    def layers(self):
        """
        Get layer names.

        :return: layer names
        :rtype: [str]
        """

        return self.__layers

    def __str__(self):
        """
        Print network.
        """

        n, _, _, _ = common.torch.parameter_sizes(self)
        string = '(W: %d)\n' % n
        string += '(N_class: %d)\n' % self.N_class
        string += '(resolution: %s)\n' % 'x'.join(list(map(str, self.resolution)))
        string += '(include_bias: %s)\n' % self.include_bias
        string += '(include_clamp: %s)\n' % self.include_clamp
        string += '(include_whiten: %s)\n' % self.include_whiten
        if self.include_whiten:
            string += '\t(weight=%s)\n' % self.whiten.weight.data.detach().cpu().numpy()
            string += '\t(bias=%s)\n' % self.whiten.bias.data.detach().cpu().numpy()
        string += '(include_rescale: %s)\n' % self.include_rescale
        string += '(init_scale: %g)\n' % self.init_scale
        string += '(scale: %g)\n' % self.scale
        for operator in self.operators:
            string += '(operators: %s)\n' % operator.__class__.__name__

        def module_description(module):
            ret = '(' + name + ', ' + module.__class__.__name__
            weight = getattr(module, 'weight', None)

            if getattr(module, 'in_channels', None) is not None:
                ret += ', in_channels=%d' % module.in_channels
            if getattr(module, 'out_channels', None) is not None:
                ret += ', out_channels=%s' % module.out_channels
            if getattr(module, 'in_features', None) is not None:
                ret += ', in_features=%d' % module.in_features
            if getattr(module, 'out_features', None) is not None:
                ret += ', out_features=%s' % module.out_features

            if weight is not None:
                ret += ', weight=%g,%g+-%g,%g' % (
                torch.min(weight).item(), torch.mean(weight).item(), torch.std(weight).item(), torch.max(weight).item())
            bias = getattr(module, 'bias', None)
            if bias is not None:
                ret += ', bias=%g,%g+-%g,%g' % (
                torch.min(bias).item(), torch.mean(bias).item(), torch.std(bias).item(), torch.max(bias).item())
            ret += ')\n'

            return ret

        for name in self.__layers:
            module = getattr(self, name)
            string += module_description(module)

            if isinstance(getattr(self, name), torch.nn.Sequential) or isinstance(getattr(self, name), ResNetBlock):
                for module in getattr(self, name).modules():
                    string += '\t' + module_description(module)

        if self.auxiliary is not None:
            string += str(self.auxiliary)

        return string

