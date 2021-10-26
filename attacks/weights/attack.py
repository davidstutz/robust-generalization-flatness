import random
from .objectives import Objective
from .projections import Projection
from .initializations import Initialization
from common.log import log, LogLevel
import common.torch


class Attack:
    """
    Generic attack.
    """

    def __init__(self):
        """
        Constructor.
        """

        self.writer = None
        """ (common.summary.SummaryWriter) Summary writer or Tensorboard writer. """

        self.prefix = ''
        """ (str) Prefix for summary writer. """

        self.progress = None
        """ (common.progress.ProgressBar) Progress bar. """

        self.layers = None
        """ ([int]) Layers. """

        self.get_layers = None
        """ (callable) Get layers to attack. """

        self.initialization = None
        """ (Initialization) Initialization. """

        self.projection = None
        """ (Projection) Projection. """

        self.randomize_values = []
        """ (dict) Randomize attack part. """

        self.training = False
        """ (bool) Training mode. """

        self.norm = None
        """ (Norm) Norm. """

        self.auxiliary = False
        """ (bool) Auxiliar layers to be included. """

    def initialize(self, model, perturbed_model):
        """
        Initialization.
        """

        if self.initialization is not None:
            self.initialization(model, perturbed_model, self.layers)
        else:
            log('no initialization!', LogLevel.WARNING)

    def project(self, model, perturbed_model):
        """
        Projection.
        """

        if self.projection is not None:
            self.projection(model, perturbed_model, self.layers)

    def quantize(self, model, quantized_model=None):
        """
        Quantization.
        """

        # Originally this method allowed attacking quantized models.
        if quantized_model is not None:
            parameters = list(model.parameters())
            quantized_parameters = list(quantized_model.parameters())
            assert len(parameters) == len(quantized_parameters)

            for i in range(len(parameters)):
                quantized_parameters[i].data = parameters[i].data

        return common.torch.clone(model), None

    def layers_(self, model):
        """
        Get layers for attack.

        :param model: model to attack
        :type model: torch.nn.Module
        :return: layers
        :rtype: [int]
        """

        if self.get_layers is not None:
            layers = self.get_layers(model)
            named_parameters = dict(model.named_parameters())
            named_parameters_keys = list(named_parameters.keys())
            self.layers = [i for i in layers if named_parameters[named_parameters_keys[i]].requires_grad is True and (named_parameters_keys[i].find('auxiliary') < 0 or self.auxiliary is True)]
        else:
            named_parameters = dict(model.named_parameters())
            named_parameters_keys = list(named_parameters.keys())
            self.layers = [i for i in range(len(named_parameters_keys)) if named_parameters[named_parameters_keys[i]].requires_grad is True and (named_parameters_keys[i].find('auxiliary') < 0 or self.auxiliary is True)]
        if 0 in self.layers or 1 in self.layers:
            log('[Warning] layers 0,1 included in attack layers', LogLevel.WARNING)

        return self.layers

    def run(self, model, testset, objective):
        """
        Run attack.

        :param model: model to attack
        :type model: torch.nn.Module
        :param testset: images
        :type testset: torch.utils.data.DataLoader
        :param objective: objective
        :type objective: UntargetedObjective or TargetedObjective
        """

        assert isinstance(objective, Objective)
        assert self.projection is None or isinstance(self.projection, Projection)
        assert self.initialization is None or isinstance(self.initialization, Initialization)

        if self.writer is not None:
            self.writer.add_text('%sattack' % self.prefix, self.__class__.__name__)
            self.writer.add_text('%sobjective' % self.prefix, objective.__class__.__name__)

        if len(self.randomize_values) > 0:
            index = random.choice(range(len(self.randomize_values)))
            array = self.randomize_values[index]
            for key in array:
                setattr(self, key, array[key])

        self.layers_(model)

        if self.projection is not None:
            self.projection.reset()