import common.summary
import common.datasets
from common.log import log, LogLevel
import attacks


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

        self.training = False
        """ (bool) Training mode. """

        self.input_norm = None
        """ (attacks.norms.Norm) Input norm. """

        self.weight_norm = None
        """ (attacks.weights.norms.Norm) Weight norm. """

        self.weight_projection = None
        """ (attacks.weights.projections.Projection) Weight projection. """

        self.get_layers = None
        """ (callable) Get layers to attack. """

        self.auxiliary = False
        """ (bool) Auxiliar layers to be included. """

    def quantize(self, model, quantized_model=None):
        """
        Quantization.
        """

        # Originally, this allowed to attack quantized models.
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

        if self.layers is None:
            if self.get_layers is not None:
                layers = self.get_layers(model)
                named_parameters = dict(model.named_parameters())
                named_parameters_keys = list(named_parameters.keys())
                self.layers = [i for i in layers if named_parameters[named_parameters_keys[i]].requires_grad is True and (named_parameters_keys[i].find('auxiliary') < 0 or self.auxiliary is True)]
                if 0 in self.layers or 1 in self.layers:
                    log('[Warning] layers 0,1 included in attack layers', LogLevel.WARNING)
            else:
                named_parameters = dict(model.named_parameters())
                named_parameters_keys = list(named_parameters.keys())
                self.layers = [i for i in range(len(named_parameters_keys)) if named_parameters[named_parameters_keys[i]].requires_grad is True and (named_parameters_keys[i].find('auxiliary') < 0 or self.auxiliary is True)]
                if 0 in self.layers or 1 in self.layers:
                    log('[Warning] layers 0,1 included in attack layers', LogLevel.WARNING)

        return self.layers

    def run(self, model, testset, weight_objective, input_objective):
        """
        Run attack.

        :param model: model to attack
        :type model: torch.nn.Module
        :param testset: images
        :type testset: torch.utils.data.DataLoader
        :param input_objective: objective
        :type input_objective: UntargetedObjective or TargetedObjective
        :param weight_objective: objective
        :type weight_objective: UntargetedObjective or TargetedObjective
        """

        assert model.training is False
        assert isinstance(input_objective, attacks.objectives.Objective)
        assert isinstance(weight_objective, attacks.weights.objectives.Objective)

        if self.writer is not None:
            self.writer.add_text('%sattack' % self.prefix, self.__class__.__name__)
            self.writer.add_text('%sinput_objective' % self.prefix, input_objective.__class__.__name__)
            self.writer.add_text('%sweight_objective' % self.prefix, weight_objective.__class__.__name__)

        self.layers_(model)

        if self.weight_projection is not None:
            self.weight_projection.reset()

