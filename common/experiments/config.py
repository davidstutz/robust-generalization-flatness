import attacks
import torch.utils.data
import common.torch
from common.log import log, LogLevel


class NormalTrainingConfig:
    """
    Configuration for normal training.
    """

    def __init__(self):
        """
        Constructor.
        """

        self.directory = None

        # Fixed parameters
        self.cuda = True
        self.augmentation = None
        self.loss = common.torch.classification_loss
        self.trainloader = None
        self.testloader = None
        self.epochs = None
        self.snapshot = None
        self.finetune = None
        self.projection = None

        # weight averaging
        self.keep_average = False
        self.keep_average_tau = 0.9975

        # Writer depends on log directory
        self.get_writer = None
        # Optimizer is based on parameters
        self.get_optimizer = None
        # Scheduler is based on optimizer
        self.get_scheduler = None
        # Model is based on data resolution
        self.get_model = None

        self.summary_histograms = False
        self.summary_weights = False
        self.summary_images = False

    def validate(self):
        """
        Check validity.
        """

        assert self.directory is not None
        assert len(self.directory) > 0
        assert isinstance(self.trainloader, torch.utils.data.DataLoader)
        assert len(self.trainloader) > 0
        assert isinstance(self.testloader, torch.utils.data.DataLoader)
        assert len(self.testloader) > 0
        assert self.epochs > 0
        assert self.snapshot is None or self.snapshot > 0
        assert callable(self.get_optimizer)
        assert callable(self.get_scheduler)
        assert callable(self.get_model)
        assert callable(self.get_writer)
        assert self.loss is not None
        assert callable(self.loss)


class AdversarialTrainingConfig(NormalTrainingConfig):
    """
    Configuration for adversarial training.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(AdversarialTrainingConfig, self).__init__()

        # Fixed parameters
        self.attack = None
        self.objective = None
        self.fraction = None
        self.prevent_label_leaking = False
        self.ignore_incorrect = False

    def validate(self):
        """
        Check validity.
        """

        super(AdversarialTrainingConfig, self).validate()

        assert isinstance(self.attack, attacks.Attack)
        assert isinstance(self.objective, attacks.objectives.Objective)
        assert self.fraction > 0 and self.fraction <= 1


class ConfidenceCalibratedAdversarialTrainingConfig(AdversarialTrainingConfig):
    """
    Configuration for adversarial training.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(ConfidenceCalibratedAdversarialTrainingConfig, self).__init__()

        # Fixed parameters
        self.transition = None

    def validate(self):
        """
        Check validity.
        """

        super(ConfidenceCalibratedAdversarialTrainingConfig, self).validate()

        assert callable(self.transition)


class AdversarialWeightsInputsTrainingConfig(NormalTrainingConfig):
    """
    Configuration for adversarial training.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(AdversarialWeightsInputsTrainingConfig, self).__init__()

        # Fixed parameters
        self.weight_attack = None
        self.weight_objective = None
        self.input_attack = None
        self.input_objective = None
        self.curriculum = None
        self.average_statistics = False
        self.adversarial_statistics = False
        self.gradient_clipping = 0.05
        self.reset_iterations = 1
        self.operators = None
        self.clean = False

    def validate(self):
        """
        Check validity.
        """

        super(AdversarialWeightsInputsTrainingConfig, self).validate()

        assert isinstance(self.weight_attack, attacks.weights.Attack)
        assert isinstance(self.weight_objective, attacks.weights.objectives.Objective)
        assert isinstance(self.input_attack, attacks.Attack)
        assert isinstance(self.input_objective, attacks.objectives.Objective)

class JointAdversarialWeightsInputsTrainingConfig(NormalTrainingConfig):
    """
    Configuration for adversarial training.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(JointAdversarialWeightsInputsTrainingConfig, self).__init__()

        # Fixed parameters
        self.attack = None
        self.weight_objective = None
        self.input_objective = None
        self.curriculum = None
        self.average_statistics = False
        self.adversarial_statistics = False
        self.gradient_clipping = 0.05
        self.reset_iterations = 1

    def validate(self):
        """
        Check validity.
        """

        super(JointAdversarialWeightsInputsTrainingConfig, self).validate()

        assert isinstance(self.attack, attacks.weights_inputs.Attack)
        assert isinstance(self.weight_objective, attacks.weights.objectives.Objective)
        assert isinstance(self.input_objective, attacks.objectives.Objective)


class SemiSupervisedTrainingConfig(NormalTrainingConfig):
    """
    Configuration for adversarial training.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(SemiSupervisedTrainingConfig, self).__init__()

        self.unsup_weight = 1
        self.unsup_loss = common.torch.classification_loss
        self.get_auxiliary_model = None
        self.confidence_threshold = 0

    def validate(self):
        """
        Check validity.
        """

        super(SemiSupervisedTrainingConfig, self).validate()

        assert self.unsup_weight > 0
        assert callable(self.unsup_loss)


class AdversarialSemiSupervisedTrainingConfig(AdversarialTrainingConfig):
    """
    Configuration for adversarial training.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(AdversarialSemiSupervisedTrainingConfig, self).__init__()

        self.unsup_weight = 1
        self.unsup_loss = common.torch.classification_loss
        self.get_auxiliary_model = None
        self.confidence_threshold = 0

    def validate(self):
        """
        Check validity.
        """

        super(AdversarialSemiSupervisedTrainingConfig, self).validate()

        assert self.unsup_weight > 0
        assert callable(self.unsup_loss)


class AdversarialMatchTrainingConfig(NormalTrainingConfig):
    """
    Configuration for adversarial training.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(AdversarialMatchTrainingConfig, self).__init__()

        self.unsup_weight = 1
        self.unsup_loss = common.torch.classification_loss
        self.confidence_threshold = 0
        self.attack = None
        self.objective = None

    def validate(self):
        """
        Check validity.
        """

        super(AdversarialMatchTrainingConfig, self).validate()

        assert self.attack is not None
        assert self.objective is not None


class AttackConfig:
    """
    Configuration for attacks.
    """

    def __init__(self):
        """
        Constructor.
        """

        self.directory = None

        # Fixed parameters
        self.testloader = None
        self.attack = None
        self.objective = None
        self.attempts = None
        self.snapshot = None
        self.model_specific = False

        # Depends on directory
        self.get_writer = None

    def validate(self):
        """
        Check validity.
        """

        assert self.directory is not None
        assert len(self.directory) > 0
        assert isinstance(self.testloader, torch.utils.data.DataLoader)
        assert len(self.testloader) > 0
        assert isinstance(self.attack, attacks.Attack), self.attack
        assert isinstance(self.objective, attacks.objectives.Objective)
        assert callable(self.get_writer)


class AttackWeightsConfig:
    """
    Configuration for attacks.
    """

    def __init__(self):
        """
        Constructor.
        """

        self.directory = None

        # Fixed parameters
        self.trainloader = None
        self.testloader = None
        self.attack = None
        self.objective = None
        self.attempts = None
        self.snapshot = None
        self.eval = True
        self.operators = None
        self.model_specific = False
        self.save_models = False

        # Depends on directory
        self.get_writer = None

    def validate(self):
        """
        Check validity.
        """

        assert self.directory is not None
        assert len(self.directory) > 0
        assert isinstance(self.trainloader, torch.utils.data.DataLoader)
        assert isinstance(self.testloader, torch.utils.data.DataLoader)
        assert len(self.testloader) > 0
        assert isinstance(self.attack, attacks.weights.Attack), self.attack
        assert isinstance(self.objective, attacks.weights.objectives.Objective)
        assert callable(self.get_writer)
        if self.eval is not True:
            log('[Warning] model not in eval for attack', LogLevel.WARNING)


class AttackWeightsInputsConfig:
    """
    Configuration for attacks.
    """

    def __init__(self):
        """
        Constructor.
        """

        self.directory = None

        # Fixed parameters
        self.testloader = None
        self.attack = None
        self.weight_objective = None
        self.input_objective = None
        self.attempts = None
        self.snapshot = None
        self.operators = None

        # Depends on directory
        self.get_writer = None

    def validate(self):
        """
        Check validity.
        """

        assert self.directory is not None
        assert len(self.directory) > 0
        assert isinstance(self.testloader, torch.utils.data.DataLoader)
        assert len(self.testloader) > 0
        assert isinstance(self.attack, attacks.weights_inputs.Attack), self.attack
        assert isinstance(self.weight_objective, attacks.weights.objectives.Objective)
        assert isinstance(self.input_objective, attacks.objectives.Objective)
        assert callable(self.get_writer)
