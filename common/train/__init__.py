import torch
import math
from .normal_training import NormalTraining
from .adversarial_training import AdversarialTraining
from .entropy_adversarial_training import EntropyAdversarialTraining
from .adversarial_weights_inputs_training import AdversarialWeightsInputsTraining
from .semi_supervised_training import SemiSupervisedTraining
from .adversarial_semi_supervised_training import AdversarialSemiSupervisedTraining
from .mart_adversarial_training import MARTAdversarialTraining
from .trades_adversarial_training import TRADESAdversarialTraining


def get_cyclic_scheduler(optimizer, batches_per_epoch, base_lr, max_lr, step_size_up, step_size_down):
    """
    Get cyclic learning rate.
    """

    return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=batches_per_epoch*step_size_up, step_size_down=batches_per_epoch*step_size_down)


def get_exponential_scheduler(optimizer, batches_per_epoch, gamma=0.97):
    """
    Get exponential scheduler.

    Note that the resulting optimizer's step function is called after each batch!

    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param batches_per_epoch: number of batches per epoch
    :type batches_per_epoch: int
    :param gamma: gamma
    :type gamma: float
    :return: scheduler
    :rtype: torch.optim.lr_scheduler.LRScheduler
    """

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda epoch: gamma ** math.floor(epoch/batches_per_epoch)])


def get_no_scheduler(optimizer, batches_per_epoch):
    """
    No, i.e., "empty", scheduler.
    """

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda epoch: 1])


def get_step_scheduler(optimizer, batches_per_epoch, step_size=50, gamma=0.1):
    """
    Get step scheduler.

    Note that the resulting optimizer's step function is called after each batch!

    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param batches_per_epoch: number of batches per epoch
    :type batches_per_epoch: int
    :param gamma: gamma
    :type gamma: float
    :return: scheduler
    :rtype: torch.optim.lr_scheduler.LRScheduler
    """

    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size*batches_per_epoch, gamma=gamma)


def get_multi_step_scheduler(optimizer, batches_per_epoch, milestones=[100, 150, 200], gamma=0.1):
    """
    Get step scheduler.

    Note that the resulting optimizer's step function is called after each batch!

    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param batches_per_epoch: number of batches per epoch
    :type batches_per_epoch: int
    :param gamma: gamma
    :type gamma: float
    :return: scheduler
    :rtype: torch.optim.lr_scheduler.LRScheduler
    """

    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone*batches_per_epoch for milestone in milestones], gamma=gamma)
