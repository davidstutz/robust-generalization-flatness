import torch
import common.summary
import common.torch
from common.log import log
from .progress import ProgressBar


def reset(model):
    """
    Reset BN statistics.

    :param model: model
    :type model: torch.mm.Module
    """

    if hasattr(model, 'running_var'):
        model.running_var.data.fill_(1)
    if hasattr(model, 'running_mean'):
        model.running_mean.data.fill_(0)
    if hasattr(model, 'num_batches_tracked'):
        model.num_batches_tracked.data.fill_(0)

    for module in model.children():
        reset(module)


def momentum(model, value):
    """
    Set momentum for BN.

    See https://github.com/pytorch/pytorch/blob/fa153184c8f70259337777a1fd1d803c7325f758/aten/src/ATen/native/Normalization.cpp.

    :param model: model
    :type model: torch.nn.Module
    :param value: momentum value
    :type value: value
    """

    if hasattr(model, 'momentum'):
        model.momentum = value

    for module in model.children():
        momentum(module, value)


def calibrate(model, trainset, testset, augmentation=None, epochs=5, cuda=False):
    """
    Attack model.

    :param model: model
    :type model: torch.nn.Module
    :param trainset: test set
    :type trainset: torch.utils.data.DataLoader
    :param augmentation: data augmentation
    :param epochs: number of attempts
    :type epochs: int
    :param cuda: whether to use CUDA
    :type cuda: bool
    """

    assert model.training is False
    assert len(trainset) > 0
    assert epochs > 0
    assert isinstance(trainset, torch.utils.data.DataLoader)
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    progress = ProgressBar()
    reset(model)
    layers = model.layers()

    for epoch in range(epochs):
        model.train()
        assert model.training is True

        for b, (inputs, targets) in enumerate(trainset):
            if augmentation is not None:
                inputs = augmentation.augment_images(inputs.numpy())

            inputs = common.torch.as_variable(inputs, cuda)
            inputs = inputs.permute(0, 3, 1, 2)
            assert len(targets.shape) == 1
            targets = common.torch.as_variable(targets, cuda)
            assert len(list(targets.size())) == 1

            logits = model(inputs)

            loss = common.torch.classification_loss(logits, targets)
            error = common.torch.classification_error(logits, targets)

            for layer in layers:
                if layer.find('bn') >= 0:
                    break

            progress('calibrate train %d' % epoch, b, len(trainset), info='advloss=%g adverr=%g' % (
                loss,
                error,
            ))

        model.eval()
        assert model.training is False

        for b, (inputs, targets) in enumerate(testset):
            inputs = common.torch.as_variable(inputs, cuda)
            inputs = inputs.permute(0, 3, 1, 2)
            assert len(targets.shape) == 1
            targets = common.torch.as_variable(targets, cuda)
            assert len(list(targets.size())) == 1

            logits = model(inputs)

            loss = common.torch.classification_loss(logits, targets)
            error = common.torch.classification_error(logits, targets)

            progress('calibrate test %d' % epoch, b, len(trainset), info='advloss=%g adverr=%g' % (
                loss,
                error,
            ))
