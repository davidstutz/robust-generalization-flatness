import torch
import numpy
import common.torch
import common.numpy
import common.summary
from common.progress import ProgressBar


def test(model, testset, eval=True, loss=True, operators=None, cuda=False):
    """
    Test a model on a clean or adversarial dataset.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param cuda: use CUDA
    :type cuda: bool
    """

    assert model.training is not eval
    assert len(testset) > 0
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    progress = ProgressBar()
    probabilities = None

    # should work with and without labels
    for b, data in enumerate(testset):
        targets = None
        if isinstance(data, tuple) or isinstance(data, list):
            inputs = data[0]
            targets = data[1]
        else:
            inputs = data

        assert isinstance(inputs, torch.Tensor)

        inputs = common.torch.as_variable(inputs, cuda)
        targets = common.torch.as_variable(targets, cuda)
        inputs = inputs.permute(0, 3, 1, 2)

        logits = model.forward(inputs, operators=operators)

        probabilities_ = common.torch.softmax(logits, dim=1).detach().cpu().numpy()
        probabilities = common.numpy.concatenate(probabilities, probabilities_)

        if targets is not None and loss:
            targets = common.torch.as_variable(targets, cuda)
            error = common.torch.classification_error(logits, targets)
            loss = common.torch.classification_loss(logits, targets)
            progress('test', b, len(testset), info='error=%g loss=%g' % (error.item(), loss.item()))
        else:
            progress('test', b, len(testset))

    return probabilities


def logits(model, testset, eval=True, loss=True, operators=None, cuda=False):
    """
    Test a model on a clean or adversarial dataset.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param cuda: use CUDA
    :type cuda: bool
    """

    assert model.training is not eval
    assert len(testset) > 0
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    progress = ProgressBar()
    logits = None

    # should work with and without labels
    for b, data in enumerate(testset):
        targets = None
        if isinstance(data, tuple) or isinstance(data, list):
            inputs = data[0]
            targets = data[1]
        else:
            inputs = data

        assert isinstance(inputs, torch.Tensor)

        inputs = common.torch.as_variable(inputs, cuda)
        targets = common.torch.as_variable(targets, cuda)
        inputs = inputs.permute(0, 3, 1, 2)

        logits_ = model(inputs, operators=operators)
        logits = common.numpy.concatenate(logits, logits_.detach().cpu().numpy())

        if targets is not None and loss:
            targets = common.torch.as_variable(targets, cuda)
            error = common.torch.classification_error(logits_, targets)
            loss = common.torch.classification_loss(logits_, targets)
            progress('test', b, len(testset), info='error=%g loss=%g' % (error.item(), loss.item()))
        else:
            progress('test', b, len(testset))

    return logits


def features(model, testset, eval=True, loss=True, operators=None, cuda=False, limit=False):
    """
    Test a model on a clean or adversarial dataset.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param cuda: use CUDA
    :type cuda: bool
    """

    assert model.training is not eval
    assert len(testset) > 0
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    progress = ProgressBar()
    probabilities = None
    features = None

    # should work with and without labels
    for b, data in enumerate(testset):
        if limit is not False and b >= limit:
            break

        targets = None
        if isinstance(data, tuple) or isinstance(data, list):
            inputs = data[0]
            targets = data[1]
        else:
            inputs = data

        assert isinstance(inputs, torch.Tensor)

        inputs = common.torch.as_variable(inputs, cuda)
        targets = common.torch.as_variable(targets, cuda)
        inputs = inputs.permute(0, 3, 1, 2)

        logits, features_ = model(inputs, return_features=True, operators=operators)

        probabilities_ = common.torch.softmax(logits, dim=1).detach().cpu().numpy()
        probabilities = common.numpy.concatenate(probabilities, probabilities_)

        if features is None:
            features = []
            for i in range(len(features_)):
                features.append(features_[i].reshape(logits.size(0), -1).detach().cpu().numpy())
        else:
            assert len(features) == len(features_)
            for i in range(len(features)):
                features[i] = common.numpy.concatenate(features[i], features_[i].reshape(logits.size(0), - 1).detach().cpu().numpy())

        if targets is not None and loss:
            targets = common.torch.as_variable(targets, cuda)
            error = common.torch.classification_error(logits, targets)
            loss = common.torch.classification_loss(logits, targets)
            progress('test', b, len(testset), info='error=%g loss=%g' % (error.item(), loss.item()))
        else:
            progress('test', b, len(testset))

    return probabilities, features


def attack(model, testset, attack, objective, attempts=1, writer=common.summary.SummaryWriter(), operators=None, cuda=False):
    """
    Attack model.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param attack: attack
    :type attack: attacks.Attack
    :param objective: attack objective
    :type objective: attacks.Objective
    :param attempts: number of attempts
    :type attempts: int
    :param get_writer: summary writer or utility function to get writer
    :type get_writer: torch.utils.tensorboard.SummaryWriter or callable
    :param cuda: whether to use CUDA
    :type cuda: bool
    """

    assert model.training is False
    assert len(testset) > 0
    assert attempts >= 0
    assert isinstance(testset, torch.utils.data.DataLoader)
    assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    perturbations = []
    probabilities = []
    errors = []

    # should work via subsets of datasets
    for a in range(attempts):
        perturbations_a = None
        probabilities_a = None
        errors_a = None

        for b, data in enumerate(testset):
            assert isinstance(data, tuple) or isinstance(data, list)

            inputs = common.torch.as_variable(data[0], cuda)
            inputs = inputs.permute(0, 3, 1, 2)
            labels = common.torch.as_variable(data[1], cuda)

            # attack target labels
            targets = None
            if len(list(data)) > 2:
                targets = common.torch.as_variable(data[2], cuda)

            objective.set(labels, targets)
            attack.progress = ProgressBar()
            perturbations_b, errors_b = attack.run(model, inputs, objective,
                                                   writer=writer if not callable(writer) else writer('%d-%d' % (a, b)),
                                                   prefix='%d/%d/' % (a, b) if not callable(writer) else '')

            inputs = inputs + common.torch.as_variable(perturbations_b, cuda)
            logits = model(inputs, operators=operators)
            probabilities_b = common.torch.softmax(logits, dim=1).detach().cpu().numpy()

            perturbations_a = common.numpy.concatenate(perturbations_a, perturbations_b)
            probabilities_a = common.numpy.concatenate(probabilities_a, probabilities_b)
            errors_a = common.numpy.concatenate(errors_a, errors_b)

        perturbations.append(perturbations_a)
        probabilities.append(probabilities_a)
        errors.append(errors_a)

    perturbations = numpy.array(perturbations)
    probabilities = numpy.array(probabilities)
    errors = numpy.array(errors)

    assert perturbations.shape[1] == len(testset.dataset)
    assert probabilities.shape[1] == len(testset.dataset)
    assert errors.shape[1] == len(testset.dataset)

    return perturbations, probabilities, errors
    # attempts x N x C x H x W, attempts x N x K, attempts x N


def attack_weights(model, testset, attack, objective, attempts=1, start_attempt=0, writer=common.summary.SummaryWriter(), eval=True, cuda=False):
    """
    Attack model.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param attack: attack
    :type attack: attacks.Attack
    :param objective: attack objective
    :type objective: attacks.Objective
    :param attempts: number of attempts
    :type attempts: int
    :param writer: summary writer
    :type writer: torch.utils.tensorboard.SummaryWriter
    :param cuda: whether to use CUDA
    :type cuda: bool
    """

    assert model.training is not eval
    assert len(testset) > 0
    assert attempts > 0
    assert isinstance(testset, torch.utils.data.DataLoader)
    assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    if start_attempt > 0:
        initialization = getattr(attack, 'initialization', None)
        if initialization is not None:
            randomness = getattr(initialization, 'randomness', None)
            if randomness is not None:
                for i in range(start_attempt):
                    next(randomness)

    # should work via subsets of datasets
    perturbed_models = []
    for a in range(start_attempt, attempts):
        attack.progress = ProgressBar()
        attack.writer = writer
        attack.prefix = '%d/' % a if not callable(writer) else ''
        objective.reset()
        perturbed_model = attack.run(model, testset, objective)
        if attack.writer is not None:
            attack.writer.flush()
        assert common.torch.is_cuda(perturbed_model) is False
        perturbed_models.append(perturbed_model)

    return perturbed_models


def attack_weights_inputs(model, testset, attack, weight_objective, input_objective, attempts=1, writer=common.summary.SummaryWriter(), cuda=False):
    """
    Attack model.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param attack: attack
    :type attack: attacks.Attack
    :param objective: attack objective
    :type objective: attacks.Objective
    :param attempts: number of attempts
    :type attempts: int
    :param writer: summary writer
    :type writer: torch.utils.tensorboard.SummaryWriter
    :param cuda: whether to use CUDA
    :type cuda: bool
    """

    assert model.training is not True
    assert len(testset) > 0
    assert attempts > 0
    assert isinstance(testset, torch.utils.data.DataLoader)
    assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    # should work via subsets of datasets
    perturbed_models = []
    perturbations = []
    probabilities = []
    errors = []

    for a in range(attempts):
        attack.progress = ProgressBar()
        attack.writer = writer
        attack.prefix = '%d/' % a if not callable(writer) else ''
        weight_objective.reset()
        perturbed_model, perturbations_a, errors_a = attack.run(model, testset, weight_objective, input_objective)
        if attack.writer is not None:
            attack.writer.flush()
        assert common.torch.is_cuda(perturbed_model) is False

        if cuda:
            perturbed_model = perturbed_model.cuda()

        dataset = common.datasets.AdversarialDataset(testset.dataset.images, numpy.expand_dims(perturbations_a, axis=0), testset.dataset.labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=testset.batch_size)

        probabilities_a = None
        for b, (inputs, targets) in enumerate(dataloader):
            inputs = common.torch.as_variable(inputs, cuda)
            inputs = inputs.permute(0, 3, 1, 2)

            logits = perturbed_model(inputs)
            probabilities_b = torch.nn.functional.softmax(logits, dim=1)
            probabilities_a = common.numpy.concatenate(probabilities_a, probabilities_b.detach().cpu().numpy())

        perturbed_models.append(perturbed_model.cpu())
        perturbations.append(numpy.expand_dims(perturbations_a, axis=0))
        probabilities.append(numpy.expand_dims(probabilities_a, axis=0))
        errors.append(numpy.expand_dims(errors_a, axis=0))

    perturbations = numpy.concatenate(tuple(perturbations), axis=0)
    probabilities = numpy.concatenate(tuple(probabilities), axis=0)
    errors = numpy.concatenate(tuple(errors), axis=0)
    return perturbed_models, perturbations, probabilities, errors