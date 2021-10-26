import numpy
import torch
import common.torch
import common.test
from common.progress import ProgressBar
import math


# https://github.com/tomgoldstein/loss-landscape/blob/64ef4d57f8dabe79b57a637819c44e48eda98f33/net_plotter.py
def _normalize_direction(direction, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.
        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())


def weight_attack(model, attack, objective, attackloader):
    """
    Determine a weight direction given a weight attack.

    :param model: model to attack
    :type model: torch.nn.Module
    :param attack: attack to use
    :type attack: attacks.weights.Attack
    :param objective: objective for attack
    :type objective: attacks.weights.objectives.Objective
    :param attackloader: data loader for attack
    :type attackloader: torch.data.utils.DataLoader
    :return: direction as torch.nn.Module
    :rtype: torch.nn.Module
    """

    assert model.training is False

    attack.progress = ProgressBar()
    perturbed_model = attack.run(model, attackloader, objective)

    return perturbed_model


def weight_direction(model, perturbed_model, attack, attackloader, normalization, cuda=False):
    """
    Determine a weight direction given a weight attack.

    :param model: model to attack
    :type model: torch.nn.Module
    :param perturbed_model: attacked model
    :type perturbed_model: torch.nn.Module
    :param attack: attack to use
    :type attack: attacks.weights.Attack
    :param attackloader: data loader for attack
    :type attackloader: torch.data.utils.DataLoader
    :param normalization: normalization type
    :type normalization: str
    :param cuda: CUDA use
    :type cuda: bool
    :return: direction as torch.nn.Module
    :rtype: torch.nn.Module
    """

    direction_model = common.torch.clone(perturbed_model)

    if common.torch.is_cuda(model):
        model = model.cuda()
        direction_model = direction_model.cuda()

    parameters = list(model.parameters())
    direction_parameters = list(direction_model.parameters())

    for i in range(len(parameters)):
        direction_parameters[i].data -= parameters[i].data

    def norms(parameters):
        all_parameters = None
        for parameter in parameters:
            all_parameters = common.torch.concatenate(all_parameters, parameter.view(-1))
        linf_norm = torch.max(torch.abs(all_parameters)).item()
        l2_norm = torch.norm(all_parameters).item()
        return linf_norm, l2_norm

    linf_norm, l2_norm = norms(parameters)
    direction_linf_norm, direction_l2_norm = norms(direction_parameters)

    for i in range(len(parameters)):

        # in case attack was not run, layers are not set
        if i not in attack.layers_(model):
            if torch.max(torch.abs(direction_parameters[i].data)) > 0.00001:
                print('layer %d has been changed even though it wa snot supposed to, change: %g' % (i, torch.max(torch.abs(direction_parameters[i].data))))
            direction_parameters[i].data.fill_(0)
            continue

        # epsilons are relative per layer, e.g., eps*norm(parameters_i)
        # but perturbations do not need to "use" the whole epsilon-ball
        # so we should normalize per layer by norm(parameters_i)
        # then norm(perturbed_parameters_i - parameters_i) == eps*norm(parameters_i)
        # then, step == 1 means one eps step in each layer
        # that does not tell us anything about the absolute norm/distance taken
        # for L_1: eps*norm_1(parameters_1) + ... + eps*norm_1(parameters_n)
        # for L_inf: max(eps*norm_inf(parameters1), ... , eps*norm_inf(parameters_n))
        # for L_2: sqrt(eps^2*norm_2(parameters_1)^2, ..., eps^2*norm_2(parameters_n)^2)
        # note that relative per layer normalization does change the overall model direction!

        if normalization.startswith('linf'):
            pass
        elif normalization.startswith('l2'):
            pass
        elif normalization.startswith('filter_linf'):
            for d, w in zip(direction_parameters[i].data, parameters[i].data):
                d.mul_(w.norm() / (d.norm() + 1e-10))
        elif normalization.startswith('filter_l2_05'):
            for d, w in zip(direction_parameters[i].data, parameters[i].data):
                d.mul_(0.5*torch.abs(torch.max(w)) / (torch.abs(torch.max(d)) + 1e-10))
        elif normalization.startswith('filter_l2_005'):
            for d, w in zip(direction_parameters[i].data, parameters[i].data):
                d.mul_(0.05*torch.abs(torch.max(w)) / (torch.abs(torch.max(d)) + 1e-10))
        elif normalization.startswith('filter_l2_0025'):
            for d, w in zip(direction_parameters[i].data, parameters[i].data):
                d.mul_(0.025*torch.abs(torch.max(w)) / (torch.abs(torch.max(d)) + 1e-10))
        elif normalization.startswith('filter_l2'):
            for d, w in zip(direction_parameters[i].data, parameters[i].data):
                d.mul_(torch.abs(torch.max(w)) / (torch.abs(torch.max(d)) + 1e-10))
        elif normalization.startswith('layer_linf'):
            direction_parameters[i].data.mul_(torch.max(torch.abs(parameters[i].data)) / (torch.max(torch.abs(direction_parameters[i].data)) + 1e-10))
        elif normalization.startswith('layer_l2_05'):
            direction_parameters[i].data.mul_(0.5*torch.max(torch.abs(parameters[i].data)) / (torch.max(torch.abs(direction_parameters[i].data)) + 1e-10))
        elif normalization.startswith('layer_l2_005'):
            direction_parameters[i].data.mul_(0.05*torch.max(torch.abs(parameters[i].data)) / (torch.max(torch.abs(direction_parameters[i].data)) + 1e-10))
        elif normalization.startswith('layer_l2_0025'):
            direction_parameters[i].data.mul_(0.025*torch.max(torch.abs(parameters[i].data)) / (torch.max(torch.abs(direction_parameters[i].data)) + 1e-10))
        elif normalization.startswith('layer_l2_001'):
            direction_parameters[i].data.mul_(0.01*torch.max(torch.abs(parameters[i].data)) / (torch.max(torch.abs(direction_parameters[i].data)) + 1e-10))
        elif normalization.startswith('layer_l2'):
            direction_parameters[i].data.mul_(torch.max(torch.abs(parameters[i].data)) / (torch.max(torch.abs(direction_parameters[i].data)) + 1e-10))
        elif normalization.startswith('relative_linf'):
            direction_parameters[i].data.mul_(linf_norm/direction_linf_norm)
        elif normalization.startswith('relative_l2_05'):
            direction_parameters[i].data.mul_(0.5 * l2_norm / direction_l2_norm)
        elif normalization.startswith('relative_l2_005'):
            direction_parameters[i].data.mul_(0.05*l2_norm/direction_l2_norm)
        elif normalization.startswith('relative_l2_0025'):
            direction_parameters[i].data.mul_(0.05*l2_norm/direction_l2_norm)
        elif normalization.startswith('relative_l2'):
            direction_parameters[i].data.mul_(l2_norm/direction_l2_norm)
        elif normalization.startswith('direction_linf'):
            direction_parameters[i].data.div_(direction_linf_norm)
        elif normalization.startswith('direction_l2'):
            direction_parameters[i].data.div_(direction_l2_norm)
        else:
            raise NotImplementedError

    # instead of analytically judgind the step size taken in the direction, just compute it
    direction_linf_norm, direction_l2_norm = norms(direction_parameters)

    # none for filter normalization
    if normalization.find('linf') >= 0:
        scale_factor = direction_linf_norm
    elif normalization.find('l2') >= 0:
        scale_factor = direction_l2_norm
    else:
        raise NotImplementedError

    if normalization.endswith('_rte'):
        def evaluate_model(model_to_evaluate):
            total_error = 0.
            for b, (inputs, targets) in enumerate(attackloader):
                inputs = common.torch.as_variable(inputs, cuda)
                targets = common.torch.as_variable(targets, cuda)
                inputs = inputs.permute(0, 3, 1, 2)

                logits = model_to_evaluate(inputs)
                error = common.torch.classification_error(logits, targets)

                total_error += error.item()
            total_error /= len(attackloader)
            return total_error

        reference_error = evaluate_model(model)
        target_error = min(1, reference_error + 0.1)
        min_step = 1./50.
        step = None
        step_model = common.torch.clone(model)

        for i in range(1, 101):
            parameters = list(model.parameters())
            step_parameters = list(step_model.parameters())
            direction_parameters = list(direction_model.parameters())
            for k in range(len(step_parameters)):
                step_parameters[k].data = parameters[k].data + i*min_step*direction_parameters[k].data

            # define outside of if statement below to catch the case where error will not be above target_error
            step = i*min_step

            step_error = evaluate_model(step_model)
            if step_error >= target_error:
                break

        assert step is not None
        scale_factor *= step
        direction_parameters = list(direction_model.parameters())
        for i in range(len(direction_parameters)):
            direction_parameters[i].data *= 1/step

    return direction_model, scale_factor


def input_attack(model, attack, objective, testloader, cuda=False):
    """
    Run attack, separate from input_direction due to reproducibility.

    :param model: model
    :type model: torch.nn.Module
    :param attack: attack to use
    :type attack: attacks.Attack
    :param objective: objective for attack
    :type objective: attacks.objectives.Objective
    :param testloader: test loader
    :type testloader: torch.data.utils.DataLoader
    """

    perturbations = None
    for b, data in enumerate(testloader):
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
        perturbations_, _ = attack.run(model, inputs, objective)

        perturbations = common.numpy.concatenate(perturbations, perturbations_)

    return perturbations


def input_direction(perturbations, normalization, cuda=False):
    """
    Get input directions from ana ttack.

    :param perturbations: perturbations
    :type perturbations: numpy.array
    :param normalization: normalization
    :type normalization: str
    :param cuda: CUDA use
    :type cuda: bool
    :return: directions, inputs and targets as numpy.ndarray
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
    """

    inputs_as_numpy = None
    labels_as_numpy = None

    for b, (inputs, labels) in enumerate(perturbations):
        inputs_as_numpy = common.numpy.concatenate(inputs_as_numpy, inputs.cpu().detach().numpy())
        labels_as_numpy = common.numpy.concatenate(labels_as_numpy, labels.cpu().detach().numpy())

    if normalization == 'linf':
        pass
    elif normalization == 'l2':
        pass
    elif normalization == 'direction_linf':
        norms = numpy.max(numpy.abs(perturbations.reshape(perturbations.shape[0], -1)), axis=1)
        perturbations /= norms.reshape(norms.shape[0], 1, 1, 1)
    elif normalization == 'direction_l2':
        norms = numpy.norm(perturbations.reshape(perturbations.shape[0], -1), axis=1)
        perturbations /= norms.reshape(norms.shape[0], 1, 1, 1)
    else:
        raise NotImplementedError

    if normalization.find('linf') >= 0:
        scale_factors_ = numpy.max(numpy.abs(perturbations.reshape(perturbations.shape[0], -1)), axis=1)
    elif normalization.find('l2') >= 0:
        scale_factors_ = numpy.norm(perturbations.reshape(perturbations.shape[0], -1), axis=1)
    else:
        raise NotImplementedError

    directions = perturbations
    scale_factors = scale_factors_

    return directions, inputs_as_numpy, labels_as_numpy, scale_factors


# input normalization does not make sense as loss is calculated on all examples!
def weight_input_attack(model, attack, weight_objective, input_objective, testloader, weight_normalization, input_normalization, epsilon=None, cuda=False):
    """
    Get an input and a weight direction from a joint attack.

    :param model: model
    :type model: torch.nn.Module
    :param attack: attack
    :type attack: attacks.weights_inputs.Attack
    :param weight_objective: objective for weights
    :type weight_objective: attacks.weights.objectives.Objective
    :param input_objective: objective for inputs
    :type input_objective: attacks.objectives.Objective
    :param testloader: test loader
    :type testloader: torch.data.utils.DataLoader
    :param weight_normalization: weight normalization
    :type weight_normalization: str
    :param input_normalization: input normalization
    :type input_normalization: str
    :param epsilon: epsilon
    :type : float
    :param cuda: CUDA usepsilone
    :type cuda: bool
    """

    assert model.training is False

    attack.progress = ProgressBar()
    perturbed_model, perturbations, _ = attack.run(model, testloader, weight_objective, input_objective)

    return perturbed_model, perturbations


# input normalization does not make sense as loss is calculated on all examples!
def weight_input_direction(model, perturbed_model, perturbations, testloader, weight_normalization, input_normalization, epsilon=None, cuda=False):
    """
    Get an input and a weight direction from a joint attack.

    :param model: model
    :type model: torch.nn.Module
    :param testloader: test loader
    :type testloader: torch.data.utils.DataLoader
    :param weight_normalization: weight normalization
    :type weight_normalization: str
    :param input_normalization: input normalization
    :type input_normalization: str
    :param epsilon: epsilon
    :type : float
    :param cuda: CUDA usepsilone
    :type cuda: bool
    """

    if common.torch.is_cuda(model):
        perturbed_model = perturbed_model.cuda()
    direction_model = common.torch.clone(perturbed_model)

    parameters = list(model.parameters())
    perturbed_parameters = list(perturbed_model.parameters())
    direction_parameters = list(direction_model.parameters())

    for i in range(len(parameters)):
        direction_parameters[i].data -= parameters[i].data

    def norms(parameters):
        all_parameters = None
        for parameter in parameters:
            all_parameters = common.torch.concatenate(all_parameters, parameter.view(-1))
        linf_norm = torch.max(torch.abs(all_parameters)).item()
        l2_norm = torch.norm(all_parameters).item()
        return linf_norm, l2_norm

    linf_norm, l2_norm = norms(parameters)
    direction_linf_norm, direction_l2_norm = norms(direction_parameters)

    for i in range(len(parameters)):
        if weight_normalization == 'linf':
            pass
        elif weight_normalization == 'l2':
            pass
        elif weight_normalization == 'filter_linf':
            for d, w in zip(direction_parameters[i].data, parameters[i].data):
                d.mul_(w.norm() / (d.norm() + 1e-10))
        elif weight_normalization == 'filter_l2':
            for d, w in zip(direction_parameters[i].data, parameters[i].data):
                d.mul_(torch.abs(torch.max(w)) / (torch.abs(torch.max(d)) + 1e-10))
        elif weight_normalization == 'layer_linf':
            direction_parameters[i].data.mul_(
                torch.max(torch.abs(parameters[i].data)) / (torch.max(torch.abs(direction_parameters[i].data)) + 1e-10))
        elif weight_normalization == 'layer_l2':
            direction_parameters[i].data.mul_(
                torch.max(torch.abs(parameters[i].data)) / (torch.max(torch.abs(direction_parameters[i].data)) + 1e-10))
        elif weight_normalization == 'relative_linf':
            direction_parameters[i].data.mul_(linf_norm / direction_linf_norm)
        elif weight_normalization == 'relative_l2':
            direction_parameters[i].data.mul_(l2_norm / direction_l2_norm)
        elif weight_normalization == 'direction_linf':
            direction_parameters[i].data.div_(direction_linf_norm)
        elif weight_normalization == 'direction_l2':
            direction_parameters[i].data.div_(direction_l2_norm)
        else:
            raise NotImplementedError

    # instead of analytically judgind the step size taken in the direction, just compute it
    direction_linf_norm, direction_l2_norm = norms(direction_parameters)

    # none for filter normalization
    if weight_normalization.find('linf') >= 0:
        weight_scale_factor = direction_linf_norm
    elif weight_normalization.find('l2') >= 0:
        weight_scale_factor = direction_l2_norm
    else:
        raise NotImplementedError

    input_scale_factor = 1
    if input_normalization == '':
        pass
    else:
        assert NotImplementedError

    inputs_as_numpy = None
    labels_as_numpy = None

    for b, data in enumerate(testloader):
        assert isinstance(data, tuple) or isinstance(data, list)
        inputs = common.torch.as_variable(data[0], cuda)
        inputs = inputs.permute(0, 3, 1, 2)
        labels = common.torch.as_variable(data[1], cuda)

        inputs_as_numpy = common.numpy.concatenate(inputs_as_numpy, inputs.cpu().detach().numpy())
        labels_as_numpy = common.numpy.concatenate(labels_as_numpy, labels.cpu().detach().numpy())

    return direction_model, weight_scale_factor, perturbations, inputs_as_numpy, labels_as_numpy, input_scale_factor


# weight loss
def weight_1d(model, direction, testloader, loss=common.torch.classification_loss, error=common.torch.classification_error, steps=numpy.linspace(-1, 1, 101), cuda=False):
    """
    Visualize weights along a possibly normalized direction from weight_direction_from_attack.

    :param model: model
    :type model: torch.nn.Module
    :param direction: direction as model
    :type direction: torch.nn.Module
    :param testloader: test loader
    :type testloader: torch.data.utile.DataLoader
    :param loss: loss function
    :type loss: callable
    :param error: error function
    :type error: callable
    :param steps: steps
    :type steps: numpy.ndarray
    :param cuda: CUDA use
    :type cuda: bool
    :return: steps, losses
    :rtype: numpy.ndarray, numpy.ndarray
    """

    assert model.training is False

    losses = []
    errors = []
    progress = ProgressBar()
    direction_parameters = list(direction.parameters())

    for s in range(steps.shape[0]):
        step = steps[s]
        direction_loss = 0
        direction_error = 0
        perturbed_model = common.torch.clone(model)
        perturbed_parameters = list(perturbed_model.parameters())

        if not math.isclose(step, 0):
            for i in range(len(perturbed_parameters)):
                perturbed_parameters[i].data += direction_parameters[i].data*step

        for b, data in enumerate(testloader):
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

            logits = perturbed_model(inputs)
            direction_loss += loss(logits, targets).item()
            direction_error += error(logits, targets).item()
            progress('test', s*len(testloader) + b, len(testloader)*len(steps), info='step=%g loss=%g error=%g' % (step, direction_loss, direction_error))

        direction_loss /= len(testloader)
        direction_error /= len(testloader)
        losses.append(direction_loss)
        errors.append(direction_error)

    return steps, numpy.array(losses), numpy.array(errors)
    # output is steps, numpy.array of losses


def weight_2d(model, xdirection, ydirection, testloader, loss=common.torch.classification_loss, error=common.torch.classification_error, xsteps=numpy.linspace(-1, 1, 51), ysteps=numpy.linspace(-1, 1, 51), cuda=False):
    """
    Weight visualization in 2 directions.

    :param model: model
    :type model: torch.nn.Module
    :param xdirection: x-direction model
    :type xdirection: torch.nn.Module
    :param ydirection: y-direction model
    :type ydirection: torch.nn.Module
    :param testloader: test data
    :type testloader: torch.utils.data.DataLoader
    :param loss: loss function
    :type loss: callable
    :param error: error function
    :type error: callable
    :param xsteps: x-steps
    :type xsteps: numpy.ndarray
    :param ysteps: y-steps
    :type ysteps: numpy.ndarray
    :param cuda: use CUDA
    :type cuda: bool
    :return: x mesh, y mesh and losses
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
    """

    xsteps_mesh, ysteps_mesh = numpy.meshgrid(xsteps, ysteps)
    xsteps_mesh = xsteps_mesh.T
    ysteps_mesh = ysteps_mesh.T

    rows = xsteps_mesh.shape[0]
    cols = ysteps_mesh.shape[1]

    losses = numpy.zeros((rows, cols))
    errors = numpy.zeros((rows, cols))
    xdirection_parameters = list(xdirection.parameters())
    ydirection_parameters = list(ydirection.parameters())
    progress = ProgressBar()

    for j in range(cols):
        for i in range(rows):

            xstep = xsteps_mesh[i, j]
            ystep = ysteps_mesh[i, j]

            current_loss = 0
            current_error = 0
            perturbed_model = common.torch.clone(model)
            perturbed_parameters = list(perturbed_model.parameters())

            for k in range(len(perturbed_parameters)):
                perturbed_parameters[k].data += xdirection_parameters[k].data * xstep + ydirection_parameters[k].data * ystep

            for b, data in enumerate(testloader):
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

                logits = perturbed_model(inputs)
                current_loss += loss(logits, targets).item()
                current_error += error(logits, targets).item()
                progress('test', j*cols + i, rows * cols, info='xstep=%g ystep=%g loss=%g error=%g' % (xstep, ystep, current_loss, current_error))

            current_loss /= len(testloader)
            current_error /= len(testloader)
            losses[i, j] = current_loss
            errors[i, j] = current_error

    return xsteps_mesh, ysteps_mesh, losses, errors


def hessian_1d(model, direction, testloader, hessian_k=10, steps=numpy.linspace(-1, 1, 101), cuda=False):
    """
    Visualize weights along a possibly normalized direction from weight_direction_from_attack.

    :param model: model
    :type model: torch.nn.Module
    :param direction: direction as model
    :type direction: torch.nn.Module
    :param testloader: test loader
    :type testloader: torch.data.utile.DataLoader
    :param steps: steps
    :type steps: numpy.ndarray
    :param cuda: CUDA use
    :type cuda: bool
    :return: steps, losses
    :rtype: numpy.ndarray, numpy.ndarray
    """

    assert model.training is False

    eigs = numpy.zeros((steps.shape[0], hessian_k))
    progress = ProgressBar()
    direction_parameters = list(direction.parameters())

    for s in range(steps.shape[0]):
        step = steps[s]
        perturbed_model = common.torch.clone(model)
        perturbed_parameters = list(perturbed_model.parameters())

        for i in range(len(perturbed_parameters)):
            perturbed_parameters[i].data += direction_parameters[i].data*step

        criterion = torch.nn.CrossEntropyLoss()
        eigs_, _ = common.hessian.min_max_k_hessian_eigs(perturbed_model, testloader, criterion, k=hessian_k, use_cuda=cuda)

        eigs[s, :] = numpy.array(eigs)
        progress('test', s, len(steps), info='step=%g maxeig=%g mineig=%g' % (step, numpy.max(eigs_), numpy.min(eigs_)))

    return steps, eigs


def hessian_2d(model, xdirection, ydirection, testloader, hessian_k=10, xsteps=numpy.linspace(-1, 1, 51), ysteps=numpy.linspace(-1, 1, 51), cuda=False):
    """
    Weight visualization in 2 directions.

    :param model: model
    :type model: torch.nn.Module
    :param xdirection: x-direction model
    :type xdirection: torch.nn.Module
    :param ydirection: y-direction model
    :type ydirection: torch.nn.Module
    :param testloader: test data
    :type testloader: torch.utils.data.DataLoader
    :param xsteps: x-steps
    :type xsteps: numpy.ndarray
    :param ysteps: y-steps
    :type ysteps: numpy.ndarray
    :param cuda: use CUDA
    :type cuda: bool
    :return: x mesh, y mesh and losses
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
    """

    xsteps_mesh, ysteps_mesh = numpy.meshgrid(xsteps, ysteps)
    xsteps_mesh = xsteps_mesh.T
    ysteps_mesh = ysteps_mesh.T

    rows = xsteps_mesh.shape[0]
    cols = ysteps_mesh.shape[1]

    eigs = numpy.zeros((rows, cols, hessian_k))
    xdirection_parameters = list(xdirection.parameters())
    ydirection_parameters = list(ydirection.parameters())
    progress = ProgressBar()

    for j in range(cols):
        for i in range(rows):

            xstep = xsteps_mesh[i, j]
            ystep = ysteps_mesh[i, j]

            perturbed_model = common.torch.clone(model)
            perturbed_parameters = list(perturbed_model.parameters())

            for k in range(len(perturbed_parameters)):
                perturbed_parameters[k].data += xdirection_parameters[k].data * xstep + ydirection_parameters[k].data * ystep

            criterion = torch.nn.CrossEntropyLoss()
            eigs_, _ = common.hessian.min_max_k_hessian_eigs(perturbed_model, testloader, criterion, k=hessian_k, use_cuda=cuda)

            eigs[i, j] = eigs_
            progress('test', j*cols + i, rows * cols, info='xstep=%g ystep=%g maxeig=%g mineig=%g' % (xstep, ystep, numpy.max(eigs_), numpy.min(eigs_)))

    return xsteps_mesh, ysteps_mesh, eigs


# input loss for each example in testloader (e.g., one batch)
def input_1d(model, directions, inputs, targets, loss=common.torch.classification_loss, error=common.torch.classification_error, steps=numpy.linspace(-1, 1, 101), cuda=False):
    """
    Visualize input loss surface along given directions.

    :param model: model
    :type model: torch.nn.Module
    :param directions: directions (i.e., perturbations)
    :type directions: numpy.ndarray
    :param inputs: inputs
    :type directions: numpy.ndarray
    :param targets: targets
    :type targets: numpy.ndarray
    :param loss: loss function
    :type loss: callable
    :param error: error function
    :type error: callable
    :param steps: steps
    :type steps: numpy.ndarray
    :param cuda: CUDA use
    :type cuda: bool
    :return: steps, losses and probabilities
    :rtype: numpy.ndarray, numpy.ndarray and numpy.ndarray
    """

    assert model.training is False
    assert isinstance(inputs, numpy.ndarray)
    assert isinstance(directions, numpy.ndarray)

    losses = None
    errors = None
    probabilities = None
    progress = ProgressBar()

    for s in range(len(steps)):
        step = steps[s]
        perturbed_inputs = numpy.clip(inputs + step*directions, 0, 1)
        perturbed_inputs = common.torch.as_variable(perturbed_inputs, cuda)
        targets = common.torch.as_variable(targets, cuda)

        logits = model(perturbed_inputs)
        probabilities_ = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()
        losses_ = loss(logits, targets, reduction='none').cpu().detach().numpy()
        errors_ = error(logits, targets, reduction='none').cpu().detach().numpy()

        probabilities = common.numpy.concatenate(probabilities, probabilities_.reshape(1, losses_.shape[0], -1), axis=0)
        losses = common.numpy.concatenate(losses, losses_.reshape(1, -1), axis=0)
        errors = common.numpy.concatenate(errors, errors_.reshape(1, -1), axis=0)

        progress('test', s, len(steps), info='step=%g loss=%g error=%g' % (step, numpy.mean(losses), numpy.mean(errors)))

    # output is steps, numpy.array of losses, numpy.array of probabilities
    return steps, losses, probabilities, errors


def input_2d(model, xdirections, ydirections, inputs, targets, loss=common.torch.classification_loss, error=common.torch.classification_error, xsteps=numpy.linspace(-1, 1, 51), ysteps=numpy.linspace(-1, 1, 51), cuda=False):
    """
    Visualize input loss surface along given directions.

    :param model: model
    :type model: torch.nn.Module
    :param xdirections: x-directions (i.e., perturbations)
    :type xdirections: numpy.ndarray
    :param ydirections: y-directions (i.e., perturbations)
    :type ydirections: numpy.ndarray
    :param inputs: inputs
    :type inputs: numpy.ndarray
    :param targets: targets
    :type targets: numpy.ndarray
    :param loss: loss function
    :type loss: callable
    :param error: error function
    :type error: callable
    :param xsteps: x-steps
    :type xsteps: numpy.ndarray
    :param ysteps: y-steps
    :type ysteps: numpy.ndarray
    :param cuda: CUDA use
    :type cuda: bool
    :return: x-steps, y-steps, losses and probabilities
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray and numpy.ndarray
    """

    assert model.training is False
    assert isinstance(inputs, numpy.ndarray)
    assert isinstance(xdirections, numpy.ndarray)
    assert isinstance(ydirections, numpy.ndarray)

    xsteps_mesh, ysteps_mesh = numpy.meshgrid(xsteps, ysteps)
    xsteps_mesh = xsteps_mesh.T
    ysteps_mesh = ysteps_mesh.T

    rows = xsteps_mesh.shape[0]
    cols = ysteps_mesh.shape[1]

    losses = None
    errors = None
    probabilities = None
    progress = ProgressBar()

    for j in range(cols):
        for i in range(rows):
            xstep = xsteps_mesh[i, j]
            ystep = ysteps_mesh[i, j]

            perturbed_inputs = numpy.clip(inputs + xstep*xdirections + ystep*ydirections, 0, 1)
            perturbed_inputs = common.torch.as_variable(perturbed_inputs, cuda)
            targets = common.torch.as_variable(targets, cuda)

            logits = model(perturbed_inputs)
            probabilities_ = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()
            losses_ = loss(logits, targets, reduction='none').cpu().detach().numpy()
            errors_ = error(logits, targets, reduction='none').cpu().detach().numpy()

            if losses is None:
                losses = numpy.zeros((rows, cols, losses_.shape[0]))
            losses[i, j, :] = losses_

            if errors is None:
                errors = numpy.zeros((rows, cols, errors_.shape[0]))
            errors[i, j, :] = errors_

            if probabilities is None:
                probabilities = numpy.zeros((rows, cols, probabilities_.shape[0], probabilities_.shape[1]))
            probabilities[i, j, :, :] = probabilities_

            progress('test', j * cols + i, rows * cols, info='xstep=%g ystep=%g loss=%g error=%g' % (xstep, ystep, numpy.mean(losses), numpy.mean(errors)))

    # output is steps, numpy.array of losses, numpy.array of probabilities
    return xsteps_mesh, ysteps_mesh, losses, probabilities, errors


def weight_input_2d(model, model_direction, input_directions, testloader, loss=common.torch.classification_loss, error=common.torch.classification_error, model_steps=numpy.linspace(-1, 1, 51), input_steps=numpy.linspace(-1, 1, 51), cuda=False):
    """
    Losses for weight and input directions.

    :param model: model
    :type model: torch.nn.Module
    :param model_direction: model direction
    :type model_direction: torch.nn.Module
    :param input_directions: input directions
    :type input_directions: numpy.ndarray
    :param testloader: test loader
    :type testloader: torch.utils.data.DataLoader
    :param loss: loss to use
    :type loss: callable
    :param error: error function
    :type error: callable
    :param model_steps: model steps
    :type model_steps: numpy.ndarray
    :param input_steps: input steps
    :type input_steps: numpy.ndarray
    :param cuda: CUDA use
    :type cuda: bool
    :return: model steps, input steps, losses
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
    """

    model_steps_mesh, input_steps_mesh = numpy.meshgrid(model_steps, input_steps)
    model_steps_mesh = model_steps_mesh.T
    input_steps_mesh = input_steps_mesh.T
    rows = model_steps_mesh.shape[0]
    cols = model_steps_mesh.shape[1]

    losses = numpy.zeros((rows, cols))
    errors = numpy.zeros((rows, cols))
    direction_parameters = list(model_direction.parameters())
    input_directions = numpy.transpose(input_directions, (0, 2, 3, 1))
    progress = ProgressBar()

    for j in range(cols):
        for i in range(rows):

            model_step = model_steps_mesh[i, j]
            input_step = input_steps_mesh[i, j]

            current_loss = 0
            current_error = 0
            perturbed_model = common.torch.clone(model)
            perturbed_parameters = list(perturbed_model.parameters())

            for k in range(len(perturbed_parameters)):
                perturbed_parameters[k].data += direction_parameters[k].data * model_step

            dataset = common.datasets.AdversarialDataset(testloader.dataset.images, input_step*input_directions, testloader.dataset.labels)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=testloader.batch_size)

            for b, data in enumerate(dataloader):
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

                logits = perturbed_model(inputs)
                current_loss += loss(logits, targets).item()
                current_error += error(logits, targets).item()
                progress('test', j*cols + i, rows * cols, info='model_step=%g input_step=%g loss=%g error=%g' % (model_step, input_step, current_loss, current_error))

            current_loss /= len(dataloader)
            current_error /= len(testloader)
            losses[i, j] = current_loss
            errors[i, j] = current_error

    return model_steps_mesh, input_steps_mesh, losses


def adversarial_weight_1d(model, direction, input_attack, input_objective, testloader, loss=common.torch.classification_loss, error=common.torch.classification_error, steps=numpy.linspace(-1, 1, 101), cuda=False):
    """
    Visualize weights along a possibly normalized direction from weight_direction_from_attack.

    :param model: model
    :type model: torch.nn.Module
    :param direction: direction as model
    :type direction: torch.nn.Module
    :param testloader: test loader
    :type testloader: torch.data.utile.DataLoader
    :param loss: loss function
    :type loss: callable
    :param error: error function
    :type error: callable
    :param steps: steps
    :type steps: numpy.ndarray
    :param cuda: CUDA use
    :type cuda: bool
    :return: steps, losses
    :rtype: numpy.ndarray, numpy.ndarray
    """

    assert model.training is False

    losses = []
    errors = []
    progress = ProgressBar()
    direction_parameters = list(direction.parameters())

    for s in range(steps.shape[0]):
        step = steps[s]
        direction_loss = 0
        direction_error = 0
        perturbed_model = common.torch.clone(model)
        perturbed_parameters = list(perturbed_model.parameters())

        for i in range(len(perturbed_parameters)):
            perturbed_parameters[i].data += direction_parameters[i].data*step

        for b, data in enumerate(testloader):
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

            input_objective.set(targets)
            input_attack.progress = ProgressBar()
            adversarial_perturbations, adversarial_objectives = input_attack.run(perturbed_model, inputs, input_objective)
            adversarial_perturbations = common.torch.as_variable(adversarial_perturbations, cuda)
            adversarial_inputs = inputs + adversarial_perturbations

            logits = perturbed_model(adversarial_inputs)
            direction_loss += loss(logits, targets).item()
            direction_error += error(logits, targets).item()
            progress('test', s*len(testloader) + b, len(testloader)*len(steps), info='step=%g loss=%g error=%g' % (step, direction_loss, direction_error))

        direction_loss /= len(testloader)
        direction_error /= len(testloader)
        losses.append(direction_loss)
        errors.append(direction_error)

    return steps, numpy.array(losses), numpy.array(errors)
    # output is steps, numpy.array of losses


def adversarial_hessian_1d(model, direction, input_attack, input_objective, testloader, hessian_k=10, steps=numpy.linspace(-1, 1, 101), cuda=False):
    """
    Visualize weights along a possibly normalized direction from weight_direction_from_attack.

    :param model: model
    :type model: torch.nn.Module
    :param direction: direction as model
    :type direction: torch.nn.Module
    :param testloader: test loader
    :type testloader: torch.data.utile.DataLoader
    :param steps: steps
    :type steps: numpy.ndarray
    :param cuda: CUDA use
    :type cuda: bool
    :return: steps, losses
    :rtype: numpy.ndarray, numpy.ndarray
    """

    assert model.training is False

    eigs = numpy.zeros((steps.shape[0], hessian_k))
    progress = ProgressBar()
    direction_parameters = list(direction.parameters())

    for s in range(steps.shape[0]):
        step = steps[s]
        perturbed_model = common.torch.clone(model)
        perturbed_parameters = list(perturbed_model.parameters())

        for i in range(len(perturbed_parameters)):
            perturbed_parameters[i].data += direction_parameters[i].data*step

        perturbations, _, _ = common.test.attack(model, testloader, input_attack, input_objective, attempts=1, cuda=cuda)
        perturbations = numpy.transpose(perturbations[0], (0, 2, 3, 1))
        adversarialset = common.datasets.AdversarialDataset(testloader.dataset.images, perturbations, testloader.dataset.labels)
        adversarialloader = torch.utils.data.DataLoader(adversarialset, batch_size=testloader.batch_size)

        criterion = torch.nn.CrossEntropyLoss()
        eigs_, _ = common.hessian.min_max_k_hessian_eigs(perturbed_model, adversarialloader, criterion, k=hessian_k, use_cuda=cuda)

        eigs[s, :] = numpy.array(eigs)
        progress('test', s, len(steps), info='step=%g maxeig=%g mineig=%g' % (step, numpy.max(eigs_), numpy.min(eigs_)))

    return steps, eigs