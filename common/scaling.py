import torch
import numpy
import common.torch
from common.log import log


def scaling_factor(model_a, model_b, norm='linf', names=['weight', 'bias'], ignore=[]):
    """
    Compute scaling factors between two models. Scaling is model_a/model_b.

    :param model_a: first model
    :type model_a: torch.nn.Module
    :param model_b: second model
    :type model_b: torch.nn.Module
    :param name: parameter name
    :type name: str
    :return: factors, global factor
    :rtype factors: [float], float
    """

    assert len(names) > 0

    parameters_a = list(model_a.named_parameters())
    parameters_b = list(model_b.named_parameters())

    L = len(parameters_a)
    assert L == len(parameters_b)

    all_parameters_a = None
    all_parameters_b = None
    for l in range(L):
        parameter_name = parameters_a[l][0]
        assert parameter_name == parameters_b[l][0]

        include = False
        for name in names:
            if parameter_name.find(name) >= 0:
                include = True
        for name in ignore:
            if parameter_name.find(name) >= 0:
                include = False

        if include and parameters_a[l][1].requires_grad is True and parameters_b[l][1].requires_grad is True:
            all_parameters_a = common.numpy.concatenate(all_parameters_a, parameters_a[l][1].data.view(-1).cpu().numpy())
            all_parameters_b = common.numpy.concatenate(all_parameters_b, parameters_b[l][1].data.view(-1).cpu().numpy())

    assert all_parameters_a is not None
    assert all_parameters_b is not None

    if norm == 'linf':
        norm_a = numpy.max(numpy.abs(all_parameters_a))
        norm_b = numpy.max(numpy.abs(all_parameters_b))
    elif norm == 'l2':
        norm_a = numpy.linalg.norm(all_parameters_a, 2)
        norm_b = numpy.linalg.norm(all_parameters_b, 2)
    elif norm == 'l1':
        norm_a = numpy.linalg.norm(all_parameters_a, 1)
        norm_b = numpy.linalg.norm(all_parameters_b, 1)
    else:
        raise NotImplementedError

    return norm_a/norm_b


def scale(model, factor, names=['weight', 'bias'], ignore=[], bn=True):
    """
    Scale model.

    :param model: model
    :type model: torch.nn.Module
    :param factor: factor
    :type factor: float
    :param names: names
    :type names: [str]
    """

    assert isinstance(names, list) and not isinstance(names, str)
    assert isinstance(ignore, list) and not isinstance(ignore, str)

    for parameter_name, parameter in model.named_parameters():
        include = False
        for name in names:
            if parameter_name.find(name) >= 0:
                include = True
        for name in ignore:
            if parameter_name.find(name) >= 0:
                include = False

        if include and parameter.requires_grad:
            assert parameter_name.find('rebn') < 0
            parameter.data *= factor
            log('scaled %s' % parameter_name)
        else:
            log('ignored %s' % parameter_name)

    if bn:
        def flatten(model):
            flattened = [flatten(children) for children in model.children()]
            res = [model]
            for module in flattened:
                res += module
            return res

        modules = flatten(model)
        for module in modules:
            log(module.__class__.__name__)
            if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, common.torch.ReparameterizedBatchNorm1d) or isinstance(module, common.torch.ReparameterizedBatchNorm2d):
                if 'weight' in names:
                    module.running_var.data *= factor*factor
                    log('scaled var %s' % module.__class__.__name__)
                if 'bias' in names:
                    module.running_mean.data *= factor
                    log('scaled mean %s' % module.__class__.__name__)


def scaling_factors_per_layer(model_a, model_b, names=['weight', 'bias']):
    """
    Compute scaling factors between two models. Scaling is model_a/model_b.

    :param model_a: first model
    :type model_a: torch.nn.Module
    :param model_b: second model
    :type model_b: torch.nn.Module
    :param names: parameter name
    :type names: [str]
    :return: factors, global factor
    :rtype factors: [float], float
    """

    raise NotImplementedError


def scale_per_layer(model, factors, names=['weight', 'bias'], ignore=[], bn=True):
    """
    Scale model.

    :param model: model
    :type model: torch.nn.Module
    :param factors: factors
    :type factors: [float]
    :param names: names
    :type names: [str]
    """

    raise NotImplementedError


def statistics(model):
    """
    Basic model statistics to assess scaling.

    :param model: model
    :type model: torch.nn.Module
    :return: min, mean, max
    :rtype:
    """

    max_parameter = -1e12
    min_parameter = 1e12
    mean_parameter = 0
    parameter_statistics = []
    n = 0

    for parameter in model.parameters():
        if parameter.requires_grad:
            min_parameter_ = torch.min(parameter).item()
            max_parameter_ = torch.max(parameter).item()
            mean_parameter_ = torch.mean(parameter).item()
            min_parameter = min(min_parameter, min_parameter_)
            max_parameter = max(max_parameter, mean_parameter_)
            mean_parameter += mean_parameter_
            parameter_statistics.append([min_parameter_, mean_parameter_, max_parameter_])
            n += 1

    mean_parameter /= n
    parameter_statistics.insert(0, [min_parameter, mean_parameter, max_parameter])

    return parameter_statistics


def norms(model):
    all_parameters = common.torch.all_parameters(model)
    return torch.norm(all_parameters, 2).item(), torch.norm(all_parameters, 1).item(), torch.max(torch.abs(all_parameters)).item()