import torch
import numpy
import math
from copy import deepcopy
import functools
import common.numpy as cnumpy
import builtins
from common.log import log
import random


def torch_seed(number, log_seed=True):
    """
    Set torch seed.

    :param number: seed
    :type number: int
    :param log_seed: whether to log
    :type log_seed: bool
    """

    torch.cuda.manual_seed_all(number)
    torch.manual_seed(number)
    if log_seed:
        log('torch seed: %d' % number)


def torch_numpy_seed(number, log_seed=True):
    """
    Set torch seed.

    :param number: seed
    :type number: int
    :param log_seed: whether to log
    :type log_seed: bool
    """

    random.seed(number)
    numpy.random.seed(number)
    torch.cuda.manual_seed_all(number)
    torch.manual_seed(number)
    if log_seed:
        log('torch+numpy seed: %d' % number)


def is_cuda(mixed):
    """
    Check if model/tensor is on CUDA.

    :param mixed: model or tensor
    :type mixed: torch.nn.Module or torch.autograd.Variable or torch.Tensor
    :return: on cuda
    :rtype: bool
    """

    assert isinstance(mixed, torch.nn.Module) or isinstance(mixed, torch.autograd.Variable) \
        or isinstance(mixed, torch.Tensor), 'mixed has to be torch.nn.Module, torch.autograd.Variable or torch.Tensor'

    is_cuda = False
    if isinstance(mixed, torch.nn.Module):
        is_cuda = True
        for parameters in list(mixed.parameters()):
            is_cuda = is_cuda and parameters.is_cuda
    if isinstance(mixed, torch.autograd.Variable):
        is_cuda = mixed.is_cuda
    if isinstance(mixed, torch.Tensor):
        is_cuda = mixed.is_cuda

    return is_cuda


def estimate_size(mixed):
    """
    Estimate tensor size.

    :param tensor: tensor or model
    :type tensor: numpy.ndarray, torch.tensor, torch.autograd.Variable or torch.nn.Module
    :return: size in bits
    :rtype: int
    """

    # PyTorch types:
    # Data type 	dtype 	CPU tensor 	GPU tensor
    # 32-bit floating point 	torch.float32 or torch.float 	torch.FloatTensor 	torch.cuda.FloatTensor
    # 64-bit floating point 	torch.float64 or torch.double 	torch.DoubleTensor 	torch.cuda.DoubleTensor
    # 16-bit floating point 	torch.float16 or torch.half 	torch.HalfTensor 	torch.cuda.HalfTensor
    # 8-bit integer (unsigned) 	torch.uint8 	torch.ByteTensor 	torch.cuda.ByteTensor
    # 8-bit integer (signed) 	torch.int8 	torch.CharTensor 	torch.cuda.CharTensor
    # 16-bit integer (signed) 	torch.int16 or torch.short 	torch.ShortTensor 	torch.cuda.ShortTensor
    # 32-bit integer (signed) 	torch.int32 or torch.int 	torch.IntTensor 	torch.cuda.IntTensor
    # 64-bit integer (signed) 	torch.int64 or torch.long 	torch.LongTensor 	torch.cuda.LongTensor

    # Numpy types:
    # Data type 	Description
    # bool_ 	Boolean (True or False) stored as a byte
    # int_ 	Default integer type (same as C long; normally either int64 or int32)
    # intc 	Identical to C int (normally int32 or int64)
    # intp 	Integer used for indexing (same as C ssize_t; normally either int32 or int64)
    # int8 	Byte (-128 to 127)
    # int16 	Integer (-32768 to 32767)
    # int32 	Integer (-2147483648 to 2147483647)
    # int64 	Integer (-9223372036854775808 to 9223372036854775807)
    # uint8 	Unsigned integer (0 to 255)
    # uint16 	Unsigned integer (0 to 65535)
    # uint32 	Unsigned integer (0 to 4294967295)
    # uint64 	Unsigned integer (0 to 18446744073709551615)
    # float_ 	Shorthand for float64.
    # float16 	Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
    # float32 	Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
    # float64 	Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
    # complex_ 	Shorthand for complex128.
    # complex64 	Complex number, represented by two 32-bit floats (real and imaginary components)
    # complex128 	Complex number, represented by two 64-bit floats (real and imaginary components)

    types8 = [
        torch.uint8, torch.int8,
        numpy.int8, numpy.uint8, numpy.bool_,
    ]

    types16 = [
        torch.float16, torch.half,
        torch.int16, torch.short,
        numpy.int16, numpy.uint16, numpy.float16,
    ]

    types32 = [
        torch.float32, torch.float,
        torch.int32, torch.int,
        numpy.int32, numpy.uint32, numpy.float32,
    ]

    types64 = [
        torch.float64, torch.double,
        torch.int64, torch.long,
        numpy.int64, numpy.uint64, numpy.float64, numpy.complex64,
        numpy.int_, numpy.float_
    ]

    types128 = [
        numpy.complex_, numpy.complex128
    ]

    if isinstance(mixed, torch.nn.Module):

        size = 0
        modules = mixed.modules()
        for module in modules:
            for parameters in list(module.parameters()):
                size += estimate_size(parameters)
        return size

    if isinstance(mixed, (torch.Tensor, numpy.ndarray)):

        if mixed.dtype in types128:
            bits = 128
        elif mixed.dtype in types64:
            bits = 64
        elif mixed.dtype in types32:
            bits = 32
        elif mixed.dtype in types16:
            bits = 16
        elif mixed.dtype in types8:
            bits = 8
        else:
            assert False, 'could not identify torch.Tensor or numpy.ndarray type %s' % mixed.type()

        size = numpy.prod(mixed.shape)
        return size*bits

    elif isinstance(mixed, torch.autograd.Variable):
        return estimate_size(mixed.data)
    else:
        assert False, 'unsupported tensor size for estimating size, either numpy.ndarray, torch.tensor or torch.autograd.Variable'


def bits2MiB(bits):
    """
    Convert bits to MiB.

    :param bits: number of bits
    :type bits: int
    :return: MiB
    :rtype: float
    """

    return bits/(8*1024*1024)


def bits2MB(bits):
    """
    Convert bits to MB.

    :param bits: number of bits
    :type bits: int
    :return: MiB
    :rtype: float
    """

    return bits/(8*1000*1000)


def bytes2MiB(bytes):
    """
    Convert bytes to MiB.

    :param bytes: number of bytes
    :type bytes: int
    :return: MiB
    :rtype: float
    """

    return bytes/(1024*1024)


def bytes2MB(bytes):
    """
    Convert bytes to MB.

    :param bytes: number of bytes
    :type bytes: int
    :return: MiB
    :rtype: float
    """

    return bytes/(1000*1000)


bMiB = bits2MiB
BMiB = bytes2MiB
bMB = bits2MB
BMB = bytes2MB


def parameter_sizes(model, layers=None):
    """
    Get number of parameters.

    :param model: model
    :type model: torch.nn.Module
    :return: number of parameters
    :rtype: int
    """

    parameters = list(model.parameters())
    if layers is None:
        layers = list(range(len(parameters)))

    sizes = []
    min_size = [0]
    max_size = [0]

    for i in layers:
        parameter = parameters[i]
        size = list(parameter.shape)
        sizes.append(size)
        if numpy.prod(size) < numpy.prod(min_size):
            min_size = size
        if numpy.prod(size) > numpy.prod(max_size):
            max_size = size

    assert len(sizes) > 0
    if len(sizes) == 1:
        return numpy.prod(sizes[0]), sizes, min_size, max_size
    else:
        return functools.reduce(lambda a,b : numpy.prod(a) + numpy.prod(b), sizes), sizes, min_size, max_size


def all_parameters(model, layers=None):
    """
    Accumulate all parameters in one vector.

    :param model: model
    :type model: torch.nn.Module
    :return: one 1D tensor of all parameters
    :rtype: torch.Tensor
    """

    parameters = list(model.parameters())
    if layers is None:
        layers = list(range(len(parameters)))

    accumulated_parameters = None
    for i in layers:
        accumulated_parameters = concatenate(accumulated_parameters, parameters[i].data.view(-1))
    return accumulated_parameters


def clone(model):
    """
    Clone model.

    :param model: model
    :type model: torch.nn.Module
    :return: model
    :rtype: torch.nn.Module
    """

    cloned_model = deepcopy(model)
    assert cloned_model.training is model.training
    assert is_cuda(model) is is_cuda(cloned_model)

    return cloned_model


def copy(to_model, from_model):
    """
    Copy parameters.

    :param to_model: model to copy to
    :type to_model: torch.nn.Module
    :param from_model: model to copy from
    :type from_model: torch.nn.Module
    """

    to_parameters = list(to_model.parameters())
    from_parameters = list(from_model.parameters())
    assert len(to_parameters) == len(from_parameters)

    for i in range(len(to_parameters)):
        to_parameters[i].data = from_parameters[i].data


def binary_labels(classes):
    """
    Convert 0,1 labels to -1,1 labels.

    :param classes: classes as B x 1
    :type classes: torch.autograd.Variable or torch.Tensor
    """

    classes[classes == 0] = -1
    return classes


def one_hot(classes, C):
    """
    Convert class labels to one-hot vectors.

    :param classes: classes as B x 1
    :type classes: torch.autograd.Variable or torch.Tensor
    :param C: number of classes
    :type C: int
    :return: one hot vector as B x C
    :rtype: torch.autograd.Variable or torch.Tensor
    """

    assert isinstance(classes, torch.autograd.Variable) or isinstance(classes, torch.Tensor), 'classes needs to be torch.autograd.Variable or torch.Tensor'
    assert len(classes.size()) == 2 or len(classes.size()) == 1, 'classes needs to have rank 2 or 1'
    assert C > 0

    if len(classes.size()) < 2:
        classes = classes.view(-1, 1)

    one_hot = torch.Tensor(classes.size(0), C)
    if is_cuda(classes):
         one_hot = one_hot.cuda()

    if isinstance(classes, torch.autograd.Variable):
        one_hot = torch.autograd.Variable(one_hot)

    one_hot.zero_()
    one_hot.scatter_(1, classes, 1)

    return one_hot


def percentile(t, q):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + builtins.round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()

    return result


# https://github.com/pytorch/pytorch/issues/22812
def topk(tensor, k, use_sort=True):
    if use_sort:
        sort, idx = torch.sort(tensor, descending=True)
        return sort[:k], idx[:k]
    else:
        return torch.topk(tensor, k=k, sorted=False)


def project_ball(tensor, epsilon=1, ord=2):
    """
    Compute the orthogonal projection of the input tensor (as vector) onto the L_ord epsilon-ball.

    **Assumes the first dimension to be batch dimension, which is preserved.**

    :param tensor: variable or tensor
    :type tensor: torch.autograd.Variable or torch.Tensor
    :param epsilon: radius of ball.
    :type epsilon: float
    :param ord: order of norm
    :type ord: int
    :return: projected vector
    :rtype: torch.autograd.Variable or torch.Tensor
    """

    assert isinstance(tensor, torch.Tensor) or isinstance(tensor, torch.autograd.Variable), 'given tensor should be torch.Tensor or torch.autograd.Variable'

    if ord == 0:
        assert epsilon >= 0

        size = list(tensor.shape)
        flattened_size = int(numpy.prod(size[1:]))

        tensor = tensor.view(-1, flattened_size)

        k = int(math.ceil(epsilon))
        k = min(k, tensor.size(1) - 1)
        assert k > 0

        for b in range(tensor.size(0)):
            _, indices = topk(tensor[b], k=k)
            complement_indices = numpy.delete(numpy.arange(tensor.size(1)), indices.cpu().numpy())
            tensor[b][complement_indices] = 0

        tensor = tensor.view(size)
    elif ord == 1:
        # ! Does not allow differentiation obviously!
        cuda = is_cuda(tensor)
        array = tensor.detach().cpu().numpy()
        array = cnumpy.project_ball(array, epsilon=epsilon, ord=ord)
        tensor = torch.from_numpy(array)
        if cuda:
            tensor = tensor.cuda()
    elif ord == 2:
        size = list(tensor.shape)
        flattened_size = int(numpy.prod(size[1:]))

        tensor = tensor.view(-1, flattened_size)
        clamped = torch.clamp(epsilon/torch.norm(tensor, 2, dim=1), max=1)
        clamped = clamped.view(-1, 1)

        tensor = tensor * clamped
        if len(size) == 4:
            tensor = tensor.view(-1, size[1], size[2], size[3])
        elif len(size) == 2:
            tensor = tensor.view(-1, size[1])
    elif ord == float('inf'):
        tensor = torch.clamp(tensor, min=-epsilon, max=epsilon)
    else:
        raise NotImplementedError()

    return tensor


def project_sphere(tensor, epsilon=1, ord=2):
    """
    Compute the orthogonal projection of the input tensor (as vector) onto the L_ord epsilon-ball.

    **Assumes the first dimension to be batch dimension, which is preserved.**

    :param tensor: variable or tensor
    :type tensor: torch.autograd.Variable or torch.Tensor
    :param epsilon: radius of ball.
    :type epsilon: float
    :param ord: order of norm
    :type ord: int
    :return: projected vector
    :rtype: torch.autograd.Variable or torch.Tensor
    """

    assert isinstance(tensor, torch.Tensor) or isinstance(tensor, torch.autograd.Variable), 'given tensor should be torch.Tensor or torch.autograd.Variable'

    size = tensor.size()
    flattened_size = numpy.prod(numpy.array(size[1:]))

    tensor = tensor.view(-1, flattened_size)
    tensor = tensor/torch.norm(tensor, dim=1, ord=ord).view(-1, 1)
    tensor *= epsilon

    if len(size) == 4:
        tensor = tensor.view(-1, size[1], size[2], size[3])
    elif len(size) == 2:
        tensor = tensor.view(-1, size[1])

    return tensor


def tensor_or_value(mixed):
    """
    Get tensor or single value.

    :param mixed: variable, tensor or value
    :type mixed: mixed
    :return: tensor or value
    :rtype: torch.Tensor or value
    """

    if isinstance(mixed, torch.Tensor):
        if mixed.numel() > 1:
            return mixed
        else:
            return mixed.item()
    elif isinstance(mixed, torch.autograd.Variable):
        return tensor_or_value(mixed.cpu().data)
    else:
        return mixed


def as_variable(mixed, cuda=False, requires_grad=False):
    """
    Get a tensor or numpy array as variable.

    :param mixed: input tensor
    :type mixed: torch.Tensor or numpy.ndarray
    :param device: gpu or not
    :type device: bool
    :param requires_grad: gradients
    :type requires_grad: bool
    :return: variable
    :rtype: torch.autograd.Variable
    """

    assert isinstance(mixed, numpy.ndarray) or isinstance(mixed, torch.Tensor), 'input needs to be numpy.ndarray or torch.Tensor'

    if isinstance(mixed, numpy.ndarray):
        mixed = torch.from_numpy(mixed)

    if cuda:
        mixed = mixed.cuda()
    return torch.autograd.Variable(mixed, requires_grad)


def concatenate(tensor1, tensor2, axis=0):
    """
    Basically a wrapper for torch.dat, with the exception
    that the array itself is returned if its None or evaluates to False.

    :param tensor1: input array or None
    :type tensor1: mixed
    :param tensor2: input array
    :type tensor2: numpy.ndarray
    :param axis: axis to concatenate
    :type axis: int
    :return: concatenated array
    :rtype: numpy.ndarray
    """

    assert isinstance(tensor2, torch.Tensor) or isinstance(tensor2, torch.autograd.Variable)
    if tensor1 is not None:
        assert isinstance(tensor1, torch.Tensor) or isinstance(tensor1, torch.autograd.Variable)
        return torch.cat((tensor1, tensor2), axis=axis)
    else:
        return tensor2


def tile(a, dim, n_tile):
    """
    Numpy-like tiling in torch.
    https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/2

    :param a: tensor
    :type a: torch.Tensor or torch.autograd.Variable
    :param dim: dimension to tile
    :type dim: int
    :param n_tile: number of tiles
    :type n_tile: int
    :return: tiled tensor
    :rtype: torch.Tensor or torch.autograd.Variable
    """

    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(numpy.concatenate([init_dim * numpy.arange(n_tile) + i for i in range(init_dim)]))
    if is_cuda(a):
        order_index = order_index.cuda()
    return torch.index_select(a, dim, order_index)


def randn_like(tensor):
    """
    Normal random numbers like tensor.

    :param tensor: tensor
    :type tensor: torch.Tensor
    :return: random tensor
    :rtype: torch.Tensor
    """

    random = torch.randn(tensor.size())
    if is_cuda(tensor):
        random = random.cuda()
    return random


def expand_as(tensor, tensor_as):
    """
    Expands the tensor using view to allow broadcasting.

    :param tensor: input tensor
    :type tensor: torch.Tensor or torch.autograd.Variable
    :param tensor_as: reference tensor
    :type tensor_as: torch.Tensor or torch.autograd.Variable
    :return: tensor expanded with singelton dimensions as tensor_as
    :rtype: torch.Tensor or torch.autograd.Variable
    """

    view = list(tensor.size())
    for i in range(len(tensor.size()), len(tensor_as.size())):
        view.append(1)

    return tensor.view(view)


def round(tensor, decimal_places):
    """
    Round floats to the given number of decimal places.

    :param tensor: input tensor
    :type tensor: torch.Tensor
    :param decimal_places: number of decimal places
    :types decimal_places: int
    :return: rounded tensor
    :rtype: torch.Tensor
    """

    factor = 10**decimal_places
    return torch.round(tensor*factor)/factor


def uniform_norm_l0_zero(batch_size, dim, epsilon=10, cuda=False):
    """
    Uniform L_0 sampling bit setting elements to zero.

    :param batch_size: how many vectors to sample
    :type batch_size: int
    :param dim: dimensionality of vectors
    :type dim: int
    :param epsilon: epsilon-ball
    :type epsilon: float
    :param cuda: sample on cuda
    :type cuda: bool
    :return: batch_size x dim tensor
    :rtype: numpy.ndarray
    """

    epsilon = int(epsilon)

    if cuda:
        mask = torch.cuda.FloatTensor(batch_size, dim).fill_(1)
    else:
        mask = torch.FloatTensor(batch_size, dim).fill_(1)

    for b in range(batch_size):
        l = numpy.random.randint(0, epsilon)
        indices = numpy.random.choice(dim, size=l, replace=False)
        mask[b, indices] = 0

    return mask


def uniform_norm_l0(batch_size, dim, epsilon=10, cuda=False, low=0, high=1):
    """
    Uniform L_0 sampling

    :param batch_size: how many vectors to sample
    :type batch_size: int
    :param dim: dimensionality of vectors
    :type dim: int
    :param epsilon: epsilon-ball
    :type epsilon: float
    :param low: low sample value for uniform distribution
    :type low: float
    :param high: high sample value for uniform distribution
    :type high: float
    :param cuda: sample on cuda
    :type cuda: bool
    :return: batch_size x dim tensor
    :rtype: numpy.ndarray
    """

    epsilon = int(epsilon)

    if cuda:
        mask = torch.cuda.FloatTensor(batch_size, dim).fill_(0)
    else:
        mask = torch.FloatTensor(batch_size, dim).fill_(0)

    for b in range(batch_size):
        l = numpy.random.randint(0, epsilon)
        indices = numpy.random.choice(dim, size=l, replace=False)
        if cuda:
            random = torch.cuda.FloatTensor(l).uniform_(low, high)
        else:
            random = torch.FloatTensor(l).uniform_(low, high)
        mask[b, indices] = random

    return mask


def uniform_sphere_l0(batch_size, dim, epsilon=10, cuda=False, low=0, high=1):
    """
    Uniform L_0 sampling

    :param batch_size: how many vectors to sample
    :type batch_size: int
    :param dim: dimensionality of vectors
    :type dim: int
    :param epsilon: epsilon-ball
    :type epsilon: float
    :param low: low sample value for uniform distribution
    :type low: float
    :param high: high sample value for uniform distribution
    :type high: float
    :param cuda: sample on cuda
    :type cuda: bool
    :return: batch_size x dim tensor
    :rtype: numpy.ndarray
    """

    epsilon = int(epsilon)

    if cuda:
        mask = torch.cuda.FloatTensor(batch_size, dim).fill_(0)
    else:
        mask = torch.FloatTensor(batch_size, dim).fill_(0)

    for b in range(batch_size):
        indices = numpy.random.choice(dim, size=epsilon, replace=False)
        if cuda:
            random = torch.cuda.FloatTensor(epsilon).uniform_(low, high)
        else:
            random = torch.FloatTensor(epsilon).uniform_(low, high)
        mask[b, indices] = random

    return mask


def uniform_norm(batch_size, dim, epsilon=1, ord=2, cuda=False, low=0, high=1):
    """
    Sample vectors uniformly by norm and direction separately.

    :param batch_size: how many vectors to sample
    :type batch_size: int
    :param dim: dimensionality of vectors
    :type dim: int
    :param epsilon: epsilon-ball
    :type epsilon: float
    :param ord: norm to use
    :type ord: int
    :param cuda: sample on cuda
    :type cuda: bool
    :return: batch_size x dim tensor
    :rtype: numpy.ndarray
    """

    if ord == 0:
        return uniform_norm_l0(batch_size, dim, epsilon, cuda, low, high)
    else:
        if cuda:
            random = torch.cuda.FloatTensor(batch_size, dim).normal_()
        else:
            random = torch.FloatTensor(batch_size, dim).normal_()
        #random /= numpy.repeat(numpy.linalg.norm(random, ord=ord, axis=1).reshape(-1, 1), axis=1, repeats=dim)
        random /= torch.norm(random, p=ord, dim=1).view(-1, 1)
        random *= epsilon

        if cuda:
            uniform = torch.cuda.FloatTensor(batch_size, 1).uniform_(0, 1)
        else:
            uniform = torch.FloatTensor(batch_size, 1).uniform_(0, 1)

        random *= uniform

        return random


def uniform_ball(batch_size, dim, epsilon=1, ord=2, cuda=False):
    """
    Sample vectors uniformly in the n-ball.

    See Harman et al., On decompositional algorithms for uniform sampling from n-spheres and n-balls.

    :param batch_size: how many vectors to sample
    :type batch_size: int
    :param dim: dimensionality of vectors
    :type dim: int
    :param epsilon: epsilon-ball
    :type epsilon: float
    :param ord: norm to use
    :type ord: int
    :param cuda: sample on cuda
    :type cuda: bool
    :return: batch_size x dim tensor
    :rtype: numpy.ndarray
    """

    if ord == 0:
        raise NotImplementedError
    else:
        if cuda:
            random = torch.cuda.FloatTensor(batch_size, dim).normal_()
        else:
            random = torch.FloatTensor(batch_size, dim).normal_()
        #random /= numpy.repeat(numpy.linalg.norm(random, ord=ord, axis=1).reshape(-1, 1), axis=1, repeats=dim)
        random /= torch.norm(random, p=ord, dim=1).view(-1, 1)
        random *= epsilon

        if cuda:
            uniform = torch.cuda.FloatTensor(batch_size, 1).uniform_(0, 1)
        else:
            uniform = torch.FloatTensor(batch_size, 1).uniform_(0, 1)

        uniform = uniform**(1./dim)
        random *= uniform

        return random


def uniform_sphere(batch_size, dim, epsilon=1, ord=2, cuda=False, low=0, high=1):
    """
    Sample vectors uniformly on the n-sphere.

    See Harman et al., On decompositional algorithms for uniform sampling from n-spheres and n-balls.

    :param batch_size: how many vectors to sample
    :type batch_size: int
    :param dim: dimensionality of vectors
    :type dim: int
    :param epsilon: epsilon-ball
    :type epsilon: float
    :param ord: norm to use
    :type ord: int
    :param cuda: sample on cuda
    :type cuda: bool
    :return: batch_size x dim tensor
    :rtype: numpy.ndarray
    """

    if ord == 0:
        return uniform_sphere_l0(batch_size, dim, epsilon, cuda, low, high)
    else:
        if cuda:
            random = torch.cuda.FloatTensor(batch_size, dim).normal_()
        else:
            random = torch.FloatTensor(batch_size, dim).normal_()
        #random /= numpy.repeat(numpy.linalg.norm(random, ord=ord, axis=1).reshape(-1, 1), axis=1, repeats=dim)
        random /= torch.norm(random, p=ord, dim=1).view(-1, 1)
        random *= epsilon

        return random


def classification_error(logits, targets, reduction='mean'):
    """
    Accuracy.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduce: reduce to number or keep per element
    :type reduce: bool
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert logits.size()[0] == targets.size()[0]
    assert len(list(targets.size())) == 1# or (len(list(targets.size())) == 2 and targets.size(1) == 1)
    assert len(list(logits.size())) == 2

    if logits.size()[1] > 1:
        values, indices = torch.max(torch.nn.functional.softmax(logits, dim=1), dim=1)
    else:
        indices = torch.round(torch.nn.functional.sigmoid(logits)).view(-1)

    errors = torch.clamp(torch.abs(indices.long() - targets.long()), max=1)
    if reduction == 'mean':
        return torch.mean(errors.float())
    elif reduction == 'sum':
        return torch.sum(errors.float())
    else:
        return errors


def softmax(logits, dim=1):
    """
    Softmax.

    :param logits: logits
    :type logits: torch.Tensor
    :param dim: dimension
    :type dim: int
    :return: softmax
    :rtype: torch.Tensor
    """

    if logits.size()[1] > 1:
        return torch.nn.functional.softmax(logits, dim=dim)
    else:
        probabilities = torch.nn.functional.sigmoid(logits)
        return torch.cat((1 - probabilities,  probabilities), dim=dim)


def binary_classification_loss(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert logits.size()[0] == targets.size()[0]
    assert len(list(targets.size())) == 1# or (len(list(targets.size())) == 2 and targets.size(1) == 1)
    assert len(list(logits.size())) == 2

    targets = one_hot(targets, logits.size(1))
    if logits.size()[1] > 1:
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)
    else:
        raise NotImplementedError


def classification_loss(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert logits.size()[0] == targets.size()[0]
    assert len(list(targets.size())) == 1# or (len(list(targets.size())) == 2 and targets.size(1) == 1)
    assert len(list(logits.size())) == 2

    if logits.size()[1] > 1:
        return torch.nn.functional.cross_entropy(logits, targets, reduction=reduction)
    else:
        # probability 1 is class 1
        # probability 0 is class 0
        return torch.nn.functional.binary_cross_entropy(torch.nn.functional.sigmoid(logits).view(-1), targets.float(), reduction=reduction)


def noisy_classification_loss(logits, targets, reduction='mean', noise_rate=0.1, K=10):
    """
    Loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert logits.size()[0] == targets.size()[0]
    assert len(list(targets.size())) == 1# or (len(list(targets.size())) == 2 and targets.size(1) == 1)
    assert len(list(logits.size())) == 2

    noisy_targets = targets.clone()
    random_targets = torch.zeros_like(targets).random_(K)
    noisy = torch.rand(targets.size(0))
    noisy = (noisy < noise_rate)
    if is_cuda(targets):
        noisy = noisy.cuda()
    noisy_targets[noisy] = random_targets[noisy]

    if logits.size()[1] > 1:
        return torch.nn.functional.cross_entropy(logits, noisy_targets, reduction=reduction)
    else:
        raise NotImplementedError


def classification_loss_(logits, targets, reduction='mean', eps=13-6):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert len(list(logits.size())) == len(list(targets.size()))
    assert logits.size()[0] == targets.size()[0]
    assert logits.size()[1] == targets.size()[1]
    assert logits.size()[1] > 1

    targets = one_hot(targets, logits.size(1))
    divergences = torch.sum(- targets * torch.log(torch.nn.functional.softmax(logits, dim=1) + eps), dim=1)
    if reduction == 'mean':
        return torch.mean(divergences)
    elif reduction == 'sum':
        return torch.sum(divergences)
    else:
        return divergences


def smooth_classification_loss(logits, targets, reduction='mean', epsilon=0.1, K=10):
    """
    Loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert logits.size()[0] == targets.size()[0]
    assert len(list(targets.size())) == 1# or (len(list(targets.size())) == 2 and targets.size(1) == 1)
    assert len(list(logits.size())) == 2

    if logits.size()[1] > 1:
        distributions = one_hot(targets, K)
        distributions = distributions*(1 - epsilon) + 1./K*epsilon
        return cross_entropy_divergence(logits, distributions, reduction=reduction)
    else:
        raise NotImplementedError


def scaled_classification_loss(logits, targets, reduction='mean', epsilon=0.95, K=10):
    """
    Loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert logits.size()[0] == targets.size()[0]
    assert len(list(targets.size())) == 1# or (len(list(targets.size())) == 2 and targets.size(1) == 1)
    assert len(list(logits.size())) == 2

    if logits.size()[1] > 1:
        probabilities = torch.nn.functional.softmax(logits, dim=1) + 1e-9
        probabilities = probabilities * (1 - epsilon) + 1. / K * epsilon
        divergences = torch.sum(- targets * torch.log(probabilities), dim=1)
        if reduction == 'mean':
            return torch.mean(divergences)
        elif reduction == 'sum':
            return torch.sum(divergences)
        else:
            return divergences
    else:
        raise NotImplementedError


def cross_entropy_divergence(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert len(list(logits.size())) == len(list(targets.size()))
    assert logits.size()[0] == targets.size()[0]
    assert logits.size()[1] == targets.size()[1]
    assert logits.size()[1] > 1

    divergences = torch.sum(- targets * torch.nn.functional.log_softmax(logits, dim=1), dim=1)
    if reduction == 'mean':
        return torch.mean(divergences)
    elif reduction == 'sum':
        return torch.sum(divergences)
    else:
        return divergences
