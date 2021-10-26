import common.torch
import torch
import torch.utils.data
import numpy
import common.numpy
from common.log import log


class Initialization:
    """
    Interface for initialization.
    """

    def __call__(self, model, perturbed_model, layers):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        """

        raise NotImplementedError()


class L2UniformNormInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_inf ball.
    """

    def __init__(self, epsilon=None, relative_epsilon=None, randomness=None):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

        self.relative_epsilon = (relative_epsilon is not None)
        """ (bool) Relative epsilon. """

        self.relative_epsilon_fraction = relative_epsilon
        """ (float) Relative epsilon fraction. """

        self.callable = common.torch.uniform_norm
        """ (callable) Sampler. """

        self.ord = 2
        """ (float) Norm. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        """

        assert len(layers) > 0

        if self.relative_epsilon:
            parameters = None
            for parameter in model.parameters():
                parameters = common.numpy.concatenate(parameters, parameter.view(-1).detach().cpu().numpy())
            self.epsilon = numpy.linalg.norm(parameters, ord=self.ord)*self.relative_epsilon_fraction

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())
        assert len(parameters) == len(perturbed_parameters)

        cuda = common.torch.is_cuda(model)
        n, _, _, _ = common.torch.parameter_sizes(model, layers=None)

        if self.randomness is not None:
            try:
                seed = next(self.randomness)[0].item()
            except StopIteration as e:
                log('resetting seeds')
                self.randomness = iter(self.randomness_)
                seed = next(self.randomness)[0].item()
        else:
            seed = numpy.random.randint(0, 2**32 - 1)
        common.torch.torch_numpy_seed(seed)

        random = self.callable(1, n, epsilon=self.epsilon, ord=self.ord, cuda=cuda).view(-1)

        n_i = 0
        for i in layers:
            size_i = list(parameters[i].data.shape)
            perturbed_parameters[i].data = parameters[i].data + random[n_i: n_i + numpy.prod(size_i)].view(size_i)
            n_i += numpy.prod(size_i)


class LayerWiseL2UniformNormInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_2 ball.
    """

    def __init__(self, relative_epsilon, randomness=None):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.relative_epsilon = relative_epsilon
        """ (float) Relative epsilon. """

        self.callable = common.torch.uniform_norm
        """ (callable) Sampler. """

        self.ord = 2
        """ (float) Norm. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        """

        assert len(layers) > 0

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())
        assert len(parameters) == len(perturbed_parameters)

        cuda = common.torch.is_cuda(model)
        if self.randomness is not None:
            try:
                seed = next(self.randomness)[0].item()
            except StopIteration as e:
                log('resetting seeds')
                self.randomness = iter(self.randomness_)
                seed = next(self.randomness)[0].item()
        else:
            seed = numpy.random.randint(0, 2**32 - 1)
        common.torch.torch_numpy_seed(seed)

        for i in layers:
            size = list(parameters[i].data.shape)
            epsilon = self.relative_epsilon*torch.norm(parameters[i].data.view(-1), self.ord)
            perturbed_parameters[i].data = parameters[i].data + self.callable(1, numpy.prod(size), epsilon=epsilon, ord=self.ord, cuda=cuda).view(size)


class LayerWiseL2UniformSphereInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_2 ball.
    """

    def __init__(self, relative_epsilon, randomness=None):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.relative_epsilon = relative_epsilon
        """ (float) Relative epsilon. """

        self.callable = common.torch.uniform_sphere
        """ (callable) Sampler. """

        self.ord = 2
        """ (float) Norm. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        """

        assert len(layers) > 0

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())
        assert len(parameters) == len(perturbed_parameters)

        cuda = common.torch.is_cuda(model)
        if self.randomness is not None:
            try:
                seed = next(self.randomness)[0].item()
            except StopIteration as e:
                log('resetting seeds')
                self.randomness = iter(self.randomness_)
                seed = next(self.randomness)[0].item()
        else:
            seed = numpy.random.randint(0, 2**32 - 1)
        common.torch.torch_numpy_seed(seed)

        for i in layers:
            size = list(parameters[i].data.shape)
            epsilon = self.relative_epsilon*torch.norm(parameters[i].data.view(-1), self.ord)
            perturbed_parameters[i].data = parameters[i].data + self.callable(1, numpy.prod(size), epsilon=epsilon, ord=self.ord, cuda=cuda).view(size)


class LayerWiseL2UniformBallInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_2 ball.
    """

    def __init__(self, relative_epsilon, randomness=None):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.relative_epsilon = relative_epsilon
        """ (float) Relative epsilon. """

        self.callable = common.torch.uniform_ball
        """ (callable) Sampler. """

        self.ord = 2
        """ (float) Norm. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        """

        assert len(layers) > 0

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())
        assert len(parameters) == len(perturbed_parameters)

        cuda = common.torch.is_cuda(model)
        if self.randomness is not None:
            try:
                seed = next(self.randomness)[0].item()
            except StopIteration as e:
                log('resetting seeds')
                self.randomness = iter(self.randomness_)
                seed = next(self.randomness)[0].item()
        else:
            seed = numpy.random.randint(0, 2**32 - 1)
        common.torch.torch_numpy_seed(seed)

        for i in layers:
            size = list(parameters[i].data.shape)
            epsilon = self.relative_epsilon*torch.norm(parameters[i].data.view(-1), self.ord)
            perturbed_parameters[i].data = parameters[i].data + self.callable(1, numpy.prod(size), epsilon=epsilon, ord=self.ord, cuda=cuda).view(size)


class FilterWiseL2UniformNormInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_2 ball.
    """

    def __init__(self, relative_epsilon, randomness=None):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.relative_epsilon = relative_epsilon
        """ (float) Relative epsilon. """

        self.callable = common.torch.uniform_norm
        """ (callable) Sampler. """

        self.ord = 2
        """ (float) Norm. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        """

        assert len(layers) > 0

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())
        assert len(parameters) == len(perturbed_parameters)

        cuda = common.torch.is_cuda(model)
        if self.randomness is not None:
            try:
                seed = next(self.randomness)[0].item()
            except StopIteration as e:
                log('resetting seeds')
                self.randomness = iter(self.randomness_)
                seed = next(self.randomness)[0].item()
        else:
            seed = numpy.random.randint(0, 2**32 - 1)
        common.torch.torch_numpy_seed(seed)

        for i in layers:
            size = list(parameters[i].data.shape)
            if len(size) > 2: # conv
                for j in range(size[0]):
                    epsilon = self.relative_epsilon * torch.norm(parameters[i].data[j].view(-1), self.ord)
                    perturbed_parameters[i].data[j] = parameters[i].data[j] + self.callable(1, numpy.prod(size[1:]), epsilon=epsilon, ord=self.ord, cuda=cuda).view(size[1:])
            else: # fc + bias
                epsilon = self.relative_epsilon*torch.norm(parameters[i].data.view(-1), self.ord)
                perturbed_parameters[i].data = parameters[i].data + self.callable(1, numpy.prod(size), epsilon=epsilon, ord=self.ord, cuda=cuda).view(size)


class FilterWiseL2UniformSphereInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_2 ball.
    """

    def __init__(self, relative_epsilon, randomness=None):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.relative_epsilon = relative_epsilon
        """ (float) Relative epsilon. """

        self.callable = common.torch.uniform_sphere
        """ (callable) Sampler. """

        self.ord = 2
        """ (float) Norm. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        """

        assert len(layers) > 0

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())
        assert len(parameters) == len(perturbed_parameters)

        cuda = common.torch.is_cuda(model)
        if self.randomness is not None:
            try:
                seed = next(self.randomness)[0].item()
            except StopIteration as e:
                log('resetting seeds')
                self.randomness = iter(self.randomness_)
                seed = next(self.randomness)[0].item()
        else:
            seed = numpy.random.randint(0, 2**32 - 1)
        common.torch.torch_numpy_seed(seed)

        for i in layers:
            size = list(parameters[i].data.shape)
            if len(size) > 2: # conv
                for j in range(size[0]):
                    epsilon = self.relative_epsilon * torch.norm(parameters[i].data[j].view(-1), self.ord)
                    perturbed_parameters[i].data[j] = parameters[i].data[j] + self.callable(1, numpy.prod(size[1:]), epsilon=epsilon, ord=self.ord, cuda=cuda).view(size[1:])
            else: # fc + bias
                epsilon = self.relative_epsilon*torch.norm(parameters[i].data.view(-1), self.ord)
                perturbed_parameters[i].data = parameters[i].data + self.callable(1, numpy.prod(size), epsilon=epsilon, ord=self.ord, cuda=cuda).view(size)


class LInfUniformNormInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_inf ball.
    """

    def __init__(self, epsilon=None, relative_epsilon=None, randomness=None):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

        self.relative_epsilon = (relative_epsilon is not None)
        """ (bool) Relative epsilon. """

        self.relative_epsilon_fraction = relative_epsilon
        """ (float) Relative epsilon fraction. """

        self.callable = common.torch.uniform_norm
        """ (callable) Sampler. """

        self.ord = float('inf')
        """ (float) Norm. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        assert len(layers) > 0

        if self.relative_epsilon:
            max_parameter = 0
            for parameter in model.parameters():
                max_parameter = max(max_parameter, torch.max(torch.abs(parameter)).item())
            self.epsilon = max_parameter*self.relative_epsilon_fraction

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())
        assert len(parameters) == len(perturbed_parameters)

        cuda = common.torch.is_cuda(model)
        n, _, _, _ = common.torch.parameter_sizes(model, layers=None)

        if self.randomness is not None:
            try:
                seed = next(self.randomness)[0].item()
            except StopIteration as e:
                log('resetting seeds')
                self.randomness = iter(self.randomness_)
                seed = next(self.randomness)[0].item()
        else:
            seed = numpy.random.randint(0, 2**32 - 1)
        common.torch.torch_numpy_seed(seed)

        random = self.callable(1, n, epsilon=self.epsilon, ord=self.ord, cuda=cuda).view(-1)

        n_i = 0
        for i in layers:
            size_i = list(parameters[i].data.shape)
            perturbed_parameters[i].data = parameters[i].data + random[n_i: n_i + numpy.prod(size_i)].view(size_i)
            n_i += numpy.prod(size_i)


class L2UniformSphereInitialization(LInfUniformNormInitialization):
    """
    Uniform initialization, wrt. norm and direction, in L_2 ball.
    """

    def __init__(self, epsilon=None, relative_epsilon=None, randomness=None):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        super(L2UniformSphereInitialization, self).__init__(epsilon, relative_epsilon)

        self.callable = common.torch.uniform_sphere
        """ (callable) Sampler. """

        self.ord = 2
        """ (float) Norm. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)


class L2RelativeUniformNormInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_2 ball.
    """

    def __init__(self, relative_epsilon, randomness=None):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = None
        """ (float) Epsilon. """

        self.relative_epsilon = relative_epsilon
        """ (float) Relative epsilon fraction. """

        self.callable = common.torch.uniform_norm
        """ (callable) Sampler. """

        self.ord = 2
        """ (float) Norm. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        """

        assert len(layers) > 0

        cuda = common.torch.is_cuda(model)
        n, _, _, _ = common.torch.parameter_sizes(model, layers=None)

        max_parameter = 0
        min_parameter = 0
        for parameter in model.parameters():
            max_parameter = max(max_parameter, torch.max(parameter).item())
            min_parameter = min(min_parameter, torch.min(parameter).item())
        self.epsilon = n*(max_parameter - min_parameter)*self.relative_epsilon

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())
        assert len(parameters) == len(perturbed_parameters)

        if self.randomness is not None:
            try:
                seed = next(self.randomness)[0].item()
            except StopIteration as e:
                log('resetting seeds')
                self.randomness = iter(self.randomness_)
                seed = next(self.randomness)[0].item()
        else:
            seed = numpy.random.randint(0, 2**32 - 1)
        common.torch.torch_numpy_seed(seed)

        random = self.callable(1, n, epsilon=self.epsilon, ord=self.ord, cuda=cuda).view(-1)

        n_i = 0
        for i in layers:
            size_i = list(parameters[i].data.shape)
            perturbed_parameters[i].data = parameters[i].data + random[n_i: n_i + numpy.prod(size_i)].view(size_i)
            n_i += numpy.prod(size_i)


class L2RelativeUniformSphereInitialization(L2RelativeUniformNormInitialization):
    """
    Uniform initialization, wrt. norm and direction, in L_2 ball.
    """

    def __init__(self, relative_epsilon, randomness=None):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = None
        """ (float) Epsilon. """

        self.relative_epsilon = relative_epsilon
        """ (float) Relative epsilon fraction. """

        self.callable = common.torch.uniform_sphere
        """ (callable) Sampler. """

        self.ord = 2
        """ (float) Norm. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)
