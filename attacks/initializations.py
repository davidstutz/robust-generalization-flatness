import common.torch
import torch
import numpy
import common.numpy


class Initialization:
    """
    Interface for initialization.
    """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        raise NotImplementedError()


class SequentialInitializations(Initialization):
    """
    Combination of multiple initializers.
    """

    def __init__(self, initializations):
        """
        Constructor.

        :param initializations: list of initializations
        :type initializations: [Initializations]
        """

        assert isinstance(initializations, list)
        assert len(initializations) > 0
        for initialization in initializations:
            assert isinstance(initialization, Initialization)

        self.initializations = initializations
        """ ([Initializations]) Initializations. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        for initialization in self.initializations:
            initialization(images, perturbations)


class ZeroInitialization(Initialization):
    """
    Zero initialization.
    """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data.zero_()


class UniformInitialization(Initialization):
    """
    Zero initialization.
    """

    def __init__(self, min_bound=0, max_bound=1):
        """
        Constructor.

        :param min_bound: minimum bound
        :type min_bound: float
        :param max_bound: maximum bound
        :type max_bound: float
        """

        self.min_bound = min_bound
        """ (float) Min. """

        self.max_bound = max_bound
        """ (float) Max. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data.uniform_(self.min_bound, self.max_bound)


class L2UniformNormInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_2 ball.
    """

    def __init__(self, epsilon):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data = torch.from_numpy(common.numpy.uniform_norm(perturbations.size()[0], numpy.prod(perturbations.size()[1:]), epsilon=self.epsilon, ord=2).reshape(perturbations.size()).astype(numpy.float32))


class L2UniformSphereInitialization(Initialization):
    """
    Uniform initialization on L_2 sphere.
    """

    def __init__(self, epsilon):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data = torch.from_numpy(common.numpy.uniform_sphere(perturbations.size()[0], numpy.prod(perturbations.size()[1:]), epsilon=self.epsilon, ord=2).reshape(perturbations.size()).astype(numpy.float32))


class LInfUniformInitialization(Initialization):
    """
    Standard L_inf initialization as by Madry et al.
    """

    def __init__(self, epsilon):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data = torch.from_numpy(numpy.random.uniform(-self.epsilon, self.epsilon, size=perturbations.size()).astype(numpy.float32))


class LInfUniformNormInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_inf ball.
    """

    def __init__(self, epsilon):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data = torch.from_numpy(common.numpy.uniform_norm(perturbations.size()[0], numpy.prod(perturbations.size()[1:]), epsilon=self.epsilon, ord=float('inf')).reshape(perturbations.size()).astype(numpy.float32))


class LInfUniformSphereInitialization(Initialization):
    """
    Uniform initialization on L_inf sphere.
    """

    def __init__(self, epsilon):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data = torch.from_numpy(common.numpy.uniform_sphere(perturbations.size()[0], numpy.prod(perturbations.size()[1:]), epsilon=self.epsilon, ord=float('inf')).reshape(perturbations.size()).astype(numpy.float32))

