import common.torch
import torch
import numpy


class Projection:
    def __call__(self, model, perturbed_model, layers):
        """
        Projection.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        """

        raise NotImplementedError()

    def reset(self):
        """
        Reset state of projection.
        """

        pass


class SequentialProjections(Projection):
    def __init__(self, projections):
        """
        Constructor.

        :param projections: list of projections
        :type projections: [Projection]
        """

        assert isinstance(projections, list)
        assert len(projections) > 0
        for projection in projections:
            assert isinstance(projection, Projection)

        self.projections = projections
        """ ([Projection]) Projections. """

    def __call__(self, model, perturbed_model, layers):
        """
        Projection.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        """

        assert len(layers) > 0

        for projection in self.projections:
            projection(model, perturbed_model, layers)


class BoxProjection(Projection):
    def __init__(self, min_bound=0, max_bound=1):
        """
        Constructor.

        :param min_bound: minimum bound
        :param min_bound: float
        :param max_bound: maximum bound
        :type: max_bound: float
        """

        self.min_bound = min_bound
        """ (float) Minimum bound. """

        self.max_bound = max_bound
        """ (float) Maximum bound. """

    def __call__(self, model, perturbed_model, layers):
        """
        Projection.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        """

        assert len(layers) > 0
        perturbed_parameters = list(perturbed_model.parameters())

        for i in layers:
            if self.max_bound is not None:
                perturbed_parameters[i].data = torch.min(torch.ones_like(perturbed_parameters[i].data) * self.max_bound, perturbed_parameters[i].data)
            if self.min_bound is not None:
                perturbed_parameters[i].data = torch.max(torch.ones_like(perturbed_parameters[i].data) * self.min_bound, perturbed_parameters[i].data)


class LayerWiseBoxProjection(Projection):
    def __init__(self, min_bounds=[], max_bounds=[]):
        """
        Constructor.

        :param min_bounds: minimum bound
        :param min_bounds: [float]
        :param max_bounds: maximum bound
        :type: max_bounds: [float]
        """

        assert len(min_bounds) > 0
        assert len(min_bounds) == len(max_bounds)

        self.min_bounds = min_bounds
        """ (float) Minimum bound. """

        self.max_bounds = max_bounds
        """ (float) Maximum bound. """

    def __call__(self, model, perturbed_model, layers):
        """
        Projection.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        """

        assert len(layers) > 0

        perturbed_parameters = list(perturbed_model.parameters())
        j = 0
        for i in layers:
            perturbed_parameters[i].data = torch.min(torch.ones_like(perturbed_parameters[i].data) * self.max_bounds[j], perturbed_parameters[i].data)
            perturbed_parameters[i].data = torch.max(torch.ones_like(perturbed_parameters[i].data) * self.min_bounds[j], perturbed_parameters[i].data)
            j += 1


class ReferenceBoxProjection(Projection):
    def __init__(self, min_bound=0, max_bound=1):
        """
        Constructor.

        :param min_bound: minimum bound
        :param min_bound: float
        :param max_bound: maximum bound
        :type: max_bound: float
        """

        self.min_bound = min_bound
        """ (float) Minimum bound. """

        self.max_bound = max_bound
        """ (float) Maximum bound. """

    def __call__(self, model, perturbed_model, layers):
        """
        Projection.

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

        max_parameter = 0
        min_parameter = 0
        for parameter in parameters:
            max_parameter = max(max_parameter, torch.max(parameter).item())
            min_parameter = max(min_parameter, torch.min(parameter).item())

        for i in layers:
            perturbed_parameters[i].data = torch.clamp(perturbed_parameters[i].data, max=max_parameter, min=min_parameter)


class ReferenceLayerWiseBoxProjection(Projection):
    def __call__(self, model, perturbed_model, layers):
        """
        Projection.

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

        for i in layers:
            perturbed_parameters[i].data = torch.clamp(perturbed_parameters[i].data, max=torch.max(parameters[i].data).item(), min=torch.min(parameters[i].data).item())


class ReferenceFilterWiseBoxProjection(Projection):
    def __call__(self, model, perturbed_model, layers):
        """
        Projection.

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

        for i in layers:
            size = list(parameters[i].data.shape)
            if len(size) > 2: # conv
                for j in range(size[0]):
                    perturbed_parameters[i].data[j] = torch.clamp(perturbed_parameters[i].data[j], max=torch.max(parameters[i].data[j]).item(), min=torch.min(parameters[i].data[j]).item())
            else: # bias + fc
                perturbed_parameters[i].data = torch.clamp(perturbed_parameters[i].data, max=torch.max(parameters[i].data).item(), min=torch.min(parameters[i].data).item())


class RelativeLayerWiseBoxProjection(Projection):
    def __init__(self, min_bound=0, max_bound=1, fractions=[]):
        """
        Constructor.

        :param min_bound: minimum bound
        :param min_bound: float
        :param max_bound: maximum bound
        :type: max_bound: float
        :param fractions: fractions
        :type fractions: [float]
        """

        assert len(fractions) > 0

        self.min_bound = min_bound
        """ (float) Minimum bound. """

        self.max_bound = max_bound
        """ (float) Maximum bound. """

        self.fractions = fractions
        """ ([float]) Fractions. """

    def __call__(self, model, perturbed_model, layers):
        """
        Projection.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        """

        assert len(layers) > 0

        perturbed_parameters = list(perturbed_model.parameters())
        j = 0
        for i in layers:
            perturbed_parameters[i].data = torch.clamp(perturbed_parameters[i].data, max=self.max_bound * self.fractions[j], min=self.min_bound * self.fractions[j])
            j += 1


class L2Projection(Projection):
    def __init__(self, epsilon=None, relative_epsilon=None):
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

    def __call__(self, model, perturbed_model, layers):
        """
        Projection.

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
            self.epsilon = numpy.linalg.norm(parameters, ord=2) * self.relative_epsilon_fraction

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())

        sizes = {}
        perturbations = None
        for i in layers:
            perturbation = perturbed_parameters[i].data - parameters[i].data
            sizes[i] = list(perturbation.shape)
            perturbations = common.torch.concatenate(perturbations, perturbation.view(-1))

        perturbations = common.torch.project_ball(perturbations.view(1, -1), epsilon=self.epsilon, ord=2).view(-1)
        perturbations = perturbations.view(-1)

        n_im1 = 0
        n_i = 0
        for i in layers:
            n_i += numpy.prod(sizes[i])
            perturbed_parameters[i].data = parameters[i].data + perturbations[n_im1:n_i].view(sizes[i])
            n_im1 = n_i


class LayerWiseL2Projection(Projection):
    def __init__(self, relative_epsilon):
        """
        Constructor.

        :param relative_epsilon: epsilon to project on
        :type relative_epsilon: float
        """

        self.relative_epsilon = relative_epsilon
        """ (float) Relative epsilon. """

        self.ord = 2
        """ (int) Project order. """

    def __call__(self, model, perturbed_model, layers):
        """
        Projection.

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

        for i in layers:
            perturbation = perturbed_parameters[i].data - parameters[i].data
            size = list(perturbation.shape)
            epsilon = self.relative_epsilon*torch.norm(parameters[i].data.view(-1), self.ord)
            perturbation = common.torch.project_ball(perturbation.view(1, -1), epsilon=epsilon, ord=self.ord).view(-1)
            perturbation = perturbation.view(size)
            perturbed_parameters[i].data = parameters[i].data + perturbation


class FilterWiseL2Projection(Projection):
    def __init__(self, relative_epsilon):
        """
        Constructor.

        :param relative_epsilon: epsilon to project on
        :type relative_epsilon: float
        """

        self.relative_epsilon = relative_epsilon
        """ (float) Relative epsilon. """

        self.ord = 2
        """ (int) Project order. """

    def __call__(self, model, perturbed_model, layers):
        """
        Projection.

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

        for i in layers:
            size = list(parameters[i].data.shape)
            perturbation = perturbed_parameters[i].data - parameters[i].data
            if len(size) > 2:
                for j in range(size[0]):
                    perturbation_j = perturbation[j]
                    epsilon = self.relative_epsilon * torch.norm(parameters[i].data[j].view(-1), self.ord)
                    perturbation_j = common.torch.project_ball(perturbation_j.view(1, -1), epsilon=epsilon, ord=self.ord).view(-1)
                    perturbation_j = perturbation_j.view(size[1:])
                    assert torch.norm(perturbation_j) <= epsilon + 1e-4
                    perturbed_parameters[i].data[j] = parameters[i].data[j] + perturbation_j
            else:
                epsilon = self.relative_epsilon * torch.norm(parameters[i].data.view(-1), self.ord)
                perturbation = common.torch.project_ball(perturbation.view(1, -1), epsilon=epsilon, ord=self.ord).view(-1)
                perturbation = perturbation.view(size)
                perturbed_parameters[i].data = parameters[i].data + perturbation


class L2RelativeProjection(Projection):
    def __init__(self, relative_epsilon):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = None
        """ (float) Epsilon. """

        self.relative_epsilon = relative_epsilon
        """ (bool) Relative epsilon. """

    def __call__(self, model, perturbed_model, layers):
        """
        Projection.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        """

        assert len(layers) > 0

        n, sizes, min_size, max_size = common.torch.parameter_sizes(model, layers)
        max_parameter = -1e12
        min_parameter = 1e12
        for parameter in model.parameters():
            max_parameter = max(max_parameter, torch.max(parameter).item())
            min_parameter = min(min_parameter, torch.min(parameter).item())
        self.epsilon = (max_parameter - min_parameter)*n*self.relative_epsilon_fraction

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())

        sizes = {}
        perturbations = None
        for i in layers:
            perturbation = perturbed_parameters[i].data - parameters[i].data
            sizes[i] = list(perturbation.shape)
            perturbations = common.torch.concatenate(perturbations, perturbation.view(-1))

        perturbations = common.torch.project_ball(perturbations.view(1, -1), epsilon=self.epsilon, ord=self.ord).view(-1)
        perturbations = perturbations.view(-1)

        n_im1 = 0
        n_i = 0
        for i in layers:
            n_i += numpy.prod(sizes[j])
            perturbed_parameters[i].data = parameters[i].data + perturbations[n_im1:n_i].view(sizes[j])
            n_im1 = n_i
