import torch
import common.torch


class Norm:
    def __init__(self):
        """
        Constructor.
        """

        self.norms = []
        """ ([float]) Norms per layer. """

    def __call__(self, model, perturbed_model, layers):
        """
        Norm.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to compute norm on
        :type layers: [int]
        """

        raise NotImplementedError()


class L2Norm(Norm):
    def __call__(self, model, perturbed_model, layers):
        """
        Norm.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to compute norm on
        :type layers: [int]
        """

        norms = []
        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())

        perturbations = None
        for i in layers:
            perturbation = perturbed_parameters[i].data - parameters[i].data
            perturbations = common.torch.concatenate(perturbations, perturbation.view(-1))
            norms.append(torch.norm(perturbation, p=2).item())

        self.norms = norms
        return torch.norm(perturbations, p=2).item() # important to avoid GPU memory overhead
