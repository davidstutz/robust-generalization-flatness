import torch
import common.torch


class Normalization:
    def __call__(self, model, perturbed_model, layers):
        """
        Normalize gradients.

        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to normalize
        :type layers: [int]
        """

        raise NotImplementedError()


class SignNormalization(Normalization):
    def __call__(self, model, perturbed_model, layers):
        """
        Normalize gradients.

        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to normalize
        :type layers: [int]
        """

        perturbed_parameters = list(perturbed_model.parameters())
        for i in layers:
            perturbed_parameters[i].grad.data = torch.sign(perturbed_parameters[i].grad.data)


class LInfNormalization(Normalization):
    def __call__(self, model, perturbed_model, layers):
        """
        Normalize gradients.

        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to normalize
        :type layers: [int]
        """

        grad_norm = 0
        perturbed_parameters = list(perturbed_model.parameters())
        for i in layers:
            grad_norm = max(grad_norm, torch.max(torch.abs(perturbed_parameters[i].grad.data)).item())

        for i in layers:
            perturbed_parameters[i].grad.data /= grad_norm


class RelativeLInfNormalization(Normalization):
    def __call__(self, model, perturbed_model, layers):
        """
        Normalize gradients.

        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to normalize
        :type layers: [int]
        """

        norm = 0
        parameters = list(model.parameters())
        for i in layers:
            norm = max(norm, torch.max(torch.abs(parameters[i].data)).item())

        grad_norm = 0
        perturbed_parameters = list(perturbed_model.parameters())
        for i in layers:
            grad_norm = max(grad_norm, torch.max(torch.abs(perturbed_parameters[i].grad.data)).item())

        #log('normalization: %g' % (norm/grad_norm))
        for i in layers:
            perturbed_parameters[i].grad.data *= norm/grad_norm


class LayerWiseRelativeLInfNormalization(Normalization):
    def __call__(self, model, perturbed_model, layers):
        """
        Normalize gradients.

        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to normalize
        :type layers: [int]
        """

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())

        for i in layers:
            perturbed_parameters[i].grad.data = torch.div(perturbed_parameters[i].grad.data, torch.max(torch.abs(perturbed_parameters[i].grad.data))*torch.max(torch.abs(parameters[i].data)))


class L2Normalization(Normalization):
    def __call__(self, model, perturbed_model, layers):
        """
        Normalize gradients.

        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to normalize
        :type layers: [int]
        """

        sizes = {}
        lengths = {}

        gradients = None
        perturbed_parameters = list(perturbed_model.parameters())

        for i in layers:
            gradient = perturbed_parameters[i].grad.data
            sizes[i] = gradient.shape
            gradients = common.torch.concatenate(gradients, gradient.view(-1))
            lengths[i] = (gradients.nelement() - gradient.nelement(), gradients.nelement())

        gradients = torch.div(gradients, torch.norm(gradients, p=2))

        for i in layers:
            perturbed_parameters[i].grad.data = gradients[lengths[i][0]:lengths[i][1]].view(sizes[i])


class RelativeL2Normalization(Normalization):
    def __call__(self, model, perturbed_model, layers):
        """
        Normalize gradients.

        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to normalize
        :type layers: [int]
        """

        sizes = {}
        lengths = {}

        weights = None
        gradients = None
        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())

        for i in layers:
            weight = parameters[i].data
            gradient = perturbed_parameters[i].grad.data
            sizes[i] = gradient.shape

            weights = common.torch.concatenate(weights, weight.view(-1))
            gradients = common.torch.concatenate(gradients, gradient.view(-1))

            lengths[i] = (gradients.nelement() - gradient.nelement(), gradients.nelement())

        gradients = torch.div(gradients, torch.norm(gradients, p=2))*torch.norm(weights, p=2)

        for i in layers:
            perturbed_parameters[i].grad.data = gradients[lengths[i][0]:lengths[i][1]].view(sizes[i])


class LayerWiseRelativeL2Normalization(Normalization):
    def __call__(self, model, perturbed_model, layers):
        """
        Normalize gradients.

        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to normalize
        :type layers: [int]
        """

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())

        for i in layers:
            perturbed_parameters[i].grad.data = torch.div(perturbed_parameters[i].grad.data, torch.norm(perturbed_parameters[i].grad.data, p=2))*torch.norm(parameters[i].data, p=2)
