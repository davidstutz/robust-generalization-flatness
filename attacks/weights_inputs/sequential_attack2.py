from .attack import Attack
import common.torch
import common.numpy
import numpy


class SequentialAttack2(Attack):
    """
    Generic attack.
    """

    def __init__(self, weight_attack, input_attack):
        """
        Constructor.
        """

        super(SequentialAttack2, self).__init__()

        self.input_attack = input_attack
        """ (attacks.Attack) Attack. """

        self.weight_attack = weight_attack
        """ (attacks.weights.Attack) Attack. """

        self.input_norm = getattr(input_attack, 'norm', None)
        """ (attacks.norms.Norm) Input norm. """

        self.weight_norm = getattr(weight_attack, 'norm', None)
        """ (attacks.weights.norms.Norm) Weight norm. """

        self.weight_projection = self.weight_attack.projection
        """ (attacks.weights.projections.Projection) Weight projection. """

    def run(self, model, testset, weight_objective, input_objective):
        """
        Run attack.

        :param model: model to attack
        :type model: torch.nn.Module
        :param testset: images
        :type testset: torch.utils.data.DataLoader
        :param input_objective: objective
        :type input_objective: UntargetedObjective or TargetedObjective
        :param weight_objective: objective
        :type weight_objective: UntargetedObjective or TargetedObjective
        """

        super(SequentialAttack2, self).run(model, testset, weight_objective, input_objective)
        self.weight_attack.projection = self.weight_projection

        self.weight_attack.training = True
        self.weight_attack.progress = self.progress
        weight_objective.reset()
        perturbed_model = self.weight_attack.run(model, testset, weight_objective)

        cuda = common.torch.is_cuda(model)
        images = None
        labels = None
        perturbations = None
        errors = None
        batch_size = None

        forward_model, _ = self.quantize(model)

        for b, (inputs, targets) in enumerate(testset):
            inputs = common.torch.as_variable(inputs, cuda)
            inputs = inputs.permute(0, 3, 1, 2)
            targets = common.torch.as_variable(targets, cuda)

            if batch_size is None:
                batch_size = inputs.size(0)

            input_objective.set(targets)
            self.input_attack.progress = self.progress
            perturbations_, errors_ = self.input_attack.run(perturbed_model, inputs, input_objective)

            labels = common.numpy.concatenate(labels, targets.detach().cpu().numpy())
            images = common.numpy.concatenate(images, inputs.detach().cpu().numpy())
            perturbations = common.numpy.concatenate(perturbations, perturbations_)
            errors = common.numpy.concatenate(errors, errors_)

        perturbations = numpy.transpose(perturbations, (0, 2, 3, 1))

        if not self.training:
            perturbed_model = perturbed_model.cpu()

        return perturbed_model, perturbations, errors

