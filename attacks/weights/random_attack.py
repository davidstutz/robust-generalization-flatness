import torch
from .attack import *
import common.torch
from .norms import Norm


class RandomAttack(Attack):
    """
    Simple floating point attack on network weights using additive perturbations.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(RandomAttack, self).__init__()

        self.epochs = None
        """ (int) Maximum number of iterations. """

    def run(self, model, testset, objective):
        """
        Run attack.

        :param model: model to attack
        :type model: torch.nn.Module
        :param testset: test set
        :type testset: torch.utils.data.DataLoader
        :param objective: objective
        :type objective: UntargetedObjective or TargetedObjective
        """

        super(RandomAttack, self).run(model, testset, objective)

        assert self.epochs is not None
        assert self.epochs > 0
        assert self.norm is not None
        assert isinstance(self.norm, Norm)

        is_cuda = common.torch.is_cuda(model)
        perturbed_model = common.torch.clone(model)
        forward_model, _ = self.quantize(model)

        success_error = 1e12
        # also allowing zero epochs
        if self.training:
            success_model = common.torch.clone(perturbed_model)
        else:
            success_model = common.torch.clone(perturbed_model).cpu()

        for i in range(self.epochs):
            self.initialize(model, perturbed_model)

            self.project(model, perturbed_model)
            self.quantize(perturbed_model, forward_model)

            if self.epochs <= 1:
                if self.training:
                    success_model = common.torch.clone(forward_model)
                else:
                    success_model = common.torch.clone(forward_model).cpu()

                norm = self.norm(model, forward_model, self.layers)
                if self.progress is not None:
                    self.progress('attack weights', self.epochs * len(testset) - 1, self.epochs * len(testset), info='norm=%g' % norm, width=10)
            else:
                total_error = 0
                norm = self.norm(model, forward_model, self.layers)

                for b, (inputs, targets) in enumerate(testset):
                    inputs = common.torch.as_variable(inputs, is_cuda)
                    inputs = inputs.permute(0, 3, 1, 2)
                    targets = common.torch.as_variable(targets, is_cuda)

                    output_logits = forward_model(inputs)
                    assert not torch.isnan(output_logits).any()
                    assert not torch.isinf(output_logits).any()

                    error = objective(output_logits, targets).item()

                    success = objective.success(output_logits, targets).item()
                    true_confidence = objective.true_confidence(output_logits, targets).item()
                    target_confidence = objective.target_confidence(output_logits, targets).item()

                    global_step = i * len(testset) + b
                    if self.writer:
                        self.writer.add_scalar('%ssuccess' % self.prefix, success, global_step=global_step)
                        self.writer.add_scalar('%serror' % self.prefix, error, global_step=global_step)
                        self.writer.add_scalar('%snorm' % self.prefix, norm, global_step=global_step)
                        self.writer.add_scalar('%strue_confidence' % self.prefix, true_confidence, global_step=global_step)
                        self.writer.add_scalar('%starget_confidence' % self.prefix, target_confidence, global_step=global_step)

                    if self.progress is not None:
                        self.progress('attack weights', i * len(testset) + b, self.epochs*len(testset), info='success=%g error=%g norm=%g' % (
                            success,
                            error,
                            norm,
                        ), width=10)

                    total_error += error

                total_error /= len(testset)
                if total_error < success_error:
                    success_error = total_error
                    if self.training:
                        success_model = common.torch.clone(forward_model)
                    else:
                        success_model = common.torch.clone(forward_model).cpu()

        assert success_model is not None
        return success_model

