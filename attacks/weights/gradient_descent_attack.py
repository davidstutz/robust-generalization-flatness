import torch
from .attack import *
import common.torch
from common.log import log, LogLevel
from .norms import Norm
from .initializations import Initialization


class GradientDescentAttack(Attack):
    """
    Simple floating point attack on network weights using additive perturbations.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(GradientDescentAttack, self).__init__()

        self.epochs = None
        """ (int) Maximum number of iterations. """

        self.normalization = None
        """ (Normalization) Normalization. """

        self.backtrack = False
        """ (bool) Backtracking. """

        self.base_lr = None
        """ (float) Base learning rate. """

        self.momentum = None
        """ (float) Momentum. """

        self.lr_factor = 1
        """ (float) Learning rate decay. """

        self.error_bound = -1e12
        """ (float) Lower error bound. """

        self.early_stopping = False
        """ (bool) Stop on error bound. """

        self.lr = None
        """ (float) Learning rate. """

        self.eval = True
        """ (bool) Eval mode for network, BN. """

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

        super(GradientDescentAttack, self).run(model, testset, objective)

        assert self.epochs is not None
        assert self.epochs >= 0
        assert self.base_lr is not None
        assert self.momentum is not None
        assert self.backtrack is not None
        assert self.lr_factor is not None
        assert self.lr_factor >= 1
        assert self.norm is not None
        assert isinstance(self.norm, Norm)
        assert self.initialization is None or isinstance(self.initialization, Initialization)
        is_cuda = common.torch.is_cuda(model)
        assert model.training is not self.eval

        if self.early_stopping:
            log('using early stopping', LogLevel.WARNING)

        perturbed_model = common.torch.clone(model)

        # Quantization contexts enforces same quantization throughout the attack;
        # if no quantization is used, it's just None.
        forward_model, _ = self.quantize(model)  # avoid cloning models all over

        # model for backtracking to avoid memory leak
        next_perturbed_model = None
        next_forward_model = None
        if self.backtrack or self.early_stopping:
            next_perturbed_model = common.torch.clone(perturbed_model)
            next_forward_model, _ = self.quantize(model) # avoid cloning models all over

        i = 0
        gradients = []
        for param in perturbed_model.parameters():
            if i not in self.layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
            # add gradient for every layer so it can be indexed by i in self.layers
            gradients.append(torch.zeros_like(param))
            i += 1

        success_error = 1e12
        self.lr = self.base_lr

        self.initialize(model, perturbed_model)

        # also allowing zero epochs
        if self.training:
            success_model = common.torch.clone(perturbed_model)
        else:
            success_model = common.torch.clone(perturbed_model).cpu()

        for e in range(self.epochs):
            epoch_error = 0
            for b, (inputs, targets) in enumerate(testset):

                # !
                forward_parameters = list(forward_model.parameters())
                for i in self.layers:
                    if forward_parameters[i].grad is not None:
                        forward_parameters[i].grad.data.zero_()

                inputs = common.torch.as_variable(inputs, is_cuda)
                inputs = inputs.permute(0, 3, 1, 2)
                targets = common.torch.as_variable(targets, is_cuda)

                output_logits = forward_model(inputs)
                assert not torch.isnan(output_logits).any()
                assert not torch.isinf(output_logits).any()

                norm = self.norm(model, forward_model, self.layers)
                error = objective(output_logits, targets)

                error.backward()
                error = error.item()
                epoch_error += error

                success = objective.success(output_logits, targets).item()
                true_confidence = objective.true_confidence(output_logits, targets).item()
                target_confidence = objective.target_confidence(output_logits, targets).item()

                global_step = e*len(testset) + b
                if self.writer is not None:
                    self.writer.add_scalar('%ssuccess' % self.prefix, success, global_step=global_step)
                    self.writer.add_scalar('%serror' % self.prefix, error, global_step=global_step)
                    self.writer.add_scalar('%snorm' % self.prefix, norm, global_step=global_step)
                    self.writer.add_scalar('%strue_confidence' % self.prefix, true_confidence, global_step=global_step)
                    self.writer.add_scalar('%starget_confidence' % self.prefix, target_confidence, global_step=global_step)
                    self.writer.add_scalar('%slr' % self.prefix, self.lr, global_step=global_step)

                if self.progress is not None:
                    self.progress('attack weights', e * len(testset) + b, self.epochs*len(testset), info='success=%g error=%.2f/%.2f norm=%g lr=%g' % (
                        success,
                        error,
                        self.error_bound,
                        norm,
                        self.lr,
                    ), width=10)

                if b == len(testset) - 1:
                    epoch_error /= len(testset)
                    if epoch_error < success_error:
                       success_error = epoch_error
                       if self.training:
                           success_model = common.torch.clone(forward_model)
                       else:
                           success_model = common.torch.clone(forward_model).cpu()

                if self.normalization is not None:
                    self.normalization(model, forward_model, self.layers)

                forward_parameters = list(forward_model.parameters())
                perturbed_parameters = list(perturbed_model.parameters())

                for i in self.layers:
                    gradients[i].data = self.momentum*gradients[i].data + (1 - self.momentum)*forward_parameters[i].grad.data

                if self.backtrack:
                    next_perturbed_parameters = list(next_perturbed_model.parameters())
                    for i in self.layers:
                        next_perturbed_parameters[i].data = perturbed_parameters[i].data - self.lr * gradients[i].data

                    self.project(model, next_perturbed_model)
                    self.quantize(next_perturbed_model, next_forward_model)

                    next_output_logits = next_forward_model(inputs)
                    assert not torch.isnan(next_output_logits).any()
                    assert not torch.isinf(next_output_logits).any()

                    next_error = objective(next_output_logits, targets).item()

                    if next_error < self.error_bound:
                        self.lr = max(self.lr / self.lr_factor, 1e-12)
                    else:
                        if next_error < error:
                            for i in self.layers:
                                perturbed_parameters[i].data -= self.lr * gradients[i].data
                        else:
                            self.lr = max(self.lr/self.lr_factor, 1e-12)
                elif self.early_stopping:
                    next_perturbed_parameters = list(next_perturbed_model.parameters())
                    for i in self.layers:
                        next_perturbed_parameters[i].data = perturbed_parameters[i].data - self.lr * gradients[i].data

                    self.project(model, next_perturbed_model)
                    self.quantize(next_perturbed_model, next_forward_model)

                    next_output_logits = next_forward_model(inputs)
                    assert not torch.isnan(next_output_logits).any()
                    assert not torch.isinf(next_output_logits).any()

                    next_error = objective(next_output_logits, targets).item()

                    if next_error < self.error_bound:
                        self.lr = max(self.lr / self.lr_factor, 1e-12)
                    else:
                        for i in self.layers:
                            perturbed_parameters[i].data -= self.lr * gradients[i].data
                else:
                    if error < self.error_bound:
                        self.lr = max(self.lr / self.lr_factor, 1e-12)
                    else:
                        for i in self.layers:
                            perturbed_parameters[i].data -= self.lr * gradients[i].data

                self.project(model, perturbed_model)
                self.quantize(perturbed_model, forward_model)

                if e == self.epochs - 1 and b == len(testset) - 1 and self.progress is not None:
                    output_logits = forward_model(inputs)
                    assert not torch.isnan(output_logits).any()
                    assert not torch.isinf(output_logits).any()

                    error = objective(output_logits, targets)
                    success = objective.success(output_logits, targets).item()

                    if self.progress is not None:
                        self.progress('attack weights', e * len(testset) + b, self.epochs * len(testset),
                                      info='success=%g error=%.2f/%.2f norm=%g lr=%g' % (
                                          success,
                                          error,
                                          self.error_bound,
                                          norm,
                                          self.lr,
                                      ))

        assert success_model is not None
        return success_model

