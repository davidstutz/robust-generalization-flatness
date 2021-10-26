import torch
import numpy
from .attack import *
import common.torch


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

        self.weight_norm = None
        """ (attacks.weights.norms.Norm) Weight norm. """

        self.weight_initialization = None
        """ (attacks.weights.initializations.Initialization) Weight norm. """

        self.weight_projection = None
        """ (attacks.weights.projections.Projection) Weight norm. """

        self.weight_normalization = None
        """ (attacks.weights.normalizations.Normalization) Normalization. """

        self.input_norm = None
        """ (attacks.norms.Norm) Weight norm. """

        self.input_initialization = None
        """ (attacks.initializations.Initialization) Weight norm. """

        self.input_projection = None
        """ (attacks.projections.Projection) Weight norm. """

        self.input_normalized = False
        """ (bool) Input normalized. """

        self.weight_lr = None
        """ (float) Base learning rate. """

        self.input_lr = None
        """ (float) Base learning rate. """

        self.perturbations = None
        """ ([torch.Tensor]) Example perturbations. """

        self.errors = []
        """ ([torch.Tensor]) Errors. """

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

        super(GradientDescentAttack, self).run(model, testset, weight_objective, input_objective)

        assert self.epochs > 0
        assert self.weight_lr is not None
        assert self.input_lr is not None
        is_cuda = common.torch.is_cuda(model)
        assert model.training is False

        perturbed_model = common.torch.clone(model)
        # Quantization contexts enforces same quantization throughout the attack;
        # if no quantization is used, it's just None.
        forward_model, _ = self.quantize(model)  # avoid cloning models all over

        # initialize weight perturbation
        if self.weight_initialization is not None:
            self.weight_initialization(model, perturbed_model, self.layers)
        self.quantize(perturbed_model, forward_model)

        # initialize input perturbations
        self.perturbations = []
        self.errors = []
        for b, (inputs, targets) in enumerate(testset):
            perturbations = torch.from_numpy(numpy.zeros(inputs.size()))
            if is_cuda:
                perturbations = perturbations.cuda()
            perturbations = perturbations.permute(0, 3, 1, 2)
            self.input_initialization(inputs, perturbations)
            self.perturbations.append(perturbations.cpu())
            self.errors.append(torch.from_numpy(numpy.zeros(perturbations.size(0))))

        for e in range(self.epochs):
            for b, (inputs, targets) in enumerate(testset):
                inputs = common.torch.as_variable(inputs, is_cuda)
                inputs = inputs.permute(0, 3, 1, 2)
                targets = common.torch.as_variable(targets, is_cuda)
                perturbations = common.torch.as_variable(self.perturbations[b], is_cuda)

                if perturbations.grad is not None:
                    perturbations.grad.data.zero_()
                perturbations.requires_grad = True
                inputs.requires_grad = False

                forward_parameters = list(forward_model.parameters())
                for i in self.layers:
                    if forward_parameters[i].grad is not None:
                        forward_parameters[i].grad.data.zero_()

                output_logits = forward_model(inputs + perturbations)
                input_objective.set(targets)
                errors = input_objective(output_logits)

                error = torch.mean(errors)
                error.backward()
                error = error.item()

                self.errors[b] = errors.cpu().detach().numpy()

                success = torch.sum(input_objective.success(output_logits)).item()
                true_confidence = torch.mean(input_objective.true_confidence(output_logits)).item()
                target_confidence = torch.mean(input_objective.target_confidence(output_logits)).item()

                weight_norm = self.weight_norm(model, forward_model, self.layers)
                input_norm = torch.mean(self.input_norm(perturbations)).item()

                global_step = e*len(testset) + b
                if self.writer is not None:
                    self.writer.add_scalar('%ssuccess' % self.prefix, success, global_step=global_step)
                    self.writer.add_scalar('%serror' % self.prefix, error, global_step=global_step)
                    self.writer.add_scalar('%sweight_norm' % self.prefix, weight_norm, global_step=global_step)
                    self.writer.add_scalar('%sinput_norm' % self.prefix, input_norm, global_step=global_step)
                    self.writer.add_scalar('%strue_confidence' % self.prefix, true_confidence, global_step=global_step)
                    self.writer.add_scalar('%starget_confidence' % self.prefix, target_confidence, global_step=global_step)
                    self.writer.add_scalar('%sinput_lr' % self.prefix, self.input_lr, global_step=global_step)
                    self.writer.add_scalar('%sweight_lr' % self.prefix, self.weight_lr, global_step=global_step)

                if self.progress is not None:
                    self.progress('attack weights', e * len(testset) + b, self.epochs*len(testset), info='success=%g error=%.2f inorm=%g wnorm=%g wlr=%g ilr=%g' % (
                        success,
                        error,
                        input_norm,
                        weight_norm,
                        self.weight_lr,
                        self.input_lr,
                    ), width=10)

                if self.weight_normalization is not None:
                    self.weight_normalization(model, forward_model, self.layers)
                if self.input_normalized:
                    self.input_norm.normalize(perturbations.grad)

                forward_parameters = list(forward_model.parameters())
                perturbed_parameters = list(perturbed_model.parameters())

                for i in self.layers:
                    perturbed_parameters[i].data -= self.weight_lr * forward_parameters[i].grad.data

                perturbations.data -= self.input_lr*perturbations.grad.data

                # model projection
                if self.weight_projection is not None:
                    self.weight_projection(model, perturbed_model, self.layers)
                self.quantize(perturbed_model, forward_model)

                # input projection
                if self.input_projection is not None:
                    self.input_projection(inputs, perturbations)

                self.perturbations[b] = perturbations.data.cpu().detach().numpy()

        self.quantize(perturbed_model, forward_model)
        if self.training:
            forward_model = forward_model.cuda()
        else:
            forward_model = forward_model.cpu()

        perturbations = numpy.concatenate(tuple(self.perturbations), axis=0)
        perturbations = numpy.transpose(perturbations, (0, 2, 3, 1))
        errors = numpy.concatenate(tuple(self.errors), axis=0)
        return forward_model, perturbations, errors

