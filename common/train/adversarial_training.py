import torch
import numpy
import common.torch
import common.summary
import common.numpy
import attacks
from imgaug import augmenters as iaa
from .normal_training import NormalTraining


class AdversarialTraining(NormalTraining):
    """
    Adversarial training.
    """

    def __init__(self, model, trainset, testset, optimizer, scheduler, attack, objective, fraction=0.5, augmentation=None, loss=common.torch.classification_loss, writer=common.summary.SummaryWriter(), cuda=False):
        """
        Constructor.

        :param model: model
        :type model: torch.nn.Module
        :param trainset: training set
        :type trainset: torch.utils.data.DataLoader
        :param testset: test set
        :type testset: torch.utils.data.DataLoader
        :param optimizer: optimizer
        :type optimizer: torch.optim.Optimizer
        :param scheduler: scheduler
        :type scheduler: torch.optim.LRScheduler
        :param attack: attack
        :type attack: attacks.Attack
        :param objective: objective
        :type objective: attacks.Objective
        :param fraction: fraction of adversarial examples per batch
        :type fraction: float
        :param augmentation: augmentation
        :type augmentation: imgaug.augmenters.Sequential
        :param writer: summary writer
        :type writer: torch.utils.tensorboard.SummaryWriter or TensorboardX equivalent
        :param cuda: run on CUDA device
        :type cuda: bool
        """

        assert fraction > 0
        assert fraction <= 1
        assert isinstance(attack, attacks.Attack)
        assert isinstance(objective, attacks.objectives.Objective)
        assert getattr(attack, 'norm', None) is not None

        super(AdversarialTraining, self).__init__(model, trainset, testset, optimizer, scheduler, augmentation, loss, writer, cuda)

        self.attack = attack
        """ (attacks.Attack) Attack. """

        self.objective = objective
        """ (attacks.Objective) Objective. """

        # in code, we want fraction to be the fraction of clean samples for simplicity
        self.fraction = 1 - fraction
        """ (float) Fraction of adversarial examples ."""

        self.max_batches = 10
        """ (int) Number of batches to test adversarially on. """

        self.prevent_label_leaking = False
        """ (bool) Whether to use pseudo labels for the attack. """

        self.ignore_incorrect = False
        """ (bool) Ignore incorrect classified in adversarial example computation. """

        self.writer.add_text('config/attack', self.attack.__class__.__name__)
        self.writer.add_text('config/objective', self.objective.__class__.__name__)
        self.writer.add_text('config/fraction', str(fraction))
        self.writer.add_text('attack', str(common.summary.to_dict(self.attack)))
        if getattr(attack, 'initialization', None) is not None:
            self.writer.add_text('attack/initialization', str(common.summary.to_dict(self.attack.initialization)))
            if getattr(self.attack.initialization, 'initializations', None) is not None:
                for i in range(len(self.attack.initialization.initializations)):
                    self.writer.add_text('attack/initialization_%d' % i, str(common.summary.to_dict(self.attack.initialization.initializations[i])))
        if getattr(attack, 'projection', None) is not None:
            self.writer.add_text('attack/projection', str(common.summary.to_dict(self.attack.projection)))
            if getattr(self.attack.projection, 'projections', None) is not None:
                for i in range(len(self.attack.projection.projections)):
                    self.writer.add_text('attack/projection_%d' % i, str(common.summary.to_dict(self.attack.projection.projections[i])))
        if getattr(attack, 'norm', None) is not None:
            self.writer.add_text('attack/norm', str(common.summary.to_dict(self.attack.norm)))
        self.writer.add_text('objective', str(common.summary.to_dict(self.objective)))
        self.writer.add_text('config/prevent_label_leaking', str(self.prevent_label_leaking))

    def train(self, epoch):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """

        self.quantize()
        self.update_average()

        for b, (inputs, targets) in enumerate(self.trainset):
            if self.augmentation is not None:
                if isinstance(self.augmentation, iaa.meta.Augmenter):
                    inputs = self.augmentation.augment_images(inputs.numpy())
                else:
                    inputs = self.augmentation(inputs)

            inputs = common.torch.as_variable(inputs, self.cuda)
            inputs = inputs.permute(0, 3, 1, 2)
            targets = common.torch.as_variable(targets, self.cuda)

            fraction = self.fraction
            split = int(fraction*inputs.size(0))
            # update fraction for correct loss computation
            fraction = split / float(inputs.size(0))

            clean_inputs = inputs[:split]
            adversarial_inputs = inputs[split:]
            clean_targets = targets[:split]
            adversarial_targets = targets[split:]

            if self.prevent_label_leaking:
                # should still be in training mode
                logits_for_attack = self.model(adversarial_inputs)
                adversarial_targets_for_attack = torch.argmax(logits_for_attack, dim=1)
            else:
                adversarial_targets_for_attack = adversarial_targets

            self.model.eval()
            self.objective.set(adversarial_targets_for_attack)
            adversarial_perturbations, adversarial_objectives = self.attack.run(self.model, adversarial_inputs, self.objective)
            adversarial_perturbations = common.torch.as_variable(adversarial_perturbations, self.cuda)
            adversarial_inputs = adversarial_inputs + adversarial_perturbations

            if adversarial_inputs.shape[0] < inputs.shape[0]: # fraction is not 1
                combined_inputs = torch.cat((clean_inputs, adversarial_inputs), dim=0)
            else:
                combined_inputs = adversarial_inputs
                # targets remain unchanged

            #
            self.project()
            self.model.train()
            forward_model, _ = self.quantize()
            assert forward_model.training is True

            self.optimizer.zero_grad()

            if self.ignore_incorrect:
                logits = forward_model(inputs)
                errors = common.torch.classification_error(logits, targets, reduction='none')

                # use errors to remove adversarial examples for incorrect predictions
                errors = (errors == 1) # bool conversion
                combined_inputs[errors] = inputs[errors]

            logits = forward_model(combined_inputs)
            clean_logits = logits[:split]
            adversarial_logits = logits[split:]

            adversarial_loss = self.loss(adversarial_logits, adversarial_targets)
            adversarial_error = common.torch.classification_error(adversarial_logits, adversarial_targets)

            if adversarial_inputs.shape[0] < combined_inputs.shape[0]:
                clean_loss = self.loss(clean_logits, clean_targets)
                clean_error = common.torch.classification_error(clean_logits, clean_targets)
                loss = (1 - fraction) * clean_loss + fraction * adversarial_loss
            else:
                clean_loss = torch.zeros(1)
                clean_error = torch.zeros(1)
                loss = adversarial_loss

            loss.backward()

            if forward_model is not self.model:
                forward_parameters = list(forward_model.parameters())
                backward_parameters = list(self.model.parameters())

                for i in range(len(forward_parameters)):
                    if backward_parameters[i].requires_grad is False:  # normalization
                        continue
                    backward_parameters[i].grad = forward_parameters[i].grad

                # take care of BN statistics
                forward_buffers = dict(forward_model.named_buffers())
                backward_buffers = dict(self.model.named_buffers())
                for key in forward_buffers.keys():
                    layer_name = key.split('.')[-2]
                    layer = getattr(self.model, layer_name, None)
                    momentum = 0.1

                    if layer is not None:
                        momentum = layer.momentum
                    if key.find('running_var') >= 0 or key.find('running_mean') >= 0:
                        backward_buffers[key].data = (1 - momentum) * forward_buffers[key].data + momentum * backward_buffers[key].data
                    if key.find('num_batches_tracked') >= 0:
                        backward_buffers[key].data += 1  # forward_buffers[key].data

            self.optimizer.step()
            self.scheduler.step()

            global_step = epoch * len(self.trainset) + b
            self.writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=global_step)

            if adversarial_inputs.shape[0] < combined_inputs.shape[0]: # fraction is not 1
                self.writer.add_scalar('train/loss', clean_loss.item(), global_step=global_step)
                self.writer.add_scalar('train/error', clean_error.item(), global_step=global_step)
                self.writer.add_scalar('train/confidence', torch.mean(torch.max(common.torch.softmax(clean_logits, dim=1), dim=1)[0]).item(), global_step=global_step)

                if self.summary_histograms:
                    self.writer.add_histogram('train/logits', torch.max(clean_logits, dim=1)[0], global_step=global_step)
                    self.writer.add_histogram('train/confidences', torch.max(common.torch.softmax(clean_logits, dim=1), dim=1)[0], global_step=global_step)

            success = torch.clamp(torch.abs(adversarial_targets - torch.max(common.torch.softmax(adversarial_logits, dim=1), dim=1)[1]), max=1)
            self.writer.add_scalar('train/adversarial_loss', adversarial_loss.item(), global_step=global_step)
            self.writer.add_scalar('train/adversarial_error', adversarial_error.item(), global_step=global_step)
            self.writer.add_scalar('train/adversarial_confidence', torch.mean(torch.max(common.torch.softmax(adversarial_logits, dim=1), dim=1)[0]).item(), global_step=global_step)
            self.writer.add_scalar('train/adversarial_success', torch.mean(success.float()).item(), global_step=global_step)

            if self.summary_histograms:
                self.writer.add_histogram('train/adversarial_logits', torch.max(adversarial_logits, dim=1)[0], global_step=global_step)
                self.writer.add_histogram('train/adversarial_confidences', torch.max(common.torch.softmax(adversarial_logits, dim=1), dim=1)[0], global_step=global_step)

            if self.summary_histograms:
                adversarial_norms = self.attack.norm(adversarial_perturbations)
                self.writer.add_histogram('train/adversarial_objectives', adversarial_objectives, global_step=global_step)
                self.writer.add_histogram('train/adversarial_norms', adversarial_norms, global_step=global_step)
            if self.summary_weights:
                self.writer_add_model(global_step)
            if self.summary_images:
                if adversarial_inputs.shape[0] < combined_inputs.shape[0]: # fraction is not 1
                    self.writer.add_images('train/images', combined_inputs[:min(16, split)], global_step=global_step)
                self.writer.add_images('train/adversarial_images', combined_inputs[split:split + 16], global_step=global_step)

            self.progress('train %d' % epoch, b, len(self.trainset), info='loss=%g err=%g advloss=%g adverr=%g lr=%g' % (
                clean_loss.item(),
                clean_error.item(),
                adversarial_loss.item(),
                adversarial_error.item(),
                self.scheduler.get_lr()[0],
            ))

            self.update_average()

    def test(self, epoch):
        """
        Test on adversarial examples.

        :param epoch: epoch
        :type epoch: int
        """

        probabilities, forward_model = super(AdversarialTraining, self).test(epoch)

        self.model.eval()

        clean_losses = None
        clean_errors = None
        clean_confidences = None

        losses = None
        errors = None
        confidences = None
        successes = None
        norms = None
        objectives = None

        for b, (inputs, targets) in enumerate(self.testset):
            if b >= self.max_batches:
                break

            inputs = common.torch.as_variable(inputs, self.cuda)
            inputs = inputs.permute(0, 3, 1, 2)
            targets = common.torch.as_variable(targets, self.cuda)

            clean_logits = forward_model(inputs)
            b_clean_losses = self.loss(clean_logits, targets, reduction='none')
            b_clean_errors = common.torch.classification_error(clean_logits, targets, reduction='none')

            clean_losses = common.numpy.concatenate(clean_losses, b_clean_losses.detach().cpu().numpy())
            clean_errors = common.numpy.concatenate(clean_errors, b_clean_errors.detach().cpu().numpy())
            clean_confidences = common.numpy.concatenate(clean_confidences, torch.max(common.torch.softmax(clean_logits, dim=1), dim=1)[0].detach().cpu().numpy())

            self.objective.set(targets)
            adversarial_perturbations, adversarial_objectives = self.attack.run(self.model, inputs, self.objective)
            objectives = common.numpy.concatenate(objectives, adversarial_objectives)

            adversarial_perturbations = common.torch.as_variable(adversarial_perturbations, self.cuda)
            inputs = inputs + adversarial_perturbations

            with torch.no_grad():
                logits = forward_model(inputs)

                b_losses = self.loss(logits, targets, reduction='none')
                b_errors = common.torch.classification_error(logits, targets, reduction='none')

                losses = common.numpy.concatenate(losses, b_losses.detach().cpu().numpy())
                errors = common.numpy.concatenate(errors, b_errors.detach().cpu().numpy())
                confidences = common.numpy.concatenate(confidences, torch.max(common.torch.softmax(logits, dim=1), dim=1)[0].detach().cpu().numpy())
                successes = common.numpy.concatenate(successes, torch.clamp(torch.abs(targets - torch.max(common.torch.softmax(logits, dim=1), dim=1)[1]), max=1).detach().cpu().numpy())
                norms = common.numpy.concatenate(norms, self.attack.norm(adversarial_perturbations).detach().cpu().numpy())
                self.progress('test %d' % epoch, b, self.max_batches, info='loss=%g error=%g' % (
                    torch.mean(b_losses).item(),
                    torch.mean(b_errors.float()).item()
                ))

        global_step = epoch + 1# * len(self.trainset) + len(self.trainset) - 1
        self.writer.add_scalar('test/adversarial_loss', numpy.mean(losses), global_step=global_step)
        self.writer.add_scalar('test/adversarial_error', numpy.mean(errors), global_step=global_step)
        self.writer.add_scalar('test/adversarial_confidence', numpy.mean(confidences), global_step=global_step)
        self.writer.add_scalar('test/adversarial_success', numpy.mean(successes), global_step=global_step)
        self.writer.add_scalar('test/adversarial_norm', numpy.mean(norms), global_step=global_step)
        self.writer.add_scalar('test/adversarial_objective', numpy.mean(objectives), global_step=global_step)

        self.writer.add_scalar('test/adversarial_correct_loss', numpy.mean(losses[clean_errors == 0]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_correct_error', numpy.mean(errors[clean_errors == 0]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_correct_confidence', numpy.mean(confidences[clean_errors == 0]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_correct_success', numpy.mean(successes[clean_errors == 0]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_correct_norm', numpy.mean(norms[clean_errors == 0]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_correct_objective', numpy.mean(objectives[clean_errors == 0]), global_step=global_step)

        self.writer.add_scalar('test/adversarial_incorrect_loss', numpy.mean(losses[clean_errors == 1]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_incorrect_error', numpy.mean(errors[clean_errors == 1]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_incorrect_confidence', numpy.mean(confidences[clean_errors == 1]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_incorrect_success', numpy.mean(successes[clean_errors == 1]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_incorrect_norm', numpy.mean(norms[clean_errors == 1]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_incorrect_objective', numpy.mean(objectives[clean_errors == 1]), global_step=global_step)

        self.writer.add_scalar('test/adversarial_correct_robust_loss', numpy.mean(losses[numpy.logical_and(clean_errors == 0, errors == 0)]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_correct_robust_error', numpy.mean(errors[numpy.logical_and(clean_errors == 0, errors == 0)]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_correct_robust_confidence', numpy.mean(confidences[numpy.logical_and(clean_errors == 0, errors == 0)]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_correct_robust_success', numpy.mean(successes[numpy.logical_and(clean_errors == 0, errors == 0)]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_correct_robust_norm', numpy.mean(norms[numpy.logical_and(clean_errors == 0, errors == 0)]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_correct_robust_objective', numpy.mean(objectives[numpy.logical_and(clean_errors == 0, errors == 0)]), global_step=global_step)

        self.writer.add_scalar('test/adversarial_correct_inrobust_loss', numpy.mean(losses[numpy.logical_and(clean_errors == 0, errors == 1)]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_correct_inrobust_error', numpy.mean(errors[numpy.logical_and(clean_errors == 0, errors == 1)]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_correct_inrobust_confidence', numpy.mean(confidences[numpy.logical_and(clean_errors == 0, errors == 1)]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_correct_inrobust_success', numpy.mean(successes[numpy.logical_and(clean_errors == 0, errors == 1)]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_correct_inrobust_norm', numpy.mean(norms[numpy.logical_and(clean_errors == 0, errors == 1)]), global_step=global_step)
        self.writer.add_scalar('test/adversarial_correct_inrobust_objective', numpy.mean(objectives[numpy.logical_and(clean_errors == 0, errors == 1)]), global_step=global_step)

        if self.summary_histograms:
            self.writer.add_histogram('test/adversarial_losses', losses, global_step=global_step)
            self.writer.add_histogram('test/adversarial_errors', errors, global_step=global_step)
            self.writer.add_histogram('test/adversarial_confidences', confidences, global_step=global_step)
            self.writer.add_histogram('test/adversarial_norms', norms, global_step=global_step)
            self.writer.add_histogram('test/adversarial_objectives', objectives, global_step=global_step)

        if self.average is not None:
            assert self.average.training is False

            losses = None
            errors = None
            confidences = None
            successes = None
            norms = None
            objectives = None

            for b, (inputs, targets) in enumerate(self.testset):
                if b >= self.max_batches:
                    break

                inputs = common.torch.as_variable(inputs, self.cuda)
                inputs = inputs.permute(0, 3, 1, 2)
                targets = common.torch.as_variable(targets, self.cuda)

                self.objective.set(targets)
                adversarial_perturbations, adversarial_objectives = self.attack.run(self.average, inputs, self.objective)
                objectives = common.numpy.concatenate(objectives, adversarial_objectives)

                adversarial_perturbations = common.torch.as_variable(adversarial_perturbations, self.cuda)
                inputs = inputs + adversarial_perturbations

                with torch.no_grad():
                    logits = self.average(inputs)

                    b_losses = self.loss(logits, targets, reduction='none')
                    b_errors = common.torch.classification_error(logits, targets, reduction='none')

                    losses = common.numpy.concatenate(losses, b_losses.detach().cpu().numpy())
                    errors = common.numpy.concatenate(errors, b_errors.detach().cpu().numpy())
                    confidences = common.numpy.concatenate(confidences, torch.max(common.torch.softmax(logits, dim=1), dim=1)[0].detach().cpu().numpy())
                    successes = common.numpy.concatenate(successes, torch.clamp(torch.abs(targets - torch.max(common.torch.softmax(logits, dim=1), dim=1)[1]), max=1).detach().cpu().numpy())
                    norms = common.numpy.concatenate(norms, self.attack.norm(adversarial_perturbations).detach().cpu().numpy())
                    self.progress('test (average) %d' % epoch, b, self.max_batches, info='loss=%g error=%g' % (
                        torch.mean(b_losses).item(),
                        torch.mean(b_errors.float()).item()
                    ))

            global_step = epoch + 1  # * len(self.trainset) + len(self.trainset) - 1
            self.writer.add_scalar('test/average_adversarial_loss', numpy.mean(losses), global_step=global_step)
            self.writer.add_scalar('test/average_adversarial_error', numpy.mean(errors), global_step=global_step)
            self.writer.add_scalar('test/average_adversarial_confidence', numpy.mean(confidences), global_step=global_step)
            self.writer.add_scalar('test/average_adversarial_success', numpy.mean(successes), global_step=global_step)
            self.writer.add_scalar('test/average_adversarial_norm', numpy.mean(norms), global_step=global_step)
            self.writer.add_scalar('test/average_adversarial_objective', numpy.mean(objectives), global_step=global_step)

            if self.summary_histograms:
                self.writer.add_histogram('test/average_adversarial_losses', losses, global_step=global_step)
                self.writer.add_histogram('test/average_adversarial_errors', errors, global_step=global_step)
                self.writer.add_histogram('test/average_adversarial_confidences', confidences, global_step=global_step)
                self.writer.add_histogram('test/average_adversarial_norms', norms, global_step=global_step)
                self.writer.add_histogram('test/average_adversarial_objectives', objectives, global_step=global_step)

        return probabilities, forward_model