import math
import torch
import numpy
import common.torch
import common.summary
import common.numpy
import attacks
from imgaug import augmenters as iaa
from .adversarial_training import AdversarialTraining


class TRADESAdversarialTraining(AdversarialTraining):
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

        super(TRADESAdversarialTraining, self).__init__(model, trainset, testset, optimizer, scheduler, attack, objective, fraction=fraction, augmentation=augmentation, loss=loss, writer=writer, cuda=cuda)
        self.fraction = 1 - self.fraction

    def trades_loss(self,
                    x_natural,
                    x_adv,
                    y,
                    beta=1.0):
        # define KL-loss
        criterion_kl = torch.nn.KLDivLoss(size_average=False)
        batch_size = len(x_natural)

        # calculate robust loss
        logits = self.model(x_natural)
        adv_logits = self.model(x_adv)
        loss_natural = torch.nn.functional.cross_entropy(logits, y)
        error_natural = common.torch.classification_error(logits, y)
        error_robust = common.torch.classification_error(adv_logits, y)
        loss_robust = (1.0 / batch_size) * criterion_kl(torch.nn.functional.log_softmax(adv_logits, dim=1), torch.nn.functional.softmax(logits, dim=1))
        loss_robust_ce = common.torch.classification_loss(adv_logits, y)
        loss = loss_natural + beta * loss_robust
        return loss, loss_natural.item(), loss_robust.item(), loss_robust_ce.item(), error_natural.item(), error_robust.item()

    def train(self, epoch):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """

        assert isinstance(self.objective, attacks.objectives.UntargetedKLObjective)
        assert self.summary_histograms is False
        assert self.summary_images is False
        assert self.summary_weights is False
        assert self.ignore_incorrect is False
        assert self.prevent_label_leaking is False

        self.update_average()
        self.trainloader = iter(self.trainset)

        for b, (inputs, targets) in enumerate(self.trainset):
            if self.augmentation is not None:
                if isinstance(self.augmentation, iaa.meta.Augmenter):
                    inputs = self.augmentation.augment_images(inputs.numpy())
                else:
                    inputs = self.augmentation(inputs)

            inputs = common.torch.as_variable(inputs, self.cuda)
            inputs = inputs.permute(0, 3, 1, 2)
            targets = common.torch.as_variable(targets, self.cuda)

            self.model.eval()
            with torch.no_grad():
                clean_logits = self.model(inputs)
            self.objective.set(targets, None, other=clean_logits)
            adversarial_perturbations, adversarial_objectives = self.attack.run(self.model, inputs, self.objective)
            adversarial_perturbations = common.torch.as_variable(adversarial_perturbations, self.cuda)
            adversarial_inputs = inputs + adversarial_perturbations

            self.model.train()
            assert self.model.training is True
            self.optimizer.zero_grad()

            beta = 1./self.fraction
            loss, loss_natural, loss_robust, loss_robust_ce, error_natural, error_robust = self.trades_loss(inputs, adversarial_inputs, targets, beta=beta)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            loss = loss.item()
            global_step = epoch * len(self.trainset) + b
            self.writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=global_step)

            self.writer.add_scalar('train/adversarial_loss', loss, global_step=global_step)
            self.writer.add_scalar('train/loss', loss_natural, global_step=global_step)
            self.writer.add_scalar('train/adversarial_loss_robust', loss_robust, global_step=global_step)
            self.writer.add_scalar('train/adversarial_loss_ce', loss_robust_ce, global_step=global_step)
            self.writer.add_scalar('train/error', error_natural, global_step=global_step)
            self.writer.add_scalar('train/adversarial_error', error_robust, global_step=global_step)

            self.progress('train %d' % epoch, b, len(self.trainset), info='loss=%g natloss=%g robloss=%g err=%g adverr=%g beta=%g lr=%g' % (
                loss,
                loss_natural,
                loss_robust,
                error_natural,
                error_robust,
                beta,
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

            with torch.no_grad():
                clean_logits = forward_model(inputs)
            b_clean_losses = self.loss(clean_logits, targets, reduction='none')
            b_clean_errors = common.torch.classification_error(clean_logits, targets, reduction='none')

            clean_losses = common.numpy.concatenate(clean_losses, b_clean_losses.detach().cpu().numpy())
            clean_errors = common.numpy.concatenate(clean_errors, b_clean_errors.detach().cpu().numpy())
            clean_confidences = common.numpy.concatenate(clean_confidences, torch.max(common.torch.softmax(clean_logits, dim=1), dim=1)[0].detach().cpu().numpy())

            self.objective.set(targets, None, other=clean_logits)
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

                with torch.no_grad():
                    clean_logits = self.model(inputs)
                self.objective.set(targets, None, other=clean_logits)
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