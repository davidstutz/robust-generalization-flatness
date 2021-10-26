import math
import torch
import numpy
import common.torch
import common.summary
import common.numpy
import attacks
from imgaug import augmenters as iaa
from .adversarial_training import AdversarialTraining


class MARTAdversarialTraining(AdversarialTraining):
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

        super(MARTAdversarialTraining, self).__init__(model, trainset, testset, optimizer, scheduler, attack, objective, fraction=fraction, augmentation=augmentation, loss=loss, writer=writer, cuda=cuda)
        self.fraction = 1 - self.fraction

    def mart_loss(self,
                  x_natural,
                  x_adv,
                  y,
                  beta): # = 6.0
        """ https://github.com/YisenWang/MART/blob/master/mart.py """

        batch_size = len(x_natural)
        kl = torch.nn.KLDivLoss(reduction='none')

        logits = self.model(x_natural)
        logits_adv = self.model(x_adv)

        error = common.torch.classification_error(logits, y)
        error_adv = common.torch.classification_error(logits_adv, y)

        adv_probs = torch.nn.functional.softmax(logits_adv, dim=1)

        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

        new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

        loss_adv = torch.nn.functional.cross_entropy(logits_adv, y) + torch.nn.functional.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

        nat_probs = torch.nn.functional.softmax(logits, dim=1)

        true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

        loss_robust = (1.0 / batch_size) * torch.sum(
            torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
        loss = loss_adv + float(beta) * loss_robust

        loss_adv_ce = common.torch.classification_loss(logits_adv, y)
        return loss, loss_adv.item(), loss_robust.item(), loss_adv_ce.item(), error.item(), error_adv.item()

    def train(self, epoch):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """

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
            self.objective.set(targets)
            adversarial_perturbations, adversarial_objectives = self.attack.run(self.model, inputs, self.objective)
            adversarial_perturbations = common.torch.as_variable(adversarial_perturbations, self.cuda)
            adversarial_inputs = inputs + adversarial_perturbations

            self.model.train()
            assert self.model.training is True
            self.optimizer.zero_grad()

            beta = 1./self.fraction
            loss, loss_adv, loss_robust, loss_adv_ce, error, error_adv = self.mart_loss(inputs, adversarial_inputs, targets, beta=beta)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            loss = loss.item()
            global_step = epoch * len(self.trainset) + b
            self.writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=global_step)

            self.writer.add_scalar('train/adversarial_loss', loss, global_step=global_step)
            self.writer.add_scalar('train/adversarial_loss_adv', loss_adv, global_step=global_step)
            self.writer.add_scalar('train/adversarial_loss_robust', loss_robust, global_step=global_step)
            self.writer.add_scalar('train/adversarial_loss_ce', loss_adv_ce, global_step=global_step)
            self.writer.add_scalar('train/error', error, global_step=global_step)
            self.writer.add_scalar('train/adversarial_error', error_adv, global_step=global_step)

            self.progress('train %d' % epoch, b, len(self.trainset), info='loss=%g advloss=%g robloss=%g err=%g adverr=%g beta=%g lr=%g' % (
                loss,
                loss_adv,
                loss_robust,
                error,
                error_adv,
                beta,
                self.scheduler.get_lr()[0],
            ))

            self.update_average()