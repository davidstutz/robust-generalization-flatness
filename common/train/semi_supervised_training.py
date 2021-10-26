import torch
import common.torch
import common.summary
import common.numpy
from imgaug import augmenters as iaa
from .normal_training import NormalTraining


class SemiSupervisedTraining(NormalTraining):
    """
    Normal training.
    """

    def __init__(self, model, trainset, testset, optimizer, scheduler, unsup_weight=1, augmentation=None, loss=common.torch.classification_loss, unsup_loss=common.torch.classification_loss, writer=common.summary.SummaryWriter(), cuda=False):
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
        :param augmentation: augmentation
        :type augmentation: imgaug.augmenters.Sequential
        :param writer: summary writer
        :type writer: torch.utils.tensorboard.SummaryWriter or TensorboardX equivalent
        :param cuda: run on CUDA device
        :type cuda: bool
        """

        super(SemiSupervisedTraining, self).__init__(model, trainset, testset, optimizer, scheduler, augmentation, loss, writer, cuda)

        self.unsup_weight = unsup_weight
        """ (float) Unsupervised weight. """

        self.unsup_loss = unsup_loss
        """ (callable) Unsupervised loss. """

        self.writer.add_text('config/unsup_weight', str(self.unsup_weight))
        self.writer.add_text('config/unsup_loss', getattr(self.unsup_loss, '__name__', repr(self.unsup_loss)))

    def train(self, epoch):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """

        self.model.train()
        assert self.model.training is True

        self.quantize()

        for b, (inputs, targets, supervised) in enumerate(self.trainset):
            if self.augmentation is not None:
                if isinstance(self.augmentation, iaa.meta.Augmenter):
                    inputs = self.augmentation.augment_images(inputs.numpy())
                else:
                    inputs = self.augmentation(inputs)

            inputs = common.torch.as_variable(inputs, self.cuda)
            inputs = inputs.permute(0, 3, 1, 2)
            targets = common.torch.as_variable(targets, self.cuda)
            supervised = common.torch.as_variable(supervised, self.cuda)

            self.project()
            forward_model, _ = self.quantize()
            assert forward_model.training is True

            self.optimizer.zero_grad()
            logits, auxiliary_logits = forward_model(inputs, auxiliary=True)

            supervised_logits = logits[supervised == 1]
            unsupervised_logits = auxiliary_logits[supervised == 0]

            supervised_targets = targets[supervised == 1]
            unsupervised_targets = targets[supervised == 0]

            supervised_loss = self.loss(supervised_logits, supervised_targets)
            supervised_error = common.torch.classification_error(supervised_logits, supervised_targets)
            unsupervised_loss = self.unsup_loss(unsupervised_logits, unsupervised_targets)

            loss = supervised_loss + self.unsup_weight*unsupervised_loss
            loss.backward()

            if forward_model is not self.model:
                forward_parameters = list(forward_model.parameters())
                backward_parameters = list(self.model.parameters())

                for i in range(len(forward_parameters)):
                    if backward_parameters[i].requires_grad is False:  # normalization
                        continue

                    backward_parameters[i].grad = forward_parameters[i].grad

                forward_buffers = dict(forward_model.named_buffers())
                backward_buffers = dict(self.model.named_buffers())
                for key in forward_buffers.keys():
                    if key.find('running_var') >= 0 or key.find('running_mean') >= 0 or key.find('num_batches_tracked') >= 0:
                        backward_buffers[key].data = forward_buffers[key].data

            self.optimizer.step()
            self.scheduler.step()

            global_step = epoch * len(self.trainset) + b
            self.writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=global_step)
            self.writer.add_scalar('train/loss', supervised_loss.item(), global_step=global_step)
            self.writer.add_scalar('train/error', supervised_error.item(), global_step=global_step)
            self.writer.add_scalar('train/unsupervised_loss', unsupervised_loss.item(), global_step=global_step)
            self.writer.add_scalar('train/confidence', torch.mean(torch.max(common.torch.softmax(logits, dim=1), dim=1)[0]).item(), global_step=global_step)

            if self.summary_histograms:
                self.writer.add_histogram('train/logits', torch.max(logits, dim=1)[0], global_step=global_step)
                self.writer.add_histogram('train/confidences', torch.max(common.torch.softmax(logits, dim=1), dim=1)[0], global_step=global_step)
            if self.summary_weights:
                self.writer_add_model(global_step)
            if self.summary_images:
                self.writer.add_images('train/images', inputs[:16], global_step=global_step)

            self.progress('train %d' % epoch, b, len(self.trainset), info='sup_loss=%g sup_error=%g unsup_loss=%g lr=%g' % (
                supervised_loss.item(),
                supervised_error.item(),
                unsupervised_loss.item(),
                self.scheduler.get_lr()[0],
            ))