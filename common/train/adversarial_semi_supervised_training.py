import torch
import common.torch
import common.summary
import common.numpy
from imgaug import augmenters as iaa
from .adversarial_training import AdversarialTraining


class AdversarialSemiSupervisedTraining(AdversarialTraining):
    """
    Normal training.
    """

    def __init__(self, model, trainset, testset, optimizer, scheduler, attack, objective, fraction=0.5, unsup_weight=1, augmentation=None, loss=common.torch.classification_loss, unsup_loss=common.torch.classification_loss, writer=common.summary.SummaryWriter(), cuda=False):
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

        super(AdversarialSemiSupervisedTraining, self).__init__(model, trainset, testset, optimizer, scheduler, attack, objective, fraction, augmentation, loss, writer, cuda)

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

            fraction = self.fraction
            split = int(fraction * inputs.size(0))
            # update fraction for correct loss computation

            clean_inputs = inputs[:split]
            adversarial_inputs = inputs[split:]
            clean_targets = targets[:split]
            adversarial_targets = targets[split:]
            clean_supervised = inputs[:split]
            adversarial_supervised = supervised[split:]

            supervised_adversarial_inputs = adversarial_inputs[adversarial_supervised == 1]
            unsupervised_adversarial_inputs = adversarial_inputs[adversarial_supervised == 0]

            supervised_adversarial_targets = adversarial_targets[adversarial_supervised == 1]
            unsupervised_adversarial_targets = adversarial_targets[adversarial_supervised == 0]

            self.project()
            forward_model, _ = self.quantize()
            self.optimizer.zero_grad()

            self.model.eval()
            forward_model.eval()

            self.objective.set(supervised_adversarial_targets)
            supervised_adversarial_perturbations, supervised_adversarial_errors = self.attack.run(forward_model, supervised_adversarial_inputs, self.objective)
            supervised_adversarial_perturbations = common.torch.as_variable(supervised_adversarial_perturbations, self.cuda)
            supervised_adversarial_perturbations = supervised_adversarial_inputs + supervised_adversarial_perturbations

            self.objective.set(unsupervised_adversarial_targets)
            self.attack.auxiliary = True
            unsupervised_adversarial_perturbations, unsupervised_adversarial_errors = self.attack.run(forward_model, unsupervised_adversarial_inputs, self.objective)
            unsupervised_adversarial_perturbations = common.torch.as_variable(unsupervised_adversarial_perturbations, self.cuda)
            unsupervised_adversarial_perturbations = unsupervised_adversarial_inputs + unsupervised_adversarial_perturbations
            self.attack.auxiliary = False

            adversarial_inputs[adversarial_supervised == 1] = supervised_adversarial_perturbations
            adversarial_inputs[adversarial_supervised == 0] = unsupervised_adversarial_perturbations

            if adversarial_inputs.shape[0] < inputs.shape[0]:  # fraction is not 1
                inputs = torch.cat((clean_inputs, adversarial_inputs), dim=0)
            else:
                inputs = adversarial_inputs
                # targets remain unchanged

            #
            self.model.train()
            forward_model.train()
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
                        backward_buffers[key].data = (1 - momentum) * forward_buffers[key].data + momentum * \
                                                     backward_buffers[key].data
                    if key.find('num_batches_tracked') >= 0:
                        backward_buffers[key].data += 1  # forward_buffers[key].data

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
