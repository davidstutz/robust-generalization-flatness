import math
import torch
import numpy
import common.torch
import common.summary
import common.numpy
import attacks
from imgaug import augmenters as iaa
from .adversarial_training import AdversarialTraining


#def helper():
#    def feval():
#        x, y = next(train_loader)
#        if opt['cuda']:
#            x, y = x.cuda(), y.cuda()
#
#        x, y = Variable(x), Variable(y.squeeze())
#        bsz = x.size(0)
#
#        optimizer.zero_grad()
#        yh = model(x)
#        f = criterion.forward(yh, y)
#        f.backward()
#
#        prec1, = accuracy(yh.data, y.data, topk=(1,))
#        err = 100. - prec1[0]
#        return (f.data[0], err)
#
#    return feval
#
#f, err = optimizer.step(helper(), model, criterion)
class EntropyAdversarialTraining(AdversarialTraining):
    def train(self, epoch):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """

        assert math.isclose(self.fraction, 0), self.fraction
        assert self.summary_histograms is False
        assert self.summary_images is False
        assert self.summary_weights is False
        assert self.ignore_incorrect is False
        assert self.prevent_label_leaking is False

        self.update_average()
        criterion = torch.nn.CrossEntropyLoss()
        self.trainloader = iter(self.trainset)

        for b in range(len(self.trainset)):
            def helper():
                def feval():
                    try:
                        inputs, targets = next(self.trainloader)
                    except StopIteration as e:
                        self.trainloader = iter(self.trainset)
                        inputs, targets = next(self.trainloader)

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

                    logits = self.model(adversarial_inputs)

                    loss = criterion(logits, targets)
                    error = common.torch.classification_error(logits, targets)

                    loss.backward()
                    return (loss.item(), error.item())

                return feval

            loss, error = self.optimizer.step(helper(), self.model, criterion)
            self.scheduler.step()

            global_step = epoch * len(self.trainset) + b
            self.writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=global_step)

            self.writer.add_scalar('train/adversarial_loss', loss, global_step=global_step)
            self.writer.add_scalar('train/adversarial_error', error, global_step=global_step)

            self.progress('train %d' % epoch, b, len(self.trainset), info='advloss=%g adverr=%g lr=%g' % (
                loss,
                error,
                self.scheduler.get_lr()[0],
            ))

            self.update_average()