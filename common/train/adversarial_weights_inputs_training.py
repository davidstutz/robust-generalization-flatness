import numpy
import torch
import common.torch
import common.summary
import common.numpy
import attacks.weights
from imgaug import augmenters as iaa
from .normal_training import NormalTraining


class AdversarialWeightsInputsTraining(NormalTraining):
    """
    Adversarial training.
    """

    def __init__(self, model, trainset, testset, optimizer, scheduler, weight_attack, weight_objective, input_attack, input_objective, operators=None, augmentation=None, loss=common.torch.classification_loss, writer=common.summary.SummaryWriter(), cuda=False):
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
        :param weight_attack: attack
        :type weight_attack: attacks.Attack
        :param weight_objective: objective
        :type weight_objective: attacks.Objective
        :param input_attack: attack
        :type input_attack: attacks.Attack
        :param input_objective: objective
        :type input_objective: attacks.Objective
        :param augmentation: augmentation
        :type augmentation: imgaug.augmenters.Sequential
        :param writer: summary writer
        :type writer: torch.utils.tensorboard.SummaryWriter or TensorboardX equivalent
        :param cuda: run on CUDA device
        :type cuda: bool
        """

        assert isinstance(weight_attack, attacks.weights.Attack)
        assert getattr(weight_attack, 'training', None) is not None
        assert isinstance(weight_objective, attacks.weights.objectives.Objective)
        assert getattr(weight_attack, 'norm', None) is not None

        assert isinstance(input_attack, attacks.Attack)
        assert isinstance(input_objective, attacks.objectives.Objective)
        assert getattr(input_attack, 'norm', None) is not None

        super(AdversarialWeightsInputsTraining, self).__init__(model, trainset, testset, optimizer, scheduler, augmentation, loss, writer, cuda)

        self.weight_attack = weight_attack
        """ (attacks.weights.Attack) Attack. """

        self.weight_objective = weight_objective
        """ (attacks.weights.Objective) Objective. """

        self.input_attack = input_attack
        """ (attacks.Attack) Attack. """

        self.input_objective = input_objective
        """ (attacks.Objective) Objective. """

        self.max_batches = 10
        """ (int) Number of batches to test adversarially on. """

        self.curriculum = None
        """ (None or callable) Curriculum for attack. """

        self.population = 0
        """ (int) Population. """

        self.gradient_clipping = 0.05
        """ (float) Clipping. """

        self.reset_iterations = 1
        """ (int) Reset objective iterations. """

        self.average_statistics = False
        """ (bool) Average bn statistics. """

        self.adversarial_statistics = False
        """ (bool) Adversarial bn statistics. """

        self.operators = operators
        """ ([attacks.activation.Attack]) Activation operators. """

        self.clean = False
        """ (bool) Adversarial weights on clean examples. """

        self.weight_attack.training = True
        self.writer.add_text('config/weight_attack', self.weight_attack.__class__.__name__)
        self.writer.add_text('config/weight_objective', self.weight_objective.__class__.__name__)
        self.writer.add_text('weight_attack', str(common.summary.to_dict(self.weight_attack)))
        if getattr(weight_attack, 'initialization', None) is not None:
            self.writer.add_text('aweight_ttack/initialization', str(common.summary.to_dict(self.weight_attack.initialization)))
            if getattr(self.weight_attack.initialization, 'initializations', None) is not None:
                for i in range(len(self.weight_attack.initialization.initializations)):
                    self.writer.add_text('weight_attack/initialization_%d' % i, str(common.summary.to_dict(self.weight_attack.initialization.initializations[i])))
        if getattr(weight_attack, 'projection', None) is not None:
            self.writer.add_text('weight_attack/projection', str(common.summary.to_dict(self.weight_attack.projection)))
            if getattr(self.weight_attack.projection, 'projections', None) is not None:
                for i in range(len(self.weight_attack.projection.projections)):
                    self.writer.add_text('attack/projection_%d' % i, str(common.summary.to_dict(self.weight_attack.projection.projections[i])))
        if getattr(weight_attack, 'norm', None) is not None:
            self.writer.add_text('weight_attack/norm', str(common.summary.to_dict(self.weight_attack.norm)))
        self.writer.add_text('weight_objective', str(common.summary.to_dict(self.weight_objective)))

        self.writer.add_text('config/input_attack', self.input_attack.__class__.__name__)
        self.writer.add_text('config/input_objective', self.input_objective.__class__.__name__)
        self.writer.add_text('input_attack', str(common.summary.to_dict(self.input_attack)))
        if getattr(input_attack, 'initialization', None) is not None:
            self.writer.add_text('input_attack/initialization', str(common.summary.to_dict(self.input_attack.initialization)))
            if getattr(self.input_attack.initialization, 'initializations', None) is not None:
                for i in range(len(self.input_attack.initialization.initializations)):
                    self.writer.add_text('input_attack/initialization_%d' % i, str(common.summary.to_dict(self.input_attack.initialization.initializations[i])))
        if getattr(input_attack, 'projection', None) is not None:
            self.writer.add_text('input_attack/projection', str(common.summary.to_dict(self.input_attack.projection)))
            if getattr(self.input_attack.projection, 'projections', None) is not None:
                for i in range(len(self.input_attack.projection.projections)):
                    self.writer.add_text('input_attack/projection_%d' % i, str(common.summary.to_dict(self.input_attack.projection.projections[i])))
        if getattr(input_attack, 'input_norm', None) is not None:
            self.writer.add_text('input_attack/norm', str(common.summary.to_dict(self.input_attack.norm)))
        self.writer.add_text('input_objective', str(common.summary.to_dict(self.input_objective)))

    def train(self, epoch):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """

        assert not self.clean
        assert self.average_statistics is False
        assert not (self.average_statistics and self.adversarial_statistics)

        self.quantize()

        for b, (inputs, targets) in enumerate(self.trainset):
            if self.augmentation is not None:
                if isinstance(self.augmentation, iaa.meta.Augmenter):
                    inputs = self.augmentation.augment_images(inputs.numpy())
                else:
                    inputs = self.augmentation(inputs)

            inputs = common.torch.as_variable(inputs, self.cuda)
            inputs = inputs.permute(0, 3, 1, 2)
            targets = common.torch.as_variable(targets, self.cuda)

            self.project()
            forward_model, _ = self.quantize()
            global_step = epoch * len(self.trainset) + b

            population_norm = 0
            population_perturbed_loss = 0
            population_perturbed_error = 0

            if b%self.reset_iterations == 0:
                self.weight_objective.reset()

            self.model.eval()
            forward_model.eval()
            self.input_objective.set(targets)
            adversarial_perturbations, adversarial_objectives = self.input_attack.run(forward_model, inputs, self.input_objective)
            adversarial_perturbations = common.torch.as_variable(adversarial_perturbations, self.cuda)
            adversarial_inputs = inputs + adversarial_perturbations
            adversarial_norms = self.input_attack.norm(adversarial_perturbations)

            # before permutation!
            # works with enumerate() similar to data loader.
            batchset = [(adversarial_inputs.permute(0, 2, 3, 1), targets)]

            mean_abs_grad = 0
            if self.population > 0:
                self.model.train()
                forward_model.train()

                self.optimizer.zero_grad()
                self.model.zero_grad()
                forward_model.zero_grad()

                logits = forward_model(adversarial_inputs)

                loss = self.loss(logits, targets)
                error = common.torch.classification_error(logits, targets)

                self.writer.add_scalar('train/pre_adversarial_loss', loss.item(), global_step=global_step)
                self.writer.add_scalar('train/pre_adversarial_error', error.item(), global_step=global_step)

                if not self.adversarial_statistics:
                    forward_buffers = dict(forward_model.named_buffers())
                    backward_buffers = dict(self.model.named_buffers())
                    for key in forward_buffers.keys():
                        if key.find('running_var') >= 0 or key.find('running_mean') >= 0 or key.find('num_batches_tracked') >= 0:
                            backward_buffers[key].data = forward_buffers[key].data

                self.model.eval()
                forward_model.eval()
                self.optimizer.zero_grad()
                self.model.zero_grad()
                forward_model.zero_grad()

                for i in range(self.population):
                    #self.weight_attack.progress = ProgressBar()
                    perturbed_model = self.weight_attack.run(forward_model, batchset, self.weight_objective)

                    # This is a perturbation based on the original eval model (self.model.eval)!
                    #perturbed_model = common.torch.clone(forward_model)
                    perturbed_model.train()
                    perturbed_model.zero_grad()
                    perturbed_logits = perturbed_model(adversarial_inputs)

                    perturbed_loss = self.loss(perturbed_logits, targets)
                    perturbed_error = common.torch.classification_error(perturbed_logits, targets)

                    perturbed_loss.backward()

                    population_perturbed_loss += perturbed_loss.item()
                    population_perturbed_error += perturbed_error.item()

                    # take average of gradients
                    parameters = list(self.model.parameters())
                    perturbed_parameters = list(perturbed_model.parameters())

                    norm_ = 0
                    perturbed_norm_ = 0
                    for j in range(len(parameters)):
                        norm_ = max(norm_, torch.abs(torch.max(parameters[j].data)).item())
                        perturbed_norm_ = max(perturbed_norm_, torch.abs(torch.max(parameters[j].data - perturbed_parameters[j].data)))

                        if parameters[j].requires_grad is False:  # normalization
                            continue

                        if parameters[j].grad is None:
                            parameters[j].grad = torch.clamp(perturbed_parameters[j].grad, min=-self.gradient_clipping, max=self.gradient_clipping)
                        else:
                            parameters[j].grad.data += torch.clamp(perturbed_parameters[j].grad.data, min=-self.gradient_clipping, max=self.gradient_clipping)

                    if self.adversarial_statistics:
                        perturbed_buffers = dict(perturbed_model.named_buffers())
                        backward_buffers = dict(self.model.named_buffers())
                        for key in perturbed_buffers.keys():
                            if key.find('running_var') >= 0 or key.find('running_mean') >= 0 or key.find('num_batches_tracked') >= 0:
                                if i == 0:
                                    backward_buffers[key].data.fill_(0)
                                backward_buffers[key].data += perturbed_buffers[key].data

                    if self.summary_weights:
                        # TODO
                        pass

                    self.writer.add_scalar('train/adversarial_loss%d' % i, perturbed_loss.item(), global_step=global_step)
                    self.writer.add_scalar('train/adversarial_error%d' % i, perturbed_error.item(), global_step=global_step)

                    if self.weight_attack.norm is not None:
                        norm = self.weight_attack.norm(forward_model, perturbed_model, self.weight_attack.layers)
                        population_norm += norm
                        self.writer.add_scalar('train/adversarial_norm%d' % i, norm, global_step=global_step)
                        for j in range(len(self.weight_attack.norm.norms)):
                            self.writer.add_scalar('train/adversarial_norms%d/%d' % (i, j), self.weight_attack.norm.norms[j], global_step=global_step)

                population_norm /= self.population
                population_perturbed_loss /= self.population
                population_perturbed_error /= self.population

                for parameter in self.model.parameters():
                    if parameter.requires_grad is False:  # normalization
                        continue
                    parameter.grad.data /= self.population
                    mean_abs_grad += torch.mean(torch.abs(parameter.grad.data))
                mean_abs_grad /= len(list(self.model.parameters()))

                if self.adversarial_statistics:
                    backward_buffers = dict(self.model.named_buffers())
                    for key in backward_buffers.keys():
                        if key.find('running_var') >= 0 or key.find('running_mean') >= 0 or key.find('num_batches_tracked') >= 0:
                            backward_buffers[key].data /= self.population

                self.model.train()
                forward_model.train()

                self.optimizer.step()
                self.scheduler.step()
            else:
                # basically normal training
                self.model.train()
                forward_model.train()
                self.optimizer.zero_grad()
                logits = forward_model(adversarial_inputs)
                loss = self.loss(logits, targets)
                error = common.torch.classification_error(logits, targets)
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
                        if key.find('running_var') >= 0 or key.find('running_mean') >= 0 or key.find('num_batches_tracked') >= 0:
                            backward_buffers[key].data = forward_buffers[key].data

                self.optimizer.step()
                self.scheduler.step()

            curriculum_logs = dict()
            if self.curriculum is not None:
                self.population, curriculum_logs = self.curriculum(self.weight_attack, loss, population_perturbed_loss, epoch)
                for curriculum_key, curriculum_value in curriculum_logs.items():
                    self.writer.add_scalar('train/curriculum/%s' % curriculum_key, curriculum_value, global_step=global_step)
            else:
                self.population = 1

            self.writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=global_step)

            self.writer.add_scalar('train/loss', loss.item(), global_step=global_step)
            self.writer.add_scalar('train/error', error.item(), global_step=global_step)
            self.writer.add_scalar('train/confidence', torch.mean(torch.max(common.torch.softmax(logits, dim=1), dim=1)[0]).item(), global_step=global_step)

            if self.summary_histograms:
                self.writer.add_histogram('train/logits', torch.max(logits, dim=1)[0], global_step=global_step)
                self.writer.add_histogram('train/confidences', torch.max(common.torch.softmax(logits, dim=1), dim=1)[0], global_step=global_step)
            if self.summary_weights:
                self.writer_add_model(global_step)
            if self.summary_images:
                self.writer.add_images('train/images', adversarial_inputs[:16], global_step=global_step)

            self.progress('train %d' % epoch, b, len(self.trainset), info='loss=%g err=%g advloss=%g adverr=%g advwnorm=%g advinorm=%g gradient=%g lr=%g pop=%d curr=%s' % (
                loss.item(),
                error.item(),
                population_perturbed_loss,
                population_perturbed_error,
                population_norm,
                torch.mean(adversarial_norms).item(),
                mean_abs_grad,
                self.scheduler.get_lr()[0],
                self.population,
                str(list(curriculum_logs.values())),
            ))

    def test(self, epoch):
        """
        Test on adversarial examples.

        :param epoch: epoch
        :type epoch: int
        """

        probabilities, forward_model = super(AdversarialWeightsInputsTraining, self).test(epoch)

        assert forward_model.training is False
        assert self.model.training is False

        clean_losses = None
        clean_errors = None
        clean_confidences = None

        losses = None
        errors = None
        logits = None
        confidences = None
        adversarial_losses = None
        adversarial_errors = None
        adversarial_logits = None
        adversarial_confidences = None
        norms = []

        if getattr(self.weight_attack, 'error_bound', None) is not None:
            error_bound = self.weight_attack.error_bound
            self.weight_attack.error_bound = -1e12

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

            self.input_objective.set(targets)
            adversarial_perturbations, adversarial_objectives = self.input_attack.run(forward_model, inputs, self.input_objective)
            adversarial_perturbations = common.torch.as_variable(adversarial_perturbations, self.cuda)
            adversarial_inputs = inputs + adversarial_perturbations

            outputs = forward_model(adversarial_inputs)
            b_losses = self.loss(outputs, targets, reduction='none')
            b_errors = common.torch.classification_error(outputs, targets, reduction='none')

            losses = common.numpy.concatenate(losses, b_losses.detach().cpu().numpy())
            errors = common.numpy.concatenate(errors, b_errors.detach().cpu().numpy())
            logits = common.numpy.concatenate(logits, torch.max(outputs, dim=1)[0].detach().cpu().numpy())
            confidences = common.numpy.concatenate(confidences, torch.max(common.torch.softmax(outputs, dim=1), dim=1)[0].detach().cpu().numpy())

            batchset = [(adversarial_inputs.permute(0, 2, 3, 1), targets)]
            self.weight_objective.reset()
            perturbed_model = self.weight_attack.run(forward_model, batchset, self.weight_objective)

            perturbed_model.eval()
            with torch.no_grad():
                adversarial_outputs = perturbed_model(adversarial_inputs)
                b_adversarial_losses = self.loss(adversarial_outputs, targets, reduction='none')
                b_adversarial_errors = common.torch.classification_error(adversarial_outputs, targets, reduction='none')
 
                adversarial_losses = common.numpy.concatenate(adversarial_losses, b_adversarial_losses.detach().cpu().numpy())
                adversarial_errors = common.numpy.concatenate(adversarial_errors, b_adversarial_errors.detach().cpu().numpy())
                adversarial_logits = common.numpy.concatenate(adversarial_logits, torch.max(adversarial_outputs, dim=1)[0].detach().cpu().numpy())
                adversarial_confidences = common.numpy.concatenate(adversarial_confidences, torch.max(common.torch.softmax(adversarial_outputs, dim=1), dim=1)[0].detach().cpu().numpy())
                if self.weight_attack.norm is not None:
                    norms.append(self.weight_attack.norm(forward_model, perturbed_model, self.weight_attack.layers))

                self.progress('test %d' % epoch, b, self.max_batches, info='loss=%g error=%g' % (
                    torch.mean(b_adversarial_losses).item(),
                    torch.mean(b_adversarial_errors.float()).item()
                ))

        global_step = epoch  # epoch * len(self.trainset) + len(self.trainset) - 1
        self.writer.add_scalar('test/pre_adversarial_loss', numpy.mean(losses), global_step=global_step)
        self.writer.add_scalar('test/pre_adversarial_error', numpy.mean(errors), global_step=global_step)
        self.writer.add_scalar('test/pre_adversarial_logit', numpy.mean(logits), global_step=global_step)
        self.writer.add_scalar('test/pre_adversarial_confidence', numpy.mean(confidences), global_step=global_step)

        self.writer.add_scalar('test/pre_adversarial_correct_loss', numpy.mean(losses[clean_errors == 0]), global_step=global_step)
        self.writer.add_scalar('test/pre_adversarial_correct_error', numpy.mean(errors[clean_errors == 0]), global_step=global_step)
        self.writer.add_scalar('test/pre_adversarial_correct_logit', numpy.mean(logits[clean_errors == 0]), global_step=global_step)
        self.writer.add_scalar('test/pre_adversarial_correct_confidence', numpy.mean(confidences[clean_errors == 0]), global_step=global_step)

        self.writer.add_scalar('test/pre_adversarial_incorrect_loss', numpy.mean(losses[clean_errors == 1]), global_step=global_step)
        self.writer.add_scalar('test/pre_adversarial_incorrect_error', numpy.mean(errors[clean_errors == 1]), global_step=global_step)
        self.writer.add_scalar('test/pre_adversarial_incorrect_logit', numpy.mean(logits[clean_errors == 1]), global_step=global_step)
        self.writer.add_scalar('test/pre_adversarial_incorrect_confidence', numpy.mean(confidences[clean_errors == 1]), global_step=global_step)

        self.writer.add_scalar('test/pre_adversarial_correct_robust_loss', numpy.mean(losses[numpy.logical_and(clean_errors == 0, errors == 0)]), global_step=global_step)
        self.writer.add_scalar('test/pre_adversarial_correct_robust_error', numpy.mean(errors[numpy.logical_and(clean_errors == 0, errors == 0)]), global_step=global_step)
        self.writer.add_scalar('test/pre_adversarial_correct_robust_logit', numpy.mean(logits[numpy.logical_and(clean_errors == 0, errors == 0)]), global_step=global_step)
        self.writer.add_scalar('test/pre_adversarial_correct_robust_confidence', numpy.mean(confidences[numpy.logical_and(clean_errors == 0, errors == 0)]), global_step=global_step)

        self.writer.add_scalar('test/pre_adversarial_correct_inrobust_loss', numpy.mean(losses[numpy.logical_and(clean_errors == 0, errors == 1)]), global_step=global_step)
        self.writer.add_scalar('test/pre_adversarial_ecorrect_inrobust_rror', numpy.mean(errors[numpy.logical_and(clean_errors == 0, errors == 1)]), global_step=global_step)
        self.writer.add_scalar('test/pre_adversarial_correct_inrobust_logit', numpy.mean(logits[numpy.logical_and(clean_errors == 0, errors == 1)]), global_step=global_step)
        self.writer.add_scalar('test/pre_adversarial_correct_inrobust_confidence', numpy.mean(confidences[numpy.logical_and(clean_errors == 0, errors == 1)]), global_step=global_step)

        if self.summary_histograms:
            self.writer.add_histogram('test/pre_adversarial_losses', losses, global_step=global_step)
            self.writer.add_histogram('test/pre_adversarial_errors', errors, global_step=global_step)
            self.writer.add_histogram('test/pre_adversarial_logits', logits, global_step=global_step)
            self.writer.add_histogram('test/pre_adversarial_confidences', confidences, global_step=global_step)

        self.writer.add_scalar('test/adversarial_loss', numpy.mean(adversarial_losses), global_step=global_step)
        self.writer.add_scalar('test/adversarial_error', numpy.mean(adversarial_errors), global_step=global_step)
        self.writer.add_scalar('test/adversarial_logit', numpy.mean(adversarial_logits), global_step=global_step)
        self.writer.add_scalar('test/adversarial_confidence', numpy.mean(adversarial_confidences), global_step=global_step)

        norms = numpy.array(norms)
        self.writer.add_scalar('test/adversarial_norm', numpy.mean(norms), global_step=global_step)

        if self.summary_histograms:
            self.writer.add_histogram('test/adversarial_losses', adversarial_losses, global_step=global_step)
            self.writer.add_histogram('test/adversarial_errors', adversarial_errors, global_step=global_step)
            self.writer.add_histogram('test/adversarial_logits', adversarial_logits, global_step=global_step)
            self.writer.add_histogram('test/adversarial_confidences', adversarial_confidences, global_step=global_step)
            self.writer.add_histogram('test/adversarial_norms', norms, global_step=global_step)

        if getattr(self.weight_attack, 'error_bound', None) is not None:
            self.weight_attack.error_bound = error_bound

        return probabilities, forward_model