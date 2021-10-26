import os
import numpy
import datetime
import common.paths
import common.utils
import common.train
import common.test
import common.state
import common.eval
import common.calibration
import common.hessian
import attacks.weights.projections
from common.log import log, LogLevel
import attacks
import attacks.weights
from .config import *


def find_incomplete_file(model_file, ext=common.paths.STATE_EXT):
    """
    State file.

    :param model_file: base state file
    :type model_file: str
    :return: state file of ongoing training
    :rtype: str
    """

    base_directory = os.path.dirname(os.path.realpath(model_file))
    file_name = os.path.basename(model_file)

    if os.path.exists(base_directory):
        state_files = []
        files = [os.path.basename(f) for f in os.listdir(base_directory) if os.path.isfile(os.path.join(base_directory, f))]

        for file in files:
            if file.find(file_name) >= 0 and file != file_name:
                state_files.append(file)

        if len(state_files) > 0:
            epochs = [state_files[i].replace(file_name, '').replace(ext, '').replace('.', '') for i in range(len(state_files))]
            epochs = [epoch for epoch in epochs if epoch.isdigit()]
            epochs = list(map(int, epochs))
            epochs = [epoch for epoch in epochs if epoch >= 0]

            if len(epochs) > 0:
                # list is not ordered by epochs!
                i = numpy.argmax(epochs)
                return os.path.join(base_directory, file_name + '.%d' % epochs[i]), epochs[i]

    return None, None


def find_incomplete_files(model_file, ext=common.paths.STATE_EXT):
    """
    State file.

    :param model_file: base state file
    :type model_file: str
    :return: state file of ongoing training
    :rtype: str
    """

    base_directory = os.path.dirname(os.path.realpath(model_file))
    file_name = os.path.basename(model_file)

    if os.path.exists(base_directory):
        state_files = []
        files = [os.path.basename(f) for f in os.listdir(base_directory) if os.path.isfile(os.path.join(base_directory, f))]

        for file in files:
            if file.find(file_name) >= 0 and file != file_name:
                state_files.append(file)

        if len(state_files) > 0:
            epochs = [state_files[i].replace(file_name, '').replace(ext, '').replace('.', '') for i in range(len(state_files))]
            epochs = [epoch for epoch in epochs if epoch.isdigit()]
            epochs = list(map(int, epochs))
            epochs = [epoch for epoch in epochs if epoch >= 0]
            epochs = sorted(epochs)

            return [os.path.join(base_directory, file_name + '.%d' % epoch) for epoch in epochs], epochs

    return None, None


class NormalTrainingInterface:
    """
    Interface for normal training for expeirments.
    """

    def __init__(self, config):
        """
        Initialize.

        :param config: configuration
        :type config: [str]
        """

        assert isinstance(config, NormalTrainingConfig)
        config.validate()

        self.config = config
        """ (NormalTrainingConfig) Config. """

        # Options set in setup
        self.log_dir = None
        """ (str) Log directory. """

        self.model_file = None
        """ (str) Model file. """

        self.probabilities_file = None
        """ (str) Probabilities file. """

        self.cuda = None
        """ (bool) Whether to use CUDA. """

        self.epochs = None
        """ (int) Epochs. """

        self.epoch = None
        """ (int) Start epoch. """

        self.writer = None
        """ (common.summary.SummaryWriter or torch.utils.tensorboard.SumamryWriter) Summary writer. """

        self.augmentation = None
        """ (None or iaa.meta.Augmenter or torchvision.transforms.Transform) Data augmentation. """

        self.trainloader = None
        """ (torch.utils.data.DataLoader) Train loader. """

        self.testloader = None
        """ (torch.utils.data.DataLoader) Test loader. """

        self.projection = None
        """ (attacks.weights.projections.Projection) Projection. """

        self.model = None
        """ (torch.nn.Module) Model. """

        self.optimizer = None
        """ (torch.optim.Optimizer) Optimizer. """

        self.scheduler = None
        """ (torch.optim.lr_scheduler.LRScheduler) Scheduler. """

        self.finetune = None
        """ (str) Finetune model. """

        self.summary_histograms = False
        """ (bool) Summary gradients. """

        self.summary_weights = False
        """ (bool) Summary weights. """

    def setup(self):
        """
        Setup.
        """

        dt = datetime.datetime.now()
        self.log_dir = common.paths.log_file(self.config.directory, 'logs/%s' % dt.strftime('%d%m%y%H%M%S'))
        self.model_file = common.paths.experiment_file(self.config.directory, 'classifier', common.paths.STATE_EXT)
        self.probabilities_file = common.paths.experiment_file(self.config.directory, 'probabilities', common.paths.HDF5_EXT)

        self.cuda = self.config.cuda
        self.epochs = self.config.epochs

        self.epoch = 0
        state = None

        self.writer = self.config.get_writer(self.log_dir)
        self.augmentation = self.config.augmentation
        self.loss = self.config.loss
        self.trainloader = self.config.trainloader
        self.testloader = self.config.testloader
        self.projection = self.config.projection
        self.finetune = self.config.finetune
        self.summary_histograms = self.config.summary_histograms
        self.summary_weights = self.config.summary_weights
        self.summary_images = self.config.summary_images

        state = self.setup_model()
        self.setup_optimizer(state)

    def setup_model(self):
        incomplete_model_file, epoch = find_incomplete_file(self.model_file)
        load_file = self.model_file
        if os.path.exists(load_file):
            state = common.state.State.load(load_file)
            self.model = state.model
            self.epoch = self.epochs
            log('classifier.pth.tar found, just evaluating', LogLevel.WARNING)
        elif incomplete_model_file is not None and os.path.exists(incomplete_model_file):
            load_file = incomplete_model_file
            state = common.state.State.load(load_file)
            self.model = state.model
            self.epoch = state.epoch + 1
            log('loaded %s, epoch %d' % (load_file, self.epoch))
        else:
            if self.finetune is not None:
                finetune_file = common.paths.experiment_file(self.finetune, 'classifier', common.paths.STATE_EXT)
                probabilities_file = common.paths.experiment_file(self.finetune, 'probabilities', common.paths.HDF5_EXT)
                assert os.path.exists(finetune_file), finetune_file
                assert os.path.exists(probabilities_file)
                state = common.state.State.load(finetune_file)
                self.model = state.model
                if state.epoch is None: # should not happen
                    self.epoch = 100 + 1
                else:
                    self.epoch = state.epoch + 1

                log('fine-tuning %s' % finetune_file)

                probabilities = common.utils.read_hdf5(probabilities_file, 'probabilities')
                eval = common.eval.CleanEvaluation(probabilities, self.testloader.dataset.labels, validation=0)
                log('fine-tune test error in %%: %g' % (eval.test_error() * 100))
            else:
                state = None
                self.model = self.get_model()
                assert self.model is not None

        if self.cuda:
            self.model = self.model.cuda()

        if self.finetune is not None:
            self.model.eval()
            probabilities = common.test.test(self.model, self.testloader, cuda=self.cuda)
            eval = common.eval.CleanEvaluation(probabilities, self.testloader.dataset.labels, validation=0)
            log('fine-tune checked test error in %%: %g' % (eval.test_error() * 100))

        print(self.model)
        return state

    def setup_optimizer(self, state):
        self.optimizer = self.config.get_optimizer(self.model)
        if state is not None and self.finetune is None and state.optimizer is not None:
            # fine-tuning should start with fresh optimizer and learning rate
            try:
                self.optimizer.load_state_dict(state.optimizer)
            except ValueError as e:
                log('loaded optimizer dict did not work', LogLevel.WARNING)

        self.scheduler = self.config.get_scheduler(self.optimizer)
        if state is not None and self.finetune is None and state.scheduler is not None:
            # will lead to errors when fine-tuning pruned models
            self.scheduler.load_state_dict(state.scheduler)
            log('loaded scheduler')

    def get_model(self):
        assert callable(self.config.get_model)
        N_class = numpy.max(self.trainloader.dataset.labels) + 1
        resolution = [
            self.trainloader.dataset.images.shape[3],
            self.trainloader.dataset.images.shape[1],
            self.trainloader.dataset.images.shape[2],
        ]
        model = self.config.get_model(N_class, resolution)
        return model

    def trainer(self):
        """
        Trainer.
        """

        trainer = common.train.NormalTraining(self.model, self.trainloader, self.testloader, self.optimizer, self.scheduler,
                                              augmentation=self.augmentation, loss=self.loss, writer=self.writer, cuda=self.cuda)

        return trainer

    def checkpoint(self, model_file, model, epoch=None):
        """
        Save file and check to delete previous file.

        :param model_file: path to file
        :type model_file: str
        :param model: model
        :type model: torch.nn.Module
        :param epoch: epoch of file
        :type epoch: None or int
        """

        if epoch is not None:
            checkpoint_model_file = '%s.%d' % (model_file, epoch)
            common.state.State.checkpoint(checkpoint_model_file, model, self.optimizer, self.scheduler, epoch)
        else:
            epoch = self.epochs
            checkpoint_model_file = model_file
            common.state.State.checkpoint(checkpoint_model_file, model, self.optimizer, self.scheduler, epoch)

        previous_model_file = '%s.%d' % (model_file, epoch - 1)
        if os.path.exists(previous_model_file) and (self.config.snapshot is None or (epoch - 1) % self.config.snapshot > 0):
            os.unlink(previous_model_file)

    def main(self):
        """
        Main.
        """

        self.setup()
        trainer = self.trainer()

        if self.config.loss is not None:
            trainer.loss = self.config.loss
        if self.projection is not None:
            assert isinstance(self.projection, attacks.weights.projections.Projection)
            trainer.projection = self.projection
            log('set projection')

            #max_bound = getattr(trainer.projection, 'max_bound')
            #min_bound = getattr(trainer.projection, 'min_bound')
            #if max_bound is not None:
            #    log('max_bound=%g' % max_bound)
            #if min_bound is not None:
            #    log('min_bound=%g' % min_bound)

        trainer.keep_average = self.config.keep_average
        trainer.keep_average_tau = self.config.keep_average_tau
        log('keep avarage: %r (%g)' % (self.config.keep_average, self.config.keep_average_tau))

        trainer.summary_histograms = self.summary_histograms
        trainer.summary_weights = self.summary_weights
        trainer.summary_images = self.summary_images

        if self.epoch < self.epochs:
            e = self.epochs - 1
            for e in range(self.epoch, self.epochs):
                probabilities, forward_model = trainer.step(e)
                self.writer.flush()

                self.checkpoint(self.model_file, forward_model, e)

                if trainer.average is not None:
                    self.checkpoint(self.model_file + 'average', trainer.average, e)

                probabilities_file = '%s.%d' % (self.probabilities_file, e)
                common.utils.write_hdf5(probabilities_file, probabilities, 'probabilities')

                previous_probabilities_file = '%s.%d' % (self.probabilities_file, e - 1)
                if os.path.exists(previous_probabilities_file) and (self.config.snapshot is None or (e - 1)%self.config.snapshot > 0):
                    os.unlink(previous_probabilities_file)

            self.checkpoint(self.model_file, forward_model)
            self.checkpoint(self.model_file, forward_model, e)
                
            if trainer.average is not None:
                self.checkpoint(self.model_file + 'average', trainer.average)

            previous_probabilities_file = '%s.%d' % (self.probabilities_file, e - 1)
            if os.path.exists(previous_probabilities_file) and (self.config.snapshot is None or (e - 1) % self.config.snapshot > 0):
                os.unlink(previous_probabilities_file)

            forward_model.eval()
            probabilities = common.test.test(forward_model, self.testloader, cuda=self.cuda)
            common.utils.write_hdf5(self.probabilities_file, probabilities, 'probabilities')

            if trainer.average is not None:
                log('calibrating')
                common.calibration.reset(trainer.average)
                common.calibration.calibrate(trainer.average, trainer.trainset, trainer.testset, trainer.augmentation, epochs=1, cuda=self.cuda)
                self.checkpoint(self.model_file + 'average.calibrated', trainer.average)
        else:
            # when not doing any epochs, self.model is the forward model
            self.model.eval()
            probabilities = common.test.test(self.model, self.testloader, cuda=self.cuda)
            common.utils.write_hdf5(self.probabilities_file, probabilities, 'probabilities')

        eval = common.eval.CleanEvaluation(probabilities, self.testloader.dataset.labels, validation=0)
        log('test error in %%: %g' % (eval.test_error() * 100))


class AdversarialTrainingInterface(NormalTrainingInterface):
    """
    Interface for adversarial training.
    """

    def __init__(self, config):
        """
        Initialize.

        :param config: configuration
        :type config: [str]
        """

        assert isinstance(config, AdversarialTrainingConfig)

        super(AdversarialTrainingInterface, self).__init__(config)

    def trainer(self):
        """
        Trainer.
        """

        trainer = common.train.AdversarialTraining(self.model, self.trainloader, self.testloader, self.optimizer, self.scheduler,
                                                self.config.attack, self.config.objective, self.config.fraction,
                                                augmentation=self.augmentation, loss=self.loss, writer=self.writer, cuda=self.cuda)
        trainer.prevent_label_leaking = self.config.prevent_label_leaking
        log('prevent label leaking: %s' % str(trainer.prevent_label_leaking))
        trainer.ignore_incorrect = self.config.ignore_incorrect
        log('ignore incorrect: %s' % str(trainer.ignore_incorrect))
        return trainer


class EntropyAdversarialTrainingInterface(AdversarialTrainingInterface):
    def trainer(self):
        """
        Trainer.
        """

        trainer = common.train.EntropyAdversarialTraining(self.model, self.trainloader, self.testloader, self.optimizer,
                                                          self.scheduler, self.config.attack, self.config.objective, self.config.fraction,
                                                          augmentation=self.augmentation, loss=self.loss, writer=self.writer, cuda=self.cuda)
        return trainer


class MARTAdversarialTrainingInterface(AdversarialTrainingInterface):
    def trainer(self):
        """
        Trainer.
        """

        trainer = common.train.MARTAdversarialTraining(self.model, self.trainloader, self.testloader, self.optimizer,
                                                          self.scheduler, self.config.attack, self.config.objective, self.config.fraction,
                                                          augmentation=self.augmentation, loss=self.loss, writer=self.writer, cuda=self.cuda)
        return trainer


class TRADESAdversarialTrainingInterface(AdversarialTrainingInterface):
    def trainer(self):
        """
        Trainer.
        """

        trainer = common.train.TRADESAdversarialTraining(self.model, self.trainloader, self.testloader, self.optimizer,
                                                          self.scheduler, self.config.attack, self.config.objective, self.config.fraction,
                                                          augmentation=self.augmentation, loss=self.loss, writer=self.writer, cuda=self.cuda)
        return trainer


class AdversarialWeightsInputsTrainingInterface(NormalTrainingInterface):
    """
    Interface for adversarial training.
    """

    def __init__(self, config):
        """
        Initialize.

        :param config: configuration
        :type config: [str]
        """

        assert isinstance(config, AdversarialWeightsInputsTrainingConfig), type(config)
        super(AdversarialWeightsInputsTrainingInterface, self).__init__(config)

        self.trainer_class = common.train.AdversarialWeightsInputsTraining
        """ (class) Trainer class. """

    def trainer(self):
        """
        Trainer.
        """

        if self.projection is not None:
            if self.config.weight_attack.projection is None:
                self.config.weight_attack.projection = self.projection
            else:
                self.config.weight_attack.projection = attacks.weights.projections.SequentialProjections([
                    self.projection,
                    self.config.weight_attack.projection,
                ])
            log('set/added attack projection %s' % self.projection.__class__.__name__)

        trainer = self.trainer_class(self.model, self.trainloader, self.testloader, self.optimizer, self.scheduler,
                                     self.config.weight_attack, self.config.weight_objective, self.config.input_attack, self.config.input_objective,
                                     operators=self.config.operators, augmentation=self.augmentation, loss=self.loss, writer=self.writer, cuda=self.cuda)
        trainer.clean = self.config.clean
        if trainer.clean:
            log('clean AWP')
        trainer.curriculum = self.config.curriculum
        trainer.gradient_clipping = self.config.gradient_clipping
        log('gradient clipping %g' % trainer.gradient_clipping)
        trainer.reset_iterations = self.config.reset_iterations
        log('reset_iterations %g' % trainer.reset_iterations)
        if getattr(trainer, 'average_statistics', None) is not None:
            setattr(trainer, 'average_statistics', self.config.average_statistics)
            log('average statistics %g' % getattr(trainer, 'average_statistics', None))
        if getattr(trainer, 'adversarial_statistics', None) is not None:
            setattr(trainer, 'adversarial_statistics', self.config.adversarial_statistics)
            log('adversarial statistics %g' % getattr(trainer, 'adversarial_statistics', None))

        return trainer


class SemiSupervisedTrainingInterface(NormalTrainingInterface):
    """
    Interface for adversarial training.
    """

    def __init__(self, config):
        """
        Initialize.

        :param config: configuration
        :type config: [str]
        """

        assert isinstance(config, SemiSupervisedTrainingConfig)

        super(SemiSupervisedTrainingInterface, self).__init__(config)

    def get_model(self):
        model = super(SemiSupervisedTrainingInterface, self).get_model()

        for b, (inputs, targets) in enumerate(self.testloader):
            break
        inputs = common.torch.as_variable(inputs)
        inputs = inputs.permute(0, 3, 1, 2)

        logits, features = model.forward(inputs, return_features=True)
        unsup_classes = self.trainloader.dataset.unsup_classes
        model.auxiliary = self.config.get_auxiliary_model(unsup_classes, list(features[-2].size()[1:]))

        return model

    def trainer(self):
        """
        Trainer.
        """

        return common.train.SemiSupervisedTraining(self.model, self.trainloader, self.testloader, self.optimizer, self.scheduler, unsup_weight=self.config.unsup_weight,
                                                augmentation=self.augmentation, loss=self.loss, writer=self.writer, unsup_loss=self.config.unsup_loss, cuda=self.cuda)


class AdversarialSemiSupervisedTrainingInterface(NormalTrainingInterface):
    """
    Interface for adversarial training.
    """

    def __init__(self, config):
        """
        Initialize.

        :param config: configuration
        :type config: [str]
        """

        assert isinstance(config, AdversarialSemiSupervisedTrainingConfig)

        super(AdversarialSemiSupervisedTrainingInterface, self).__init__(config)

    def get_model(self):
        model = super(AdversarialSemiSupervisedTrainingInterface, self).get_model()

        for b, (inputs, targets) in enumerate(self.testloader):
            break
        inputs = common.torch.as_variable(inputs)
        inputs = inputs.permute(0, 3, 1, 2)

        logits, features = model.forward(inputs, return_features=True)
        unsup_classes = self.trainloader.dataset.unsup_classes
        model.auxiliary = self.config.get_auxiliary_model(unsup_classes, list(features[-2].size()[1:]))

        return model

    def trainer(self):
        """
        Trainer.
        """

        return common.train.AdversarialSemiSupervisedTraining(self.model, self.trainloader, self.testloader, self.optimizer, self.scheduler,
                                                self.config.attack, self.config.objective, self.config.fraction, unsup_weight=self.config.unsup_weight,
                                                augmentation=self.augmentation, loss=self.loss, unsup_loss=self.config.unsup_loss, writer=self.writer, cuda=self.cuda)


class AttackInterface:
    """
    Regular attack interface.
    """

    def __init__(self, target_config, attack_config):
        """
        Initialize.

        :param target_config: configuration
        :type target_config: [str]
        :param attack_config: configuration
        :type attack_config: [str]
        """

        assert isinstance(target_config, NormalTrainingConfig)
        assert isinstance(attack_config, AttackConfig)

        #target_config.validate()
        attack_config.validate()

        self.target_config = target_config
        """ (NormalTrainingConfig) Config. """

        self.attack_config = attack_config
        """ (AttackConfig) Config. """

        # Options set in setup
        self.log_dir = None
        """ (str) Log directory. """

        self.model_file = None
        """ (str) Model file. """

        self.perturbations_file = None
        """ (str) Perturbations file. """

        self.cuda = None
        """ (bool) Whether to use CUDA. """

        self.writer = None
        """ (common.summary.SummaryWriter or torch.utils.tensorboard.SumamryWriter) Summary writer. """

        self.testloader = None
        """ (torch.utils.data.DataLoader) Test loader. """

        self.model = None
        """ (torch.nn.Module) Model. """

    def main(self, force=False):
        """
        Main.
        """

        self.log_dir = common.paths.log_dir('%s/%s' % (self.target_config.directory, self.attack_config.directory))
        self.perturbations_file = common.paths.experiment_file('%s/%s' % (self.target_config.directory, self.attack_config.directory), 'perturbations', common.paths.HDF5_EXT)
        self.model_file = common.paths.experiment_file(self.target_config.directory, 'classifier', common.paths.STATE_EXT)

        if self.attack_config.snapshot is not None:
            self.perturbations_file = common.paths.experiment_file('%s/%s' % (self.target_config.directory, self.attack_config.directory), 'perturbations%d' % self.attack_config.snapshot, common.paths.HDF5_EXT)
            self.model_file += '.%d' % self.attack_config.snapshot

        assert os.path.exists(self.model_file), 'file %s not found' % self.model_file

        attempts = 0
        samples = 0
        if not force and os.path.exists(self.perturbations_file):
            errors = common.utils.read_hdf5(self.perturbations_file, key='errors')
            attempts = errors.shape[0]
            samples = errors.shape[1]

        if force or not os.path.exists(self.perturbations_file) \
                or attempts < self.attack_config.attempts \
                or samples < len(self.attack_config.testloader.dataset):

            self.cuda = self.target_config.cuda
            if callable(self.attack_config.get_writer):
                self.writer = common.utils.partial(self.attack_config.get_writer, self.log_dir)
            else:
                self.writer = self.attack_config.get_writer

            state = common.state.State.load(self.model_file)
            log('read %s' % self.model_file)
            self.model = state.model
            log(self.model.__class__)

            if self.cuda:
                self.model = self.model.cuda()

            self.model.eval()
            perturbations, probabilities, errors = common.test.attack(self.model, self.attack_config.testloader, self.attack_config.attack, self.attack_config.objective, attempts=self.attack_config.attempts, writer=self.writer, cuda=self.cuda)

            eval = common.eval.CleanEvaluation(probabilities[0], self.attack_config.testloader.dataset.labels)
            log('first attempt test error: %g' % eval.test_error())

            common.utils.write_hdf5(self.perturbations_file, [
                perturbations,
                probabilities,
                errors,
            ], [
                'perturbations',
                'probabilities',
                'errors',
            ])


class AttackWeightsInterface:
    """
    Regular attack interface.
    """

    def __init__(self, target_config, attack_config):
        """
        Initialize.

        :param target_config: configuration
        :type target_config: [str]
        :param attack_config: configuration
        :type attack_config: [str]
        """

        assert isinstance(target_config, NormalTrainingConfig)
        assert isinstance(attack_config, AttackWeightsConfig)

        #target_config.validate()
        attack_config.validate()

        self.target_config = target_config
        """ (NormalTrainingConfig) Config. """

        self.attack_config = attack_config
        """ (AttackConfig) Config. """

        # Options set in setup
        self.log_dir = None
        """ (str) Log directory. """

        self.model_file = None
        """ (str) Model file. """

        self.perturbations_directory = None
        """ (str) Perturbations directory. """

        self.cuda = None
        """ (bool) Whether to use CUDA. """

        self.writer = None
        """ (common.summary.SummaryWriter or torch.utils.tensorboard.SumamryWriter) Summary writer. """

        self.model = None
        """ (torch.nn.Module) Model. """

    def main(self, force_attack=False, force_probabilities=False):
        """
        Main.
        """

        self.log_dir = common.paths.log_dir('%s/%s' % (self.target_config.directory, self.attack_config.directory))
        self.perturbations_directory = common.paths.experiment_dir('%s/%s' % (self.target_config.directory, self.attack_config.directory))
        self.model_file = common.paths.experiment_file(self.target_config.directory, 'classifier', common.paths.STATE_EXT)

        snapshot = self.attack_config.snapshot
        #if os.path.exists(self.model_file):
        #    snapshot = False

        if snapshot is not None:
            self.log_dir = common.paths.log_dir('%s/%s_%d' % (self.target_config.directory, self.attack_config.directory, snapshot))
            self.model_file = self.model_file + '.%d' % snapshot

        assert os.path.exists(self.model_file), 'file %s not found' % self.model_file

        rerun = True
        if os.path.exists(self.perturbations_directory):
            #log('found %s' % self.perturbations_directory)
            rerun = False
            for i in range(self.attack_config.attempts):
                if snapshot is not None:
                    probabilities_file = os.path.join(self.perturbations_directory, 'probabilities%d%s.%d' % (i, common.paths.HDF5_EXT, snapshot))
                else:
                    probabilities_file = os.path.join(self.perturbations_directory, 'probabilities%d%s' % (i, common.paths.HDF5_EXT))
                if not os.path.exists(probabilities_file):
                    rerun = True
                else:
                    log('found %s' % probabilities_file)

        if rerun or force_attack:
            self.cuda = self.target_config.cuda
            if callable(self.attack_config.get_writer):
                get_writer = common.utils.partial(self.attack_config.get_writer, log_dir=self.log_dir)
                self.writer = get_writer()
            else:
                self.writer = self.attack_config.get_writer

            state = common.state.State.load(self.model_file)
            log('read %s' % self.model_file)
            self.model = state.model

            print(self.model)

            if self.target_config.projection is not None:
                if self.attack_config.attack.projection is None:
                    self.attack_config.attack.projection = self.target_config.projection
                else:
                    self.attack_config.attack.projection = attacks.weights.projections.SequentialProjections([
                        self.target_config.projection,
                        self.attack_config.attack.projection,
                    ])
                log('set/added attack projection %s' % self.target_config.projection.__class__.__name__)

            if self.cuda:
                self.model = self.model.cuda()
            if self.attack_config.eval:
                self.model.eval()
            assert self.attack_config.eval is not self.model.training

            perturbed_models = common.test.attack_weights(self.model, self.attack_config.trainloader, self.attack_config.attack,
                                                          self.attack_config.objective, attempts=self.attack_config.attempts,
                                                          writer=self.writer, eval=self.attack_config.eval, cuda=self.cuda)

            for i in range(len(perturbed_models)):
                perturbed_model = perturbed_models[i]

                whiten = getattr(self.model, 'whiten', None)
                if whiten is not None:
                    assert torch.allclose(getattr(perturbed_model, 'whiten').weight.cpu(), whiten.weight.cpu()), (getattr(perturbed_model, 'whiten').weight.cpu(), whiten.weight.cpu())
                    assert torch.allclose(getattr(perturbed_model, 'whiten').bias.cpu(), whiten.bias.cpu()), (getattr(perturbed_model, 'whiten').bias.cpu(), whiten.bias.cpu())

                if os.getenv('SAVE_MODELS', None) is not None or self.attack_config.save_models:
                    perturbed_model_file = os.path.join(self.perturbations_directory, 'perturbation%d%s' % (i, common.paths.STATE_EXT))
                    if snapshot is not None:
                        perturbed_model_file = os.path.join(self.perturbations_directory, 'perturbation%d%s.%d' % (i, common.paths.STATE_EXT, snapshot))
                    common.state.State.checkpoint(perturbed_model_file, perturbed_model)
                    log('saving model file %s!' % perturbed_model_file, LogLevel.WARNING)

                if self.attack_config.eval:
                    perturbed_model.eval()
                assert self.attack_config.eval is not perturbed_model.training
                if self.cuda:
                    perturbed_model = perturbed_model.cuda()

                if self.attack_config.operators is not None:
                    for operator in self.attack_config.operators:
                        operator.reset()

                log('%d/%d' % (i, len(perturbed_models)))
                probabilities = common.test.test(perturbed_model, self.attack_config.testloader, operators=self.attack_config.operators,
                                                 eval=self.attack_config.eval, cuda=self.cuda)
                evaluation = common.eval.CleanEvaluation(probabilities, self.attack_config.testloader.dataset.labels)
                log('error: %g' % evaluation.test_error())
                probabilities_file = os.path.join(self.perturbations_directory, 'probabilities%d%s' % (i, common.paths.HDF5_EXT))
                if snapshot is not None:
                    probabilities_file = os.path.join(self.perturbations_directory, 'probabilities%d%s.%d' % (i, common.paths.HDF5_EXT, snapshot))
                common.utils.write_hdf5(probabilities_file, probabilities, 'probabilities')
                log('wrote %s' % probabilities_file)

        elif force_probabilities:
            raise NotImplementedError()


class AttackWeightsInputsInterface:
    """
    Regular attack interface.
    """

    def __init__(self, target_config, attack_config):
        """
        Initialize.

        :param target_config: configuration
        :type target_config: [str]
        :param attack_config: configuration
        :type attack_config: [str]
        """

        assert isinstance(target_config, NormalTrainingConfig)
        assert isinstance(attack_config, AttackWeightsInputsConfig)

        #target_config.validate()
        attack_config.validate()

        self.target_config = target_config
        """ (NormalTrainingConfig) Config. """

        self.attack_config = attack_config
        """ (AttackConfig) Config. """

        # Options set in setup
        self.log_dir = None
        """ (str) Log directory. """

        self.model_file = None
        """ (str) Model file. """

        self.perturbations_directory = None
        """ (str) Perturbations directory. """

        self.cuda = None
        """ (bool) Whether to use CUDA. """

        self.writer = None
        """ (common.summary.SummaryWriter or torch.utils.tensorboard.SumamryWriter) Summary writer. """

        self.model = None
        """ (torch.nn.Module) Model. """

    def main(self, force_attack=False, force_probabilities=False):
        """
        Main.
        """

        self.log_dir = common.paths.log_dir('%s/%s' % (self.target_config.directory, self.attack_config.directory))
        self.perturbations_directory = common.paths.experiment_dir('%s/%s' % (self.target_config.directory, self.attack_config.directory))
        self.model_file = common.paths.experiment_file(self.target_config.directory, 'classifier', common.paths.STATE_EXT)

        snapshot = self.attack_config.snapshot
        #if os.path.exists(self.model_file):
        #    snapshot = False

        if snapshot is not None:
            self.log_dir = common.paths.log_dir('%s/%s_%d' % (self.target_config.directory, self.attack_config.directory, snapshot))
            self.model_file = self.model_file + '.%d' % snapshot

        assert os.path.exists(self.model_file), 'file %s not found' % self.model_file

        rerun = True
        if os.path.exists(self.perturbations_directory):
            #log('found %s' % self.perturbations_directory)
            rerun = False
            for i in range(self.attack_config.attempts):
                if snapshot is not None:
                    probabilities_file = os.path.join(self.perturbations_directory, 'probabilities%d%s.%d' % (i, common.paths.HDF5_EXT, snapshot))
                else:
                    probabilities_file = os.path.join(self.perturbations_directory, 'probabilities%d%s' % (i, common.paths.HDF5_EXT))
                if not os.path.exists(probabilities_file):
                    rerun = True
                else:
                    log('found %s' % probabilities_file)

        if rerun or force_attack:
            self.cuda = self.target_config.cuda
            if callable(self.attack_config.get_writer):
                get_writer = common.utils.partial(self.attack_config.get_writer, log_dir=self.log_dir)
                self.writer = get_writer()
            else:
                self.writer = self.attack_config.get_writer

            state = common.state.State.load(self.model_file)
            log('read %s' % self.model_file)
            self.model = state.model

            print(self.model)

            if self.target_config.projection is not None:
                if self.attack_config.attack.weight_projection is None:
                    self.attack_config.attack.weight_projection = self.target_config.projection
                else:
                    self.attack_config.attack.weight_projection = attacks.weights.projections.SequentialProjections([
                        self.target_config.projection,
                        self.attack_config.attack.weight_projection,
                    ])
                log('set/added attack projection %s' % self.target_config.projection.__class__.__name__)

            if self.cuda:
                self.model = self.model.cuda()
            self.model.eval()

            perturbed_models, perturbations, probabilities, errors = common.test.attack_weights_inputs(self.model, self.attack_config.testloader, self.attack_config.attack,
                                                                 self.attack_config.weight_objective, self.attack_config.input_objective,
                                                                 attempts=self.attack_config.attempts, writer=self.writer, cuda=self.cuda)

            for i in range(len(perturbed_models)):
                perturbed_model = perturbed_models[i]

                whiten = getattr(self.model, 'whiten', None)
                if whiten is not None:
                    assert torch.allclose(getattr(perturbed_model, 'whiten').weight.cpu(), whiten.weight.cpu()), (getattr(perturbed_model, 'whiten').weight.cpu(), whiten.weight.cpu())
                    assert torch.allclose(getattr(perturbed_model, 'whiten').bias.cpu(), whiten.bias.cpu()), (getattr(perturbed_model, 'whiten').bias.cpu(), whiten.bias.cpu())

                perturbed_model_file = os.path.join(self.perturbations_directory, 'perturbation%d%s' % (i, common.paths.STATE_EXT))
                if snapshot is not None:
                    perturbed_model_file = os.path.join(self.perturbations_directory, 'perturbation%d%s.%d' % (i, common.paths.STATE_EXT, snapshot))
                #common.state.State.checkpoint(perturbed_model_file, perturbed_model)

                probabilities_file = os.path.join(self.perturbations_directory, 'perturbations%d%s' % (i, common.paths.HDF5_EXT))
                if snapshot is not None:
                    probabilities_file = os.path.join(self.perturbations_directory, 'perturbations%d%s.%d' % (i, common.paths.HDF5_EXT, snapshot))
                common.utils.write_hdf5(probabilities_file, [
                    perturbations[i], probabilities[i], errors[i]
                ], [
                    'perturbations', 'probabilities', 'errors',
                ])

        elif force_probabilities:
            raise NotImplementedError()


class Visualize1DInterface:
    """
    Visualize using input attack.
    """

    def __init__(self, target_config, xattack_config, testloader, normalization, steps=101):
        """
        Initialize.

        :param target_config: configuration
        :type target_config: [str]
        :param xattack_config: configuration
        :type xattack_config: [str]
        """

        assert isinstance(target_config, NormalTrainingConfig)
        assert isinstance(xattack_config, AttackConfig)

        self.target_config = target_config
        """ (NormalTrainingConfig) Config. """

        self.xattack_config = xattack_config
        """ (AttackConfig) Config. """

        self.testloader = testloader
        """ (torch.utils.data.DataLoader) Dataloader. """

        self.normalization = normalization
        """ (str) Input normalization. """

        assert steps % 2 == 1
        self.steps = steps
        """ (int) Number of steps. """

    def main(self, force=False, force_attack=False):
        """
        Main.
        """

        cuda = self.target_config.cuda
        model_file = common.paths.experiment_file(self.target_config.directory, 'classifier', common.paths.STATE_EXT)
        if self.xattack_config.snapshot is not None:
            model_file += '.%d' % self.xattack_config.snapshot

        assert os.path.exists(model_file), 'file %s not found' % model_file

        def get_norm(attack):
            normalization = ''
            norm = getattr(attack, 'norm', None)
            if norm is not None:
                if isinstance(norm, attacks.norms.LInfNorm):
                    normalization = 'linf'
                elif isinstance(norm, attacks.norms.L2Norm):
                    normalization = 'l2'
                log('normalization %s' % normalization)
            return normalization

        if self.normalization is None or self.normalization == '':
            self.normalization = get_norm(self.xattack_config.attack)
        log('input normalization %s' % self.normalization)

        visualization_directory = common.paths.experiment_dir('%s/%s_%s_visualization' % (self.target_config.directory, self.xattack_config.directory, self.normalization))
        visualization_file = os.path.join(visualization_directory, 'visualization%s' % common.paths.HDF5_EXT)
        if self.xattack_config.snapshot is not None:
            visualization_file += '.%d' % self.xattack_config.snapshot

        if force or not os.path.exists(visualization_file):
            state = common.state.State.load(model_file)
            model = state.model
            log(model.__class__)

            if cuda:
                model = model.cuda()

            model.eval()

            steps = None
            factors = None
            losses = None
            errors = None
            probabilities = None

            def run_or_load(attack_config):
                if attack_config.model_specific and attack_config.snapshot is not None:
                    attack_directory = common.paths.experiment_dir('%s/%s_visualization_%d' % (
                    self.target_config.directory, attack_config.directory, attack_config.snapshot))
                else:
                    attack_directory = common.paths.experiment_dir('%s/%s_visualization' % (self.target_config.directory, attack_config.directory))
                attack_perturbations_file = os.path.join(attack_directory, 'perturbations%s' % common.paths.HDF5_EXT)

                if not os.path.exists(attack_perturbations_file) or force_attack:
                    perturbations = None
                    for attempt in range(attack_config.attempts):
                        perturbed_model, perturbations_ = common.visualization.input_attack(model, attack_config.attack, attack_config.objective, self.testloader)
                        perturbations = common.numpy.concatenate(perturbations, numpy.expand_dims(perturbations_, axis=0))

                    common.utils.write_hdf5(attack_perturbations_file, perturbations)
                    log('wrote %s' % attack_directory)
                else:
                    perturbations = common.utils.read_hdf5(attack_perturbations_file)

                return perturbations

            perturbations = run_or_load(self.xattack_config)

            for attempt in range(self.xattack_config.attempts):
                directions, inputs, targets, factors_ = common.visualization.input_direction(perturbations[attempt], normalization=self.normalization, cuda=cuda)
                steps_, losses_, probabilities_, errors_ = common.visualization.input_1d(model, directions, inputs, targets, steps=numpy.linspace(-1, 1, self.steps), cuda=cuda)

                steps = common.numpy.concatenate(steps, numpy.expand_dims(steps_, 0))
                factors = common.numpy.concatenate(factors, numpy.expand_dims(factors_, 0))
                losses = common.numpy.concatenate(losses, numpy.expand_dims(losses_, 0))
                errors = common.numpy.concatenate(errors, numpy.expand_dims(errors_, 0))
                probabilities = common.numpy.concatenate(probabilities, numpy.expand_dims(probabilities_, 0))

            common.utils.write_hdf5(visualization_file, [steps, losses, probabilities, errors, factors], ['steps', 'losses', 'probabilities', 'errors', 'factors'])
            log('wrote %s' % visualization_file)
        else:
            log('found %s' % visualization_file)


class Visualize2DInterface:
    """
    Visualize using input attack.
    """

    def __init__(self, target_config, xattack_config, yattack_config, testloader, normalization, steps=51):
        """
        Initialize.

        :param target_config: configuration
        :type target_config: [str]
        :param xattack_config: configuration
        :type xattack_config: [str]
        :param yattack_config: configuration
        :type yattack_config: [str]
        """

        assert isinstance(target_config, NormalTrainingConfig)
        assert isinstance(xattack_config, AttackConfig)
        assert isinstance(yattack_config, AttackConfig)

        self.target_config = target_config
        """ (NormalTrainingConfig) Config. """

        self.xattack_config = xattack_config
        """ (AttackConfig) Config. """

        self.yattack_config = yattack_config
        """ (AttackConfig) Config. """

        self.testloader = testloader
        """ (torch.utils.data.DataLoader) Dataloader. """

        self.normalization = normalization
        """ (str) Input normalization. """

        assert steps % 2 == 1
        self.steps = steps
        """ (int) Number of steps. """

    def main(self, force=False, force_attack=False):
        """
        Main.
        """

        cuda = self.target_config.cuda
        model_file = common.paths.experiment_file(self.target_config.directory, 'classifier', common.paths.STATE_EXT)
        if self.xattack_config.snapshot is not None:
            assert self.xattack_config.snapshot == self.yattack_config.snapshot
            model_file += '.%d' % self.xattack_config.snapshot

        assert os.path.exists(model_file), 'file %s not found' % model_file

        def get_norm(attack):
            normalization = ''
            norm = getattr(attack, 'norm', None)
            if norm is not None:
                if isinstance(norm, attacks.norms.LInfNorm):
                    normalization = 'linf'
                elif isinstance(norm, attacks.norms.L2Norm):
                    normalization = 'l2'
                log('normalization %s' % normalization)
            return normalization

        if self.normalization is None or self.normalization == '':
            self.normalization = get_norm(self.xattack_config.attack)
            assert self.normalization == get_norm(self.yattack_config.attack)
        log('input normalization %s' % self.normalization)

        visualization_directory = common.paths.experiment_dir('%s/%s_%s_%s_visualization' % (self.target_config.directory, self.xattack_config.directory, self.yattack_config.directory, self.normalization))
        visualization_file = os.path.join(visualization_directory, 'visualization%s' % common.paths.HDF5_EXT)
        if self.xattack_config.snapshot is not None:
            visualization_file += '.%d' % self.xattack_config.snapshot

        if force or not os.path.exists(visualization_file):

            state = common.state.State.load(model_file)
            model = state.model
            log(model.__class__)

            if cuda:
                model = model.cuda()

            model.eval()

            xsteps = None
            ysteps = None
            xfactors = None
            yfactors = None
            losses = None
            errors = None
            probabilities = None

            assert self.xattack_config.attempts == self.yattack_config.attempts

            def run_or_load(attack_config):
                if attack_config.model_specific and attack_config.snapshot is not None:
                    attack_directory = common.paths.experiment_dir('%s/%s_visualization_%d' % (self.target_config.directory, attack_config.directory, attack_config.snapshot))
                else:
                    attack_directory = common.paths.experiment_dir('%s/%s_visualization' % (self.target_config.directory, attack_config.directory))
                attack_perturbations_file = os.path.join(attack_directory, 'perturbations%s' % common.paths.HDF5_EXT)

                if not os.path.exists(attack_perturbations_file) or force_attack:
                    perturbations = None
                    for attempt in range(attack_config.attempts):
                        perturbed_model, perturbations_ = common.visualization.input_attack(model, attack_config.attack, attack_config.objective, self.testloader)
                        perturbations = common.numpy.concatenate(perturbations, numpy.expand_dims(perturbations_, axis=0))

                    common.utils.write_hdf5(attack_perturbations_file, perturbations)
                    log('wrote %s' % attack_directory)
                else:
                    perturbations = common.utils.read_hdf5(attack_perturbations_file)

                return perturbations

            xperturbations = run_or_load(self.xattack_config)
            yperturbations = run_or_load(self.yattack_config)

            for attempt in range(self.xattack_config.attempts):
                xdirections, inputs, targets, xfactors_ = common.visualization.input_direction(xperturbations[attempt], normalization=self.normalization, cuda=cuda)
                ydirections, _, _, yfactors_ = common.visualization.input_direction(yperturbations[-attempt], normalization=self.normalization, cuda=cuda)
                xsteps_, ysteps_, losses_, probabilities_, errors_ = common.visualization.input_2d(model, xdirections, ydirections, inputs, targets,
                                                                                          xsteps=numpy.linspace(-1, 1, self.steps),
                                                                                          ysteps=numpy.linspace(-1, 1, self.steps), cuda=cuda)

                xsteps = common.numpy.concatenate(xsteps, numpy.expand_dims(xsteps_, 0))
                ysteps = common.numpy.concatenate(ysteps, numpy.expand_dims(ysteps_, 0))
                xfactors = common.numpy.concatenate(xfactors, numpy.expand_dims(xfactors_, 0))
                yfactors = common.numpy.concatenate(yfactors, numpy.expand_dims(yfactors_, 0))
                losses = common.numpy.concatenate(losses, numpy.expand_dims(losses_, 0))
                errors = common.numpy.concatenate(errors, numpy.expand_dims(errors_, 0))
                probabilities = common.numpy.concatenate(probabilities, numpy.expand_dims(probabilities_, 0))

            common.utils.write_hdf5(visualization_file, [xsteps, ysteps, losses, probabilities, errors, xfactors, yfactors], ['xsteps', 'ysteps', 'losses', 'probabilities', 'errors', 'xfactors', 'yfactors'])
            log('wrote %s' % visualization_file)
        else:
            log('found %s' % visualization_file)


class VisualizeWeights1DInterface:
    """
    Visualize using input attack.
    """

    def __init__(self, target_config, xattack_config, testloader, normalization=None, input_attack_config=None, steps=101, hessian=0):
        """
        Initialize.

        :param target_config: configuration
        :type target_config: [str]
        :param xattack_config: configuration
        :type xattack_config: [str]
        :param yattack_config: configuration
        :type yattack_config: [str]
        """

        assert isinstance(target_config, NormalTrainingConfig)
        assert isinstance(xattack_config, AttackWeightsConfig)
        assert input_attack_config is None or isinstance(input_attack_config, AttackConfig), input_attack_config

        self.target_config = target_config
        """ (NormalTrainingConfig) Config. """

        self.xattack_config = xattack_config
        """ (AttackConfig) Config. """

        self.input_attack_config = input_attack_config
        """ (AttackConfig) Input attack config. """

        self.testloader = testloader
        """ (torch.utils.data.DataLoader) Dataloader. """

        self.normalization = normalization
        """ (str) Weight normalization. """

        assert steps % 2 == 1
        self.steps = steps
        """ (int) Number of steps. """

        self.hessian = hessian
        """ (int) Compute hessian eigenvalues. """

    def main(self, force=False, force_attack=False):
        """
        Main.
        """

        cuda = self.target_config.cuda
        model_file = common.paths.experiment_file(self.target_config.directory, 'classifier', common.paths.STATE_EXT)
        if self.xattack_config.snapshot is not None:
            model_file += '.%d' % self.xattack_config.snapshot
        log('read %s' % model_file)

        assert os.path.exists(model_file), 'file %s not found' % model_file

        def get_norm(attack):
            normalization = ''
            norm = getattr(attack, 'norm', None)
            if norm is not None:
                if isinstance(norm, attacks.weights.norms.LInfNorm):
                    normalization = 'linf'
                elif isinstance(norm, attacks.weights.norms.L2Norm):
                    normalization = 'l2'
                log('normalization %s' % normalization)
            return normalization

        if self.normalization is None or self.normalization == '':
            self.normalization = get_norm(self.xattack_config.attack)
        log('weight normalization %s' % self.normalization)

        visualization_directory = common.paths.experiment_dir('%s/%s_%s_visualization' % (self.target_config.directory, self.xattack_config.directory, self.normalization))
        if self.input_attack_config is not None:
            visualization_directory += '/%s' % self.input_attack_config.directory
        visualization_file = os.path.join(visualization_directory, 'visualization%s' % common.paths.HDF5_EXT)
        if self.xattack_config.snapshot is not None:
            visualization_file += '.%d' % self.xattack_config.snapshot

        if force or not os.path.exists(visualization_file) or (self.hessian > 0 and not common.utils.check_hdf5(visualization_file, 'eigs')):

            state = common.state.State.load(model_file)
            model = state.model
            log(model.__class__)

            if cuda:
                model = model.cuda()

            model.eval()

            steps = None
            factors = []
            losses = None
            errors = None
            eigs = None

            if self.input_attack_config is not None:
                perturbations, _, _ = common.test.attack(model, self.testloader, self.input_attack_config.attack,
                                                                           self.input_attack_config.objective, attempts=1, cuda=cuda)
                perturbations = numpy.transpose(perturbations[0], (0, 2, 3, 1))
                testset = common.datasets.AdversarialDataset(self.testloader.dataset.images, perturbations,
                                                             self.testloader.dataset.labels)
                testloader = torch.utils.data.DataLoader(testset, batch_size=self.testloader.batch_size)
            else:
                testloader = self.testloader

            def run_or_load(attack_config):
                if attack_config.model_specific and attack_config.snapshot is not None:
                    attack_directory = common.paths.experiment_dir('%s/%s_visualization_%d' % (self.target_config.directory, attack_config.directory, attack_config.snapshot))
                else:
                    attack_directory = common.paths.experiment_dir('%s/%s_visualization' % (self.target_config.directory, attack_config.directory))
                attack_model_file = os.path.join(attack_directory, 'perturbation0%s' % common.paths.STATE_EXT)

                if not os.path.exists(attack_model_file) or force_attack:
                    perturbed_models = []
                    for attempt in range(self.xattack_config.attempts):
                        perturbed_model = common.visualization.weight_attack(model, attack_config.attack, attack_config.objective, testloader)
                        perturbed_models.append(perturbed_model)

                    for attempt in range(self.xattack_config.attempts):
                        model_file = os.path.join(attack_directory, 'perturbation%d%s' % (attempt, common.paths.STATE_EXT))
                        common.state.State.checkpoint(model_file, perturbed_models[attempt].cpu())
                        log('wrote %s' % model_file)
                else:
                    perturbed_models = []
                    for attempt in range(self.xattack_config.attempts):
                        model_file = os.path.join(attack_directory, 'perturbation%d%s' % (attempt, common.paths.STATE_EXT))
                        assert os.path.exists(model_file), model_file
                        perturbed_models.append(common.state.State.load(model_file).model)

                return perturbed_models

            perturbed_models = run_or_load(self.xattack_config)

            for attempt in range(self.xattack_config.attempts):
                direction, factor = common.visualization.weight_direction(model, perturbed_models[attempt], self.xattack_config.attack, testloader, normalization=self.normalization, cuda=cuda)
                steps_, losses_, errors_ = common.visualization.weight_1d(model, direction, testloader, steps=numpy.linspace(-1, 1, self.steps), cuda=cuda)
                if self.hessian > 0:
                    steps_, eigs_ = common.visualization.hessian_1d(model, direction, testloader, hessian_k=self.hessian, steps=numpy.linspace(-1, 1, self.steps), cuda=cuda)
                    eigs = common.numpy.concatenate(eigs, numpy.expand_dims(eigs_, 0))

                steps = common.numpy.concatenate(steps, numpy.expand_dims(steps_, 0))
                factors.append(factor)
                losses = common.numpy.concatenate(losses, numpy.expand_dims(losses_, 0))
                errors = common.numpy.concatenate(errors, numpy.expand_dims(errors_, 0))

            factors = numpy.array(factors)
            tensors = [steps, losses, errors, factors]
            keys = ['steps', 'losses', 'errors', 'factors']
            if eigs is not None:
                tensors.append(eigs)
                keys.append('eigs')
            common.utils.write_hdf5(visualization_file, tensors, keys)
            log('wrote %s' % visualization_file)
        else:
            log('found %s' % visualization_file)


class VisualizeAdversarialWeights1DInterface:
    """
    Visualize using input attack.
    """

    def __init__(self, target_config, xattack_config, input_attack_config, testloader, normalization=None, steps=101, hessian=0):
        """
        Initialize.

        :param target_config: configuration
        :type target_config: [str]
        :param xattack_config: configuration
        :type xattack_config: [str]
        :param yattack_config: configuration
        :type yattack_config: [str]
        """

        assert isinstance(target_config, NormalTrainingConfig)
        assert isinstance(xattack_config, AttackWeightsConfig)
        assert isinstance(input_attack_config, AttackConfig), input_attack_config

        self.target_config = target_config
        """ (NormalTrainingConfig) Config. """

        self.xattack_config = xattack_config
        """ (AttackConfig) Config. """

        self.input_attack_config = input_attack_config
        """ (AttackConfig) Input attack config. """

        self.testloader = testloader
        """ (torch.utils.data.DataLoader) Dataloader. """

        self.normalization = normalization
        """ (str) Weight normalization. """

        assert steps % 2 == 1
        self.steps = steps
        """ (int) Number of steps. """

        self.hessian = hessian
        """ (int) Compute hessian eigenvalues. """

    def main(self, force=False, force_attack=False):
        """
        Main.
        """

        cuda = self.target_config.cuda
        model_file = common.paths.experiment_file(self.target_config.directory, 'classifier', common.paths.STATE_EXT)
        if self.xattack_config.snapshot is not None:
            model_file += '.%d' % self.xattack_config.snapshot
        log('read %s' % model_file)

        assert os.path.exists(model_file), 'file %s not found' % model_file

        def get_norm(attack):
            normalization = ''
            norm = getattr(attack, 'norm', None)
            if norm is not None:
                if isinstance(norm, attacks.weights.norms.LInfNorm):
                    normalization = 'linf'
                elif isinstance(norm, attacks.weights.norms.L2Norm):
                    normalization = 'l2'
                log('normalization %s' % normalization)
            return normalization

        if self.normalization is None or self.normalization == '':
            self.normalization = get_norm(self.xattack_config.attack)
        log('weight normalization %s' % self.normalization)

        visualization_directory = common.paths.experiment_dir('%s/%s_%s_adversarial_visualization' % (self.target_config.directory, self.xattack_config.directory, self.normalization))
        visualization_directory += '/%s' % self.input_attack_config.directory
        visualization_file = os.path.join(visualization_directory, 'visualization%s' % common.paths.HDF5_EXT)
        if self.xattack_config.snapshot is not None:
            visualization_file += '.%d' % self.xattack_config.snapshot

        if force or not os.path.exists(visualization_file) or (self.hessian > 0 and not common.utils.check_hdf5(visualization_file, 'eigs')):

            state = common.state.State.load(model_file)
            model = state.model
            log(model.__class__)

            if cuda:
                model = model.cuda()

            model.eval()

            # for SSL
            model.auxiliary = None

            steps = None
            factors = []
            losses = None
            errors = None
            eigs = None

            perturbations, _, _ = common.test.attack(model, self.testloader, self.input_attack_config.attack, self.input_attack_config.objective, attempts=1, cuda=cuda)
            perturbations = numpy.transpose(perturbations[0], (0, 2, 3, 1))
            testset = common.datasets.AdversarialDataset(self.testloader.dataset.images, perturbations, self.testloader.dataset.labels)
            testloader = torch.utils.data.DataLoader(testset, batch_size=self.testloader.batch_size)

            def run_or_load(attack_config):
                if attack_config.model_specific and attack_config.snapshot is not None:
                    attack_directory = common.paths.experiment_dir('%s/%s_visualization_%d' % (self.target_config.directory, attack_config.directory, attack_config.snapshot))
                else:
                    attack_directory = common.paths.experiment_dir('%s/%s_visualization' % (self.target_config.directory, attack_config.directory))
                attack_model_file = os.path.join(attack_directory, 'perturbation0%s' % common.paths.STATE_EXT)

                if not os.path.exists(attack_model_file) or force_attack:
                    perturbed_models = []
                    for attempt in range(self.xattack_config.attempts):
                        perturbed_model = common.visualization.weight_attack(model, attack_config.attack, attack_config.objective, testloader)
                        perturbed_models.append(perturbed_model)

                    for attempt in range(self.xattack_config.attempts):
                        model_file = os.path.join(attack_directory, 'perturbation%d%s' % (attempt, common.paths.STATE_EXT))
                        common.state.State.checkpoint(model_file, perturbed_models[attempt].cpu())
                        log('wrote %s' % model_file)
                else:
                    perturbed_models = []
                    for attempt in range(self.xattack_config.attempts):
                        model_file = os.path.join(attack_directory, 'perturbation%d%s' % (attempt, common.paths.STATE_EXT))
                        assert os.path.exists(model_file), model_file
                        perturbed_models.append(common.state.State.load(model_file).model)

                return perturbed_models

            perturbed_models = run_or_load(self.xattack_config)

            for attempt in range(self.xattack_config.attempts):
                direction, factor = common.visualization.weight_direction(model, perturbed_models[attempt], self.xattack_config.attack, testloader, normalization=self.normalization, cuda=cuda)
                # computes losses on adversarial examples creafted for each step individually
                log('%s attempt %d' % (self.xattack_config.directory, attempt))
                steps_, losses_, errors_ = common.visualization.adversarial_weight_1d(model, direction, self.input_attack_config.attack, self.input_attack_config.objective,
                                                                                      self.testloader, steps=numpy.linspace(-1, 1, self.steps), cuda=cuda)
                if self.hessian > 0:
                    steps_, eigs_ = common.visualization.adversarial_hessian_1d(model, direction, self.input_attack_config.attack, self.input_attack_config.objective,
                                                                                self.testloader, hessian_k=self.hessian, steps=numpy.linspace(-1, 1, self.steps), cuda=cuda)
                    eigs = common.numpy.concatenate(eigs, numpy.expand_dims(eigs_, 0))

                steps = common.numpy.concatenate(steps, numpy.expand_dims(steps_, 0))
                factors.append(factor)
                losses = common.numpy.concatenate(losses, numpy.expand_dims(losses_, 0))
                errors = common.numpy.concatenate(errors, numpy.expand_dims(errors_, 0))

            factors = numpy.array(factors)
            tensors = [steps, losses, errors, factors]
            keys = ['steps', 'losses', 'errors', 'factors']
            if eigs is not None:
                tensors.append(eigs)
                keys.append('eigs')
            common.utils.write_hdf5(visualization_file, tensors, keys)
            log('wrote %s' % visualization_file)
        else:
            log('found %s' % visualization_file)


class VisualizeWeights2DInterface:
    """
    Visualize using input attack.
    """

    def __init__(self, target_config, xattack_config, yattack_config, testloader, normalization=None, input_attack_config=None, steps=51, hessian=0):
        """
        Initialize.

        :param target_config: configuration
        :type target_config: [str]
        :param xattack_config: configuration
        :type xattack_config: [str]
        :param yattack_config: configuration
        :type yattack_config: [str]
        """

        assert isinstance(target_config, NormalTrainingConfig)
        assert isinstance(xattack_config, AttackWeightsConfig)
        assert isinstance(yattack_config, AttackWeightsConfig)
        assert input_attack_config is None or isinstance(input_attack_config, AttackConfig)

        self.target_config = target_config
        """ (NormalTrainingConfig) Config. """

        self.xattack_config = xattack_config
        """ (AttackConfig) Config. """

        self.yattack_config = yattack_config
        """ (AttackConfig) Config. """

        self.input_attack_config = input_attack_config
        """ (AttackConfig) Input attack config. """

        self.testloader = testloader
        """ (torch.utils.data.DataLoader) Dataloader. """

        self.normalization = normalization
        """ (str) Weight normalization. """

        assert steps%2 == 1
        self.steps = steps
        """ (int) Number of steps. """

        self.hessian = hessian
        """ (int) Compute hessian. """

    def main(self, force=False, force_attack=False):
        """
        Main.
        """

        cuda = self.target_config.cuda
        model_file = common.paths.experiment_file(self.target_config.directory, 'classifier', common.paths.STATE_EXT)
        if self.xattack_config.snapshot is not None:
            assert self.xattack_config.snapshot == self.yattack_config.snapshot
            model_file += '.%d' % self.xattack_config.snapshot

        assert os.path.exists(model_file), 'file %s not found' % model_file

        def get_norm(attack):
            normalization = ''
            norm = getattr(attack, 'norm', None)
            if norm is not None:
                if isinstance(norm, attacks.weights.norms.LInfNorm):
                    normalization = 'linf'
                elif isinstance(norm, attacks.weights.norms.L2Norm):
                    normalization = 'l2'
                log('normalization %s' % normalization)
            return normalization

        if self.normalization is None or self.normalization == '':
            self.normalization = get_norm(self.xattack_config.attack)
            assert self.normalization == get_norm(self.yattack_config.attack)
        log('weight normalization %s' % self.normalization)

        visualization_directory = common.paths.experiment_dir('%s/%s_%s_%s_visualization' % (self.target_config.directory, self.xattack_config.directory, self.yattack_config.directory, self.normalization))
        if self.input_attack_config is not None:
            visualization_directory += '/%s' % self.input_attack_config.directory
        visualization_file = os.path.join(visualization_directory, 'visualization%s' % common.paths.HDF5_EXT)
        if self.xattack_config.snapshot is not None:
            visualization_file += '.%d' % self.xattack_config.snapshot

        keys = ['xsteps', 'ysteps', 'losses', 'errors', 'xfactors', 'yfactors', 'similarities']
        if self.hessian > 0:
            keys.append('eigs')
        complete = True
        for key in keys:
            if not common.utils.check_hdf5(visualization_file, key):
                complete = False

        if force or not complete:

            state = common.state.State.load(model_file)
            model = state.model
            log(model.__class__)

            if cuda:
                model = model.cuda()

            model.eval()

            xsteps = None
            ysteps = None
            xfactors = []
            yfactors = []
            similarities = []
            losses = None
            errors = None
            eigs = None

            assert self.xattack_config.attempts == self.yattack_config.attempts
            if self.input_attack_config is not None:
                perturbations, _, _ = common.test.attack(model, self.testloader, self.input_attack_config.attack,
                                                                           self.input_attack_config.objective, attempts=1, cuda=cuda)
                perturbations = numpy.transpose(perturbations[0], (0, 2, 3, 1))
                testset = common.datasets.AdversarialDataset(self.testloader.dataset.images, perturbations,
                                                             self.testloader.dataset.labels)
                testloader = torch.utils.data.DataLoader(testset, batch_size=self.testloader.batch_size)
            else:
                testloader = self.testloader

            def run_or_load(attack_config):
                if attack_config.model_specific and attack_config.snapshot is not None:
                    attack_directory = common.paths.experiment_dir('%s/%s_visualization_%d' % (self.target_config.directory, attack_config.directory, attack_config.snapshot))
                else:
                    attack_directory = common.paths.experiment_dir('%s/%s_visualization' % (self.target_config.directory, attack_config.directory))
                attack_model_file = os.path.join(attack_directory, 'perturbation0%s' % common.paths.STATE_EXT)

                if not os.path.exists(attack_model_file) or force_attack:
                    perturbed_models = []
                    for attempt in range(self.xattack_config.attempts):
                        perturbed_model = common.visualization.weight_attack(model, self.xattack_config.attack, self.xattack_config.objective, testloader)
                        perturbed_models.append(perturbed_model)

                    for attempt in range(self.xattack_config.attempts):
                        model_file = os.path.join(attack_directory, 'perturbation%d%s' % (attempt, common.paths.STATE_EXT))
                        common.state.State.checkpoint(model_file, perturbed_models[attempt].cpu())
                        log('wrote %s' % model_file)
                else:
                    perturbed_models = []
                    for attempt in range(self.xattack_config.attempts):
                        model_file = os.path.join(attack_directory, 'perturbation%d%s' % (attempt, common.paths.STATE_EXT))
                        assert os.path.exists(model_file), model_file
                        perturbed_models.append(common.state.State.load(model_file).model)

                return perturbed_models

            xperturbed_models = run_or_load(self.xattack_config)
            yperturbed_models = run_or_load(self.yattack_config)

            for attempt in range(self.xattack_config.attempts):
                xdirection, xfactor = common.visualization.weight_direction(model, xperturbed_models[attempt], self.xattack_config.attack,
                                                                                        testloader, normalization=self.normalization, cuda=cuda)
                ydirection, yfactor = common.visualization.weight_direction(model, yperturbed_models[-attempt], self.yattack_config.attack,
                                                                                        testloader, normalization=self.normalization, cuda=cuda)

                xparameters = common.torch.all_parameters(xdirection)
                yparameters = common.torch.all_parameters(ydirection)
                similarity = torch.nn.functional.cosine_similarity(xparameters, yparameters, dim=0).item()
                similarities.append(similarity)

                xsteps_, ysteps_, losses_, errors_ = common.visualization.weight_2d(model, xdirection, ydirection, testloader,
                                                                           xsteps=numpy.linspace(-1, 1, self.steps),
                                                                           ysteps=numpy.linspace(-1, 1, self.steps), cuda=cuda)

                if self.hessian > 0:
                    _, _, eigs_ = common.visualization.hessian_2d(model, xdirection, ydirection, testloader, hessian_k=self.hessian,
                                                                    xsteps=numpy.linspace(-1, 1, self.steps),
                                                                    ysteps=numpy.linspace(-1, 1, self.steps), cuda=cuda)
                    eigs = common.numpy.concatenate(eigs, numpy.expand_dims(eigs_, 0))

                xsteps = common.numpy.concatenate(xsteps, numpy.expand_dims(xsteps_, 0))
                ysteps = common.numpy.concatenate(ysteps, numpy.expand_dims(ysteps_, 0))
                xfactors.append(xfactor)
                yfactors.append(yfactor)
                losses = common.numpy.concatenate(losses, numpy.expand_dims(losses_, 0))
                errors = common.numpy.concatenate(errors, numpy.expand_dims(errors_, 0))

            xfactors = numpy.array(xfactors)
            yfactors = numpy.array(yfactors)
            similarities = numpy.array(similarities)

            tensors = [xsteps, ysteps, losses, errors, xfactors, yfactors, similarities]
            keys = ['xsteps', 'ysteps', 'losses', 'errors', 'xfactors', 'yfactors', 'similarities']
            if eigs is not None:
                tensors.append(eigs)
                keys.append('eigs')
            common.utils.write_hdf5(visualization_file, tensors, keys)
            log('wrote %s' % visualization_file)
        else:
            log('found %s' % visualization_file)


class VisualizeWeightsInputs2DInterface:
    """
    Visualize using input attack.
    """

    def __init__(self, target_config, attack_config, testloader, weight_normalization=None, input_normalization=None, steps=51):
        """
        Initialize.

        :param target_config: configuration
        :type target_config: [str]
        :param attack_config: configuration
        :type attack_config: [str]
        """

        assert isinstance(target_config, NormalTrainingConfig)
        assert isinstance(attack_config, AttackWeightsInputsConfig)

        self.target_config = target_config
        """ (NormalTrainingConfig) Config. """

        self.attack_config = attack_config
        """ (AttackConfig) Config. """

        self.testloader = testloader
        """ (torch.utils.data.DataLoader) Dataloader. """

        self.weight_normalization = weight_normalization
        """ (str) Weight normalization. """

        self.input_normalization = input_normalization
        """ (str) Input normalization. """

        assert steps%2 == 1
        self.steps = steps
        """ (int) Number of steps. """

    def main(self, force=False, force_attack=False):
        """
        Main.
        """

        cuda = self.target_config.cuda
        attack = self.attack_config.attack
        weight_objective = self.attack_config.weight_objective
        input_objective = self.attack_config.input_objective

        model_file = common.paths.experiment_file(self.target_config.directory, 'classifier', common.paths.STATE_EXT)
        if self.attack_config.snapshot is not None:
            model_file += '.%d' % self.attack_config.snapshot

        assert os.path.exists(model_file), 'file %s not found' % model_file

        def get_input_norm(attack):
            input_normalization = ''
            input_norm = getattr(attack, 'input_norm', None)
            if input_norm is not None:
                if isinstance(input_norm, attacks.norms.LInfNorm):
                    input_normalization = 'linf'
                elif isinstance(input_norm, attacks.norms.L2Norm):
                    input_normalization = 'l2'
            return input_normalization
        def get_weight_norm(attack):
            weight_normalization = ''
            weight_norm = getattr(attack, 'weight_norm', None)
            if weight_norm is not None:
                if isinstance(weight_norm, attacks.weights.norms.LInfNorm):
                    weight_normalization = 'linf'
                elif isinstance(weight_norm, attacks.weights.norms.L2Norm):
                    weight_normalization = 'l2'
            return weight_normalization

        if self.input_normalization is None or self.input_normalization == '':
            self.input_normalization = get_input_norm(self.attack_config.attack)
        if self.weight_normalization is None or self.weight_normalization == '':
            self.weight_normalization = get_weight_norm(self.attack_config.attack)

        log('input_normalization %s' % self.input_normalization)
        log('weight_normalization %s' % self.weight_normalization)

        visualization_directory = common.paths.experiment_dir('%s/%s_%s_%s_visualization' % (self.target_config.directory, self.weight_normalization, self.input_normalization, self.attack_config.directory))
        visualization_file = os.path.join(visualization_directory, 'visualization%s' % common.paths.HDF5_EXT)
        if self.attack_config.snapshot is not None:
            visualization_file += '.%d' % self.attack_config.snapshot

        if force or not os.path.exists(visualization_file):

            state = common.state.State.load(model_file)
            model = state.model
            log(model.__class__)

            if cuda:
                model = model.cuda()

            model.eval()

            weight_scale_factors = []
            input_scale_factors = []
            weight_steps = []
            input_steps = []
            losses = []
            errors = []

            perturbations = None
            perturbed_models = []

            attack_directory = common.paths.experiment_dir('%s/%s_visualization' % (self.target_config.directory, self.attack_config.directory))
            attack_model_file = os.path.join(attack_directory, 'perturbation0%s' % common.paths.STATE_EXT)
            attack_perturbations_file = os.path.join(attack_directory, 'perturbations%s' % common.paths.HDF5_EXT)

            if not os.path.exists(attack_model_file) or not os.path.exists(attack_perturbations_file) or force_attack:
                for attempt in range(self.attack_config.attempts):
                    perturbed_model, perturbations_ = common.visualization.weight_input_attack(model, attack, weight_objective, input_objective, self.testloader)
                    perturbed_models.append(perturbed_model)
                    perturbations = common.numpy.concatenate(perturbations, numpy.expand_dims(perturbations_, axis=0))

                common.utils.write_hdf5(attack_perturbations_file, perturbations)
                for attempt in range(self.attack_config.attempts):
                    common.state.State.checkpoint(os.path.join(attack_directory, 'perturbation%d%s' % (attempt, common.paths.STATE_EXT)), perturbed_models[attempt].cpu())
            else:
                perturbations = common.utils.read_hdf5(attack_perturbations_file)
                perturbed_models = []
                for attempt in range(self.attack_config.attempts):
                    model_file = os.path.join(attack_directory, 'perturbation%d%s' % (attempt, common.paths.STATE_EXT))
                    assert os.path.exists(model_file)
                    perturbed_models.append(common.state.State.load(model_file).model)

            for attempt in range(self.attack_config.attempts):
                direction_model, weight_scale_factor, perturbations, inputs_as_numpy, labels_as_numpy, input_scale_factor = \
                    common.visualization.weight_input_direction(model, perturbed_models[attempt], perturbations, self.testloader,
                                                                            weight_normalization=self.weight_normalization,
                                                                            input_normalization=self.input_normalization,
                                                                            cuda=False)
                model_steps_mesh, input_steps_mesh, loss, error = common.visualization.weight_input_2d(model, direction_model,
                                                                                                perturbations,
                                                                                                self.testloader,
                                                                                                xsteps=numpy.linspace(-1, 1, self.steps),
                                                                                                ysteps=numpy.linspace(-1, 1, self.steps),
                                                                                                cuda=True)
                weight_scale_factors.append(weight_scale_factor)
                input_scale_factors.append(input_scale_factor)
                weight_steps.append(numpy.expand_dims(model_steps_mesh, axis=0))
                input_steps.append(numpy.expand_dims(input_steps_mesh, axis=0))
                losses.append(numpy.expand_dims(loss, axis=0))
                errors.append(numpy.expand_dims(error, axis=0))

            weight_scale_factors = numpy.array(weight_scale_factors)
            input_scale_factors = numpy.array(input_scale_factors)
            weight_steps = numpy.concatenate(tuple(weight_steps), axis=0)
            input_steps = numpy.concatenate(tuple(input_steps), axis=0)
            losses = numpy.concatenate(tuple(losses), axis=0)
            errors = numpy.concatente(tuple(errors), axis=0)

            common.utils.write_hdf5(visualization_file, [
                weight_steps, input_steps, losses, errors, weight_scale_factors, input_scale_factors
            ], [
                'weight_steps', 'input_steps', 'losses', 'errors', 'weight_factors', 'input_factors'
            ])
            log('wrote %s' % visualization_file)
        else:
            log('found %s' % visualization_file)