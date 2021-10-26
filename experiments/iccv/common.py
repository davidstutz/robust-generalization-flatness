import common.experiments
import common.utils
import common.numpy
import common.paths
import common.autoaugment
import common.torch
from common.log import log, LogLevel
import attacks
import attacks.weights
import attacks.weights_inputs
import torch
import torch.utils.tensorboard
import torchvision
import numpy
from imgaug import augmenters as iaa
import experiments.iccv.helper as helper
import copy
import models
helper.guard()


def get_augmentation():
    augmenters = []
    augmenters.append(iaa.CropAndPad(
        px=((0, 4), (0, 4), (0, 4), (0, 4)),
        pad_mode='constant',
        pad_cval=(0, 0),
    ))
    augmenters.append(iaa.Fliplr(0.5))
    return iaa.Sequential(augmenters)


def get_training_writer(log_dir, sub_dir=''):
    return common.summary.SummaryPickleWriter('%s/%s' % (log_dir, sub_dir))


def get_attack_writer(log_dir, sub_dir=''):
    return common.summary.SummaryWriter()


def get_l2_optimizer(model, lr=helper.lr, momentum=0.9, weight_decay=0.0005):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)


def get_adam_optimizer(model, lr=helper.lr, weight_decay=0.0005):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_entropy_optimizer(model, lr=helper.lr, weight_decay=0.0005, L=1, eps=1e-4, gamma=1e-4, scoping=1e-3):
    return common.torch.EntropySGD([parameter for parameter in model.parameters() if parameter.requires_grad is True], config=dict(lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay, L=L, eps=eps, g0=gamma, g1=scoping))


get_default_optimizer = get_l2_optimizer


def get_multi_step_scheduler(optimizer, batches_per_epoch, milestones=[2*helper.epochs//5, 3*helper.epochs//5, 4*helper.epochs//5], lr_factor=0.1):
    return common.train.get_multi_step_scheduler(optimizer, batches_per_epoch=batches_per_epoch, milestones=milestones, gamma=lr_factor)


def get_late_multi_step_scheduler(optimizer, batches_per_epoch, milestones=[140, 145], lr_factor=0.1):
    return common.train.get_multi_step_scheduler(optimizer, batches_per_epoch=batches_per_epoch, milestones=milestones, gamma=lr_factor)


def get_cyclic_scheduler(optimizer, batches_per_epoch, base_lr=helper.cyclic_min_lr, max_lr=helper.cyclic_max_lr, step_size_up=helper.cyclic_epochs/2, step_size_down=helper.cyclic_epochs/2):
    return common.train.get_cyclic_scheduler(optimizer, batches_per_epoch=batches_per_epoch, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, step_size_down=step_size_down)


def get_no_scheduler(optimizer, batches_per_epoch):
    return common.train.get_no_scheduler(optimizer, batches_per_epoch)


get_default_scheduler = get_multi_step_scheduler
finetune_epochs = (helper.epochs + 2*helper.epochs//5)
get_finetune_scheduler = common.utils.partial(get_multi_step_scheduler, milestones=[2*finetune_epochs//5, 3*finetune_epochs//5, 4*finetune_epochs//5])


def get_auxiliary_model(N_class, resolution):
    return models.MLP(N_class, resolution, units=[256])


cuda = True
batch_size = helper.batch_size
epochs = helper.epochs

trainset = helper.trainset()
testset = helper.testset()
testtrainset = helper.trainset(indices=numpy.array(list(range(10000))))
adversarialtrainbatch = helper.testset(indices=helper.test_N - 1 - numpy.array(list(range(100))))
adversarialtrainset = helper.testset(indices=helper.test_N - 1 - numpy.array(list(range(500))))
adversarialtestset = helper.testset(indices=numpy.array(list(range(1000)))) # 9000
adversarialtesttrainset = helper.trainset(indices=numpy.array(list(range(1000))))
adversarialtraintrainbatch = helper.trainset(indices=helper.train_N - 1 - numpy.array(list(range(100))))
adversarialtraintrainset = helper.trainset(indices=helper.train_N - 1 - numpy.array(list(range(500))))

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
testtrainloader = torch.utils.data.DataLoader(testtrainset, batch_size=batch_size, shuffle=False)
adversarialtrainsetloader = torch.utils.data.DataLoader(adversarialtrainset, batch_size=batch_size, shuffle=False)
adversarialtrainbatchloader = torch.utils.data.DataLoader(adversarialtrainbatch, batch_size=batch_size, shuffle=False)
adversarialtestloader = torch.utils.data.DataLoader(adversarialtestset, batch_size=batch_size, shuffle=False)
adversarialtesttrainloader = torch.utils.data.DataLoader(adversarialtesttrainset, batch_size, shuffle=False)
adversarialtraintrainbatchloader = torch.utils.data.DataLoader(adversarialtraintrainbatch, batch_size=batch_size, shuffle=False)
adversarialtraintrainsetloader = torch.utils.data.DataLoader(adversarialtraintrainset, batch_size=batch_size, shuffle=False)

if helper.unsupset:
    trainset = common.datasets.PseudoLabeledSemiSupervisedDataset(trainset, helper.unsupset())

assert not (helper.autoaugment and helper.augment)
if helper.autoaugment:
    log('loading AutoAugment')
    assert isinstance(helper.mean, list) and len(helper.mean) > 0 and helper.cutout > 0
    data_resolution = trainset.images.shape[1]
    assert trainset.images.shape[1] == trainset.images.shape[2]
    # has to be tensor
    data_mean = torch.tensor(helper.mean)
    # has to be tuple
    data_mean_int = []
    for c in range(data_mean.numel()):
        data_mean_int.append(int(255*data_mean[c]))
    data_mean_int = tuple(data_mean_int)
    trainset.transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda array: (array*255).astype(numpy.uint8)),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomCrop(data_resolution, padding=int(data_resolution*0.125), fill=data_mean_int),
        torchvision.transforms.RandomHorizontalFlip(),
        common.autoaugment.CIFAR10Policy(fillcolor=data_mean_int),
        torchvision.transforms.ToTensor(),
        common.torch.CutoutAfterToTensor(n_holes=1, length=helper.cutout, fill_color=data_mean),
        torchvision.transforms.Lambda(lambda array: array.permute(1, 2, 0)),
    ])
    # data loader takes care of augmentation, not trainer
    augmentation = None
elif helper.augment:
    log('loading regular augmentation')
    augmentation = get_augmentation()
    assert augmentation is not None
else:
    log('[Warning] no data augmentation', LogLevel.WARNING)
    augmentation = None

if helper.unsupset:
    rot_trainset_ = common.datasets.RotatedSemiSupervisedDataset(trainset, trainset)
    rot_trainsampler_ = common.datasets.SemiSupervisedSampler(rot_trainset_.sup_indices, rot_trainset_.unsup_indices, helper.batch_size, fraction=0.5, num_batches=int(numpy.ceil(len(trainset.sup_indices) / helper.batch_size)))
    rot_trainloader = torch.utils.data.DataLoader(rot_trainset_, batch_sampler=rot_trainsampler_)

    trainsampler = common.datasets.SemiSupervisedSampler(trainset.sup_indices, trainset.unsup_indices, helper.batch_size, 0.7, num_batches=int(numpy.ceil(len(trainset.sup_indices) / helper.batch_size)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=trainsampler)
    trainsampler_4 = common.datasets.SemiSupervisedSampler(trainset.sup_indices, trainset.unsup_indices, 4, 0.7, num_batches=int(numpy.ceil(len(trainset.sup_indices) / helper.batch_size)))
    trainloader_4 = torch.utils.data.DataLoader(trainset, batch_sampler=trainsampler_4)
    trainsampler_8 = common.datasets.SemiSupervisedSampler(trainset.sup_indices, trainset.unsup_indices, 8, 0.7, num_batches=int(numpy.ceil(len(trainset.sup_indices) / helper.batch_size)))
    trainloader_8 = torch.utils.data.DataLoader(trainset, batch_sampler=trainsampler_8)
    trainsampler_16 = common.datasets.SemiSupervisedSampler(trainset.sup_indices, trainset.unsup_indices, 16, 0.7, num_batches=int(numpy.ceil(len(trainset.sup_indices) / helper.batch_size)))
    trainloader_16 = torch.utils.data.DataLoader(trainset, batch_sampler=trainsampler_16)
    trainsampler_32 = common.datasets.SemiSupervisedSampler(trainset.sup_indices, trainset.unsup_indices, 32, 0.7, num_batches=int(numpy.ceil(len(trainset.sup_indices) / helper.batch_size)))
    trainloader_32 = torch.utils.data.DataLoader(trainset, batch_sampler=trainsampler_32)
    trainsampler_64 = common.datasets.SemiSupervisedSampler(trainset.sup_indices, trainset.unsup_indices, 64, 0.7, num_batches=int(numpy.ceil(len(trainset.sup_indices) / helper.batch_size)))
    trainloader_64 = torch.utils.data.DataLoader(trainset, batch_sampler=trainsampler_64)
    trainsampler_256 = common.datasets.SemiSupervisedSampler(trainset.sup_indices, trainset.unsup_indices, 256, 0.7, num_batches=int(numpy.ceil(len(trainset.sup_indices) / helper.batch_size)))
    trainloader_256 = torch.utils.data.DataLoader(trainset, batch_sampler=trainsampler_256)
    trainsampler_512 = common.datasets.SemiSupervisedSampler(trainset.sup_indices, trainset.unsup_indices, 512, 0.7, num_batches=int(numpy.ceil(len(trainset.sup_indices) / helper.batch_size)))
    trainloader_512 = torch.utils.data.DataLoader(trainset, batch_sampler=trainsampler_512)
else:
    rot_trainset_ = common.datasets.RotatedSemiSupervisedDataset(trainset, trainset)
    rot_trainsampler_ = common.datasets.SemiSupervisedSampler(rot_trainset_.sup_indices, rot_trainset_.unsup_indices, helper.batch_size, fraction=0.5, num_batches=int(numpy.ceil(len(rot_trainset_.sup_indices) / helper.batch_size)))
    rot_trainloader = torch.utils.data.DataLoader(rot_trainset_, batch_sampler=rot_trainsampler_)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    trainloader_4 = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
    trainloader_8 = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)
    trainloader_16 = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
    trainloader_32 = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    trainloader_64 = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    trainloader_256 = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
    trainloader_512 = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True)


def external_training_config(model_file, directory_name, directory, **kwargs):
    """
    Set a variable as a global variable with given name.

    :param directory: name of global
    :type directory: str
    """

    keys = set(kwargs.keys())
    assert keys.issubset(set([
        'base_directory', 'type',
    ])), keys

    config = common.experiments.ExternalTrainingConfig()
    config.model_file = model_file
    base_directory = kwargs.get('base_directory', helper.base_directory)
    config.directory = '%s/%s' % (base_directory, directory)
    config.epochs = epochs
    config.type = kwargs.get('type', 'Linf')
    assert config.type in ['Linf', 'L2']

    assert directory_name not in globals().keys(), directory_name
    globals()[directory_name] = config


def normal_training_config(directory, **kwargs):
    """
    Set a variable as a global variable with given name.

    :param directory: name of global
    :type directory: str
    """

    keys = set(kwargs.keys())
    assert keys.issubset(set([
        'cuda', 'augmentation', 'trainloader', 'testloader',
        'epochs', 'get_writer', 'get_optimizer',
        'projection', 'finetune', 'get_scheduler', 'snapshot',
        'lr', 'loss',
        'keep_average', 'keep_average_tau',
        'base_directory',
    ])), keys

    config = common.experiments.NormalTrainingConfig()
    config.cuda = kwargs.get('cuda', cuda)
    config.augmentation = kwargs.get('augmentation', augmentation)
    assert helper.augment is False or config.augmentation is not None
    config.trainloader = kwargs.get('trainloader', trainloader)
    config.testloader = kwargs.get('testloader', testloader)
    config.epochs = kwargs.get('epochs', epochs)
    config.get_writer = kwargs.get('get_writer', get_training_writer)
    config.get_optimizer = kwargs.get('get_optimizer', common.utils.partial(get_default_optimizer, lr=kwargs.get('lr', helper.lr)))
    config.projection = kwargs.get('projection', None)
    config.loss = kwargs.get('loss', common.torch.classification_loss)
    config.finetune = kwargs.get('finetune', None)
    config.get_scheduler = kwargs.get('get_scheduler', common.utils.partial(get_default_scheduler, batches_per_epoch=len(config.trainloader)))
    config.keep_average = kwargs.get('keep_average', False)
    config.keep_average_tau = kwargs.get('keep_average_tau', 0.9975)
    config.snapshot = kwargs.get('snapshot', 5)
    base_directory = kwargs.get('base_directory', helper.base_directory)
    config.directory = '%s/%s' % (base_directory, directory)
    config.interface = common.experiments.NormalTrainingInterface

    assert directory not in globals().keys(), directory
    globals()[directory] = config
    globals()[directory + '_es'] = copy.copy(config)
    globals()[directory + '_es'].directory += '_es'


def adversarial_training_config(directory, **kwargs):
    """
    Set a variable as a global variable with given name.

    :param directory: name of global
    :type directory: str
    """

    keys = set(kwargs.keys())
    assert keys.issubset(set([
        'cuda', 'augmentation', 'trainloader', 'testloader',
        'epochs', 'get_writer', 'get_optimizer',
        'projection', 'finetune', 'get_scheduler', 'snapshot',
        'lr', 'loss', 'attack', 'objective', 'summary_weights',
        'fraction', 'prevent_label_leaking', 'ignore_incorrect',
        'keep_average', 'keep_average_tau', 'base_directory',
        'interface',
    ])), keys

    config = common.experiments.AdversarialTrainingConfig()
    config.cuda = kwargs.get('cuda', cuda)
    config.augmentation = kwargs.get('augmentation', augmentation)
    assert helper.augment is False or config.augmentation is not None
    config.trainloader = kwargs.get('trainloader', trainloader)
    config.testloader = kwargs.get('testloader', testloader)
    config.epochs = kwargs.get('epochs', epochs)
    config.get_writer = kwargs.get('get_writer', get_training_writer)
    config.get_optimizer = kwargs.get('get_optimizer', common.utils.partial(get_default_optimizer, lr=kwargs.get('lr', helper.lr)))
    config.projection = kwargs.get('projection', None)
    config.loss = kwargs.get('loss', common.torch.classification_loss)
    config.attack = kwargs.get('attack', None)
    assert config.attack is not None
    config.objective = kwargs.get('objective', attacks.objectives.UntargetedF0Objective())
    assert config.objective is not None
    config.fraction = kwargs.get('fraction', 0.5)
    config.prevent_label_leaking = kwargs.get('prevent_label_leaking', False)
    config.ignore_incorrect = kwargs.get('ignore_incorrect', False)
    config.keep_average = kwargs.get('keep_average', False)
    config.keep_average_tau = kwargs.get('keep_average_tau', 0.9975)
    finetune = kwargs.get('finetune', None)
    if finetune is not None:
        config.finetune = '%s/%s' % (helper.base_directory, finetune)
    config.get_scheduler = kwargs.get('get_scheduler', common.utils.partial(get_default_scheduler, batches_per_epoch=len(config.trainloader)))
    config.snapshot = kwargs.get('snapshot', 5)
    base_directory = kwargs.get('base_directory', helper.base_directory)
    config.directory = '%s/%s' % (base_directory, directory)
    config.summary_weights = kwargs.get('summary_weights', False)
    config.interface = kwargs.get('interface', common.experiments.AdversarialTrainingInterface)

    assert directory not in globals().keys(), directory
    globals()[directory] = config
    globals()[directory + '_es'] = copy.copy(config)
    globals()[directory + '_es'].directory += '_es'


def adversarial_semi_supervised_training_config(directory, **kwargs):
    """
    Set a variable as a global variable with given name.

    :param directory: name of global
    :type directory: str
    """

    keys = set(kwargs.keys())
    assert keys.issubset(set([
        'cuda', 'augmentation', 'trainloader', 'testloader',
        'epochs', 'get_writer', 'get_optimizer',
        'projection', 'finetune', 'get_scheduler', 'snapshot',
        'lr', 'loss', 'attack', 'objective', 'summary_weights',
        'get_auxiliary_model', 'fraction', 'unsup_weight', 'base_directory',
    ])), keys

    config = common.experiments.AdversarialSemiSupervisedTrainingConfig()
    config.cuda = kwargs.get('cuda', cuda)
    config.augmentation = kwargs.get('augmentation', augmentation)
    assert helper.augment is False or config.augmentation is not None
    config.trainloader = kwargs.get('trainloader', None)
    assert config.trainloader is not None
    config.testloader = kwargs.get('testloader', testloader)
    config.epochs = kwargs.get('epochs', epochs)
    config.get_scheduler = kwargs.get('get_scheduler', common.utils.partial(get_default_scheduler, milestones=[2 * config.epochs // 5, 3 * config.epochs // 5, 4 * config.epochs // 5], batches_per_epoch=len(config.trainloader)))
    config.get_writer = kwargs.get('get_writer', get_training_writer)
    config.get_optimizer = kwargs.get('get_optimizer', common.utils.partial(get_default_optimizer, lr=kwargs.get('lr', helper.lr)))
    config.projection = kwargs.get('projection', None)
    config.loss = kwargs.get('loss', common.torch.classification_loss)
    config.unsup_weight = kwargs.get('unsup_weight', 1)
    config.unsup_loss = kwargs.get('unsup_loss', common.torch.classification_loss)
    config.get_auxiliary_model = kwargs.get('get_auxiliary_model', get_auxiliary_model)
    config.attack = kwargs.get('attack', None)
    assert config.attack is not None
    config.objective = kwargs.get('objective', attacks.objectives.UntargetedF0Objective())
    assert config.objective is not None
    config.fraction = kwargs.get('fraction', 0.5)
    finetune = kwargs.get('finetune', None)
    if finetune is not None:
        config.finetune = '%s/%s' % (helper.base_directory, finetune)
    config.snapshot = kwargs.get('snapshot', 5)
    base_directory = kwargs.get('base_directory', helper.base_directory)
    config.directory = '%s/%s' % (base_directory, directory)
    config.summary_weights = kwargs.get('summary_weights', False)
    config.interface = common.experiments.AdversarialSemiSupervisedTrainingInterface

    assert directory not in globals().keys(), directory
    globals()[directory] = config
    globals()[directory + '_es'] = copy.copy(config)
    globals()[directory + '_es'].directory += '_es'


def adversarial_weights_inputs_training_config(directory, **kwargs):
    """
    Set a variable as a global variable with given name.

    :param directory: name of global
    :type directory: str
    """

    keys = set(kwargs.keys())
    assert keys.issubset(set([
        'cuda', 'augmentation', 'trainloader', 'testloader',
        'epochs', 'get_writer', 'get_optimizer',
        'projection', 'finetune', 'get_scheduler', 'snapshot',
        'weight_attack', 'weight_objective', 'curriculum',
        'input_attack', 'input_objective',
        'lr', 'loss', 'base_directory',
    ])), keys

    config = common.experiments.AdversarialWeightsInputsTrainingConfig()
    config.weight_attack = kwargs.get('weight_attack', None)
    assert config.weight_attack is not None
    config.weight_objective = kwargs.get('weight_objective', attacks.weights.objectives.UntargetedF0Objective())
    assert config.weight_objective is not None
    config.input_attack = kwargs.get('input_attack', None)
    assert config.input_attack is not None
    config.input_objective = kwargs.get('input_objective', attacks.objectives.UntargetedF0Objective())
    assert config.input_objective is not None
    config.curriculum = kwargs.get('curriculum', None)
    config.cuda = kwargs.get('cuda', cuda)
    config.augmentation = kwargs.get('augmentation', augmentation)
    assert helper.augment is False or config.augmentation is not None
    config.trainloader = kwargs.get('trainloader', trainloader)
    config.testloader = kwargs.get('testloader', testloader)
    config.epochs = kwargs.get('epochs', epochs)
    config.get_writer = kwargs.get('get_writer', get_training_writer)
    config.get_optimizer = kwargs.get('get_optimizer', common.utils.partial(get_default_optimizer, lr=kwargs.get('lr', helper.lr)))
    config.projection = kwargs.get('projection', None)
    config.loss = kwargs.get('loss', common.torch.classification_loss)
    config.finetune = kwargs.get('finetune', None)
    config.get_scheduler = kwargs.get('get_scheduler', common.utils.partial(get_default_scheduler, batches_per_epoch=len(config.trainloader)))
    config.snapshot = kwargs.get('snapshot', 5)
    base_directory = kwargs.get('base_directory', helper.base_directory)
    config.directory = '%s/%s' % (base_directory, directory)

    config.interface = common.experiments.AdversarialWeightsInputsTrainingInterface

    assert directory not in globals().keys(), directory
    globals()[directory] = config
    globals()[directory + '_es'] = copy.copy(config)
    globals()[directory + '_es'].directory += '_es'


def attack_config(directory, **kwargs):
    """
    Set a variable as a global variable with given name.

    :param directory: name of global
    :type directory: str
    """

    keys = set(kwargs.keys())
    assert keys.issubset(set([
        'testloader', 'attack', 'objective', 'attempts', 'get_writer', 'model_specific',
    ]))

    config = common.experiments.AttackConfig()
    config.testloader = kwargs.get('testloader', adversarialtestloader)
    assert config.testloader is not None
    config.attack = kwargs.get('attack', None)
    assert config.attack is not None
    config.objective = kwargs.get('objective', attacks.objectives.UntargetedF0Objective())
    assert config.objective is not None
    config.attempts = kwargs.get('attempts', 1)
    config.model_specific = kwargs.get('model_specific', None)
    config.get_writer = kwargs.get('get_writer', get_attack_writer)
    config.directory = directory
    config.interface = common.experiments.AttackInterface

    assert directory not in globals().keys(), directory
    globals()[directory] = config


def attack_weights_config(directory, **kwargs):
    """
    Set a variable as a global variable with given name.

    :param directory: name of global
    :type directory: str
    """

    keys = set(kwargs.keys())
    assert keys.issubset(set([
        'trainloader', 'testloader', 'attack', 'objective', 'attempts', 'get_writer', 'eval',
        'operators', 'model_specific',
    ]))

    config = common.experiments.AttackWeightsConfig()
    config.trainloader = kwargs.get('trainloader', None)
    assert config.trainloader is not None
    config.testloader = kwargs.get('testloader', adversarialtestloader)
    assert config.testloader is not None
    config.attack = kwargs.get('attack', None)
    assert config.attack is not None
    config.objective = kwargs.get('objective', attacks.weights.objectives.UntargetedF0Objective())
    assert config.objective is not None
    config.attempts = kwargs.get('attempts', 1)
    config.eval = kwargs.get('eval', True)
    config.operators = kwargs.get('operators', None)
    config.model_specific = kwargs.get('model_specific', None)
    config.get_writer = kwargs.get('get_writer', get_attack_writer)
    config.directory = directory
    config.interface = common.experiments.AttackWeightsInterface

    assert directory not in globals().keys(), directory
    globals()[directory] = config


def attack_weights_inputs_config(directory, **kwargs):
    """
    Set a variable as a global variable with given name.

    :param directory: name of global
    :type directory: str
    """

    keys = set(kwargs.keys())
    assert keys.issubset(set([
        'testloader', 'attack', 'weight_objective', 'input_objective', 'attempts', 'get_writer'
    ]))

    config = common.experiments.AttackWeightsInputsConfig()
    config.testloader = kwargs.get('testloader', adversarialtestloader)
    assert config.testloader is not None
    config.attack = kwargs.get('attack', None)
    assert config.attack is not None
    config.weight_objective = kwargs.get('weight_objective', attacks.weights.objectives.UntargetedF0Objective())
    assert config.weight_objective is not None
    config.input_objective = kwargs.get('input_objective', attacks.objectives.UntargetedF0Objective())
    assert config.input_objective is not None
    config.attempts = kwargs.get('attempts', 1)
    config.get_writer = kwargs.get('get_writer', get_attack_writer)
    config.directory = directory
    config.interface = common.experiments.AttackWeightsInputsInterface

    assert directory not in globals().keys(), directory
    globals()[directory] = config


def gformat(value):
    """
    Format value for directory name.

    :param value: value
    :type value: float
    :return: str
    :rtype: str
    """

    return ('%.7f' % float(value)).rstrip('0').replace('.', '')


adversarial_linf_epsilons = [0.0235, 0.0275, 0.0314, 0.0352]
#
random_l2_error_rates = [1, 0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1, 0.05]
adversarial_l2_error_rates = [0.05, 0.01, 0.0075, 0.005, 0.004, 0.003, 0.0025, 0.002, 0.001, 0.00075, 0.0005, 0.00025, 0.0001]
adversarial_l2_absolute_error_rates = [0.01, 0.025, 0.05, 0.25, 0.5, 1, 5]
l2_error_rates = random_l2_error_rates + adversarial_l2_error_rates


def no_normalization(model):
    exclude_layers = ['whiten', 'rebn', 'norm', 'downsample.1', 'bn', 'shortcut.1']
    n_parameters = len(list(model.parameters()))
    parameters = dict(model.named_parameters())
    names = list(parameters.keys())
    assert len(names) == n_parameters
    layers = []
    for i in range(len(names)):
        if parameters[names[i]].requires_grad:
            exclude = False
            for e in exclude_layers:
                if names[i].find(e) >= 0:
                    exclude = True
            if not exclude:
                layers.append(i)
    return layers


def no_normalization_no_bias(model):
    exclude_layers = ['whiten', 'rebn', 'norm', 'downsample.1', 'bn', 'shortcut.1', 'bias']
    n_parameters = len(list(model.parameters()))
    parameters = dict(model.named_parameters())
    names = list(parameters.keys())
    assert len(names) == n_parameters
    layers = []
    for i in range(len(names)):
        if parameters[names[i]].requires_grad:
            exclude = False
            for e in exclude_layers:
                if names[i].find(e) >= 0:
                    exclude = True
            if not exclude:
                layers.append(i)
    return layers

for adversarial_l2_error_rate in random_l2_error_rates:
    for attempts in [1, 3, 5, 10, 25, 50]:
        attack = attacks.weights.RandomAttack()
        attack.epochs = 1
        attack.initialization = attacks.weights.initializations.LayerWiseL2UniformSphereInitialization(adversarial_l2_error_rate)
        attack.norm = attacks.weights.norms.L2Norm()
        attack.projection = None
        attack_weights_config('weight_l2_random_e%s_at%d' % (gformat(adversarial_l2_error_rate), attempts),
                              attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                              attempts=attempts, trainloader=adversarialtrainbatchloader)
        attack_weights_config('weight_l2_random_e%s_at%d_train' % (gformat(adversarial_l2_error_rate), attempts),
                              attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                              attempts=attempts, trainloader=adversarialtraintrainbatchloader,
                              testloader=adversarialtesttrainloader)

        attack = attacks.weights.RandomAttack()
        attack.epochs = 1
        attack.initialization = attacks.weights.initializations.LayerWiseL2UniformSphereInitialization(adversarial_l2_error_rate)
        attack.norm = attacks.weights.norms.L2Norm()
        attack.projection = None
        attack.get_layers = no_normalization
        attack_weights_config('weight_l2_random_nonorm2_e%s_at%d' % (gformat(adversarial_l2_error_rate), attempts),
                              attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                              attempts=attempts, trainloader=adversarialtrainbatchloader)
        attack_weights_config('weight_l2_random_nonorm2_e%s_at%d_test' % (gformat(adversarial_l2_error_rate), attempts),
                              attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                              attempts=attempts, trainloader=adversarialtestloader, testloader=adversarialtestloader)
        attack_weights_config('weight_l2_random_nonorm2_e%s_at%d_es' % (gformat(adversarial_l2_error_rate), attempts),
                              attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                              attempts=attempts, trainloader=adversarialtrainsetloader, testloader=adversarialtrainsetloader)
        attack_weights_config('weight_l2_random_nonorm2_e%s_at%d_train' % (gformat(adversarial_l2_error_rate), attempts),
                              attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                              attempts=attempts, trainloader=adversarialtraintrainbatchloader,
                              testloader=adversarialtesttrainloader)

        attack = attacks.weights.RandomAttack()
        attack.epochs = 1
        attack.initialization = attacks.weights.initializations.LayerWiseL2UniformSphereInitialization(adversarial_l2_error_rate)
        attack.norm = attacks.weights.norms.L2Norm()
        attack.projection = None
        attack.get_layers = no_normalization
        attack_weights_config('weight_l2_random_nonorm2_e%s_at%d_set' % (gformat(adversarial_l2_error_rate), attempts),
                              attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                              attempts=attempts, trainloader=adversarialtrainsetloader)
        attack_weights_config('weight_l2_random_nonorm2_e%s_at%d_train_set' % (gformat(adversarial_l2_error_rate), attempts),
                              attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                              attempts=attempts, trainloader=adversarialtraintrainsetloader,
                              testloader=adversarialtesttrainloader)

        attack = attacks.weights.RandomAttack()
        attack.epochs = 1
        attack.initialization = attacks.weights.initializations.FilterWiseL2UniformSphereInitialization(adversarial_l2_error_rate)
        attack.norm = attacks.weights.norms.L2Norm()
        attack.projection = None
        attack.get_layers = no_normalization
        attack_weights_config('weight_l2_random_nonorm2_filter_e%s_at%d' % (gformat(adversarial_l2_error_rate), attempts),
                              attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                              attempts=attempts, trainloader=adversarialtrainbatchloader)
        attack_weights_config('weight_l2_random_nonorm2_filter_e%s_at%d_test' % (gformat(adversarial_l2_error_rate), attempts),
                            attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                            attempts=attempts, trainloader=adversarialtestloader, testloader=adversarialtestloader)
        attack_weights_config('weight_l2_random_nonorm2_filter_e%s_at%d_train' % (gformat(adversarial_l2_error_rate), attempts),
                              attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                              attempts=attempts, trainloader=adversarialtraintrainbatchloader,
                              testloader=adversarialtesttrainloader)

        attack = attacks.weights.RandomAttack()
        attack.epochs = 1
        attack.initialization = attacks.weights.initializations.L2UniformNormInitialization(relative_epsilon=adversarial_l2_error_rate)
        attack.norm = attacks.weights.norms.L2Norm()
        attack.projection = None
        attack.get_layers = no_normalization
        attack_weights_config('weight_l2_random_nonorm2_relative_e%s_at%d' % (gformat(adversarial_l2_error_rate), attempts),
                              attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                              attempts=attempts, trainloader=adversarialtrainbatchloader)
        attack_weights_config('weight_l2_random_nonorm2_relative_e%s_at%d_train' % (gformat(adversarial_l2_error_rate), attempts),
                              attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                              attempts=attempts, trainloader=adversarialtraintrainbatchloader,
                              testloader=adversarialtesttrainloader)


for adversarial_l2_error_rate in adversarial_l2_error_rates:
    for attempts in [1, 3, 5, 10, 25]:
        for iterations in [7, 10, 20]:
            for learning_rate in [0.0001, 0.001, 0.01, 0.1, 1, 10]:
                for lr_decay in [2]:
                    for normalization, normalization_name in zip([
                        attacks.weights.normalizations.L2Normalization(),
                        attacks.weights.normalizations.RelativeL2Normalization(),
                        attacks.weights.normalizations.LayerWiseRelativeL2Normalization(),
                        None
                    ], [
                        '_l2normalized',
                        '_rl2normalized',
                        '_lwrl2normalized',
                        ''
                    ]):
                        for backtrack, backtrack_name in zip([True, False], ['_backtrack', '']):
                            for momentum in [0, 0.9]:
                                attack = attacks.weights.GradientDescentAttack()
                                attack.epochs = iterations
                                attack.base_lr = learning_rate
                                attack.normalization = normalization
                                attack.backtrack = backtrack
                                attack.momentum = momentum
                                attack.lr_factor = lr_decay
                                attack.initialization = attacks.weights.initializations.LayerWiseL2UniformNormInitialization(adversarial_l2_error_rate)
                                attack.norm = attacks.weights.norms.L2Norm()
                                attack.projection = attacks.weights.SequentialProjections([
                                    attacks.weights.projections.LayerWiseL2Projection(adversarial_l2_error_rate)
                                ])
                                attack_weights_config(
                                    'weight_l2_gd%s%s_i%d_lr%s_mom%s_e%s_at%d_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtrainbatchloader, testloader=adversarialtestloader,
                                    model_specific=True)
                                attack_weights_config(
                                    'weight_l2_gd%s%s_i%d_lr%s_mom%s_e%s_at%d_train_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtraintrainbatchloader,
                                    testloader=adversarialtesttrainloader, model_specific=True)

                                attack = attacks.weights.GradientDescentAttack()
                                attack.epochs = iterations
                                attack.base_lr = learning_rate
                                attack.normalization = normalization
                                attack.backtrack = backtrack
                                attack.momentum = momentum
                                attack.lr_factor = lr_decay
                                attack.initialization = attacks.weights.initializations.LayerWiseL2UniformNormInitialization(adversarial_l2_error_rate)
                                attack.norm = attacks.weights.norms.L2Norm()
                                attack.projection = attacks.weights.SequentialProjections([
                                    attacks.weights.projections.LayerWiseL2Projection(adversarial_l2_error_rate)
                                ])
                                attack.get_layers = no_normalization
                                attack_weights_config(
                                    'weight_l2_gd_nonorm2%s%s_i%d_lr%s_mom%s_e%s_at%d_test_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtestloader, testloader=adversarialtestloader,
                                    model_specific=True)
                                attack_weights_config(
                                    'weight_l2_gd_nonorm2%s%s_i%d_lr%s_mom%s_e%s_at%d_es' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtrainsetloader, testloader=adversarialtrainsetloader,
                                    model_specific=True)
                                attack_weights_config(
                                    'weight_l2_gd_nonorm2%s%s_i%d_lr%s_mom%s_e%s_at%d_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtrainbatchloader, testloader=adversarialtestloader,
                                    model_specific=True)
                                attack_weights_config(
                                    'weight_l2_gd_nonorm2%s%s_i%d_lr%s_mom%s_e%s_at%d_train_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtraintrainbatchloader,
                                    testloader=adversarialtesttrainloader, model_specific=True)

                                attack = attacks.weights.GradientDescentAttack()
                                attack.epochs = iterations
                                attack.base_lr = learning_rate
                                attack.normalization = normalization
                                attack.backtrack = backtrack
                                attack.momentum = momentum
                                attack.lr_factor = lr_decay
                                attack.initialization = attacks.weights.initializations.LayerWiseL2UniformNormInitialization(
                                    adversarial_l2_error_rate)
                                attack.norm = attacks.weights.norms.L2Norm()
                                attack.projection = attacks.weights.SequentialProjections([
                                    attacks.weights.projections.LayerWiseL2Projection(adversarial_l2_error_rate)
                                ])
                                attack.get_layers = no_normalization

                                attack_weights_config(
                                    'weight_l2_gd_nonorm2%s%s_i%d_lr%s_mom%s_e%s_at%d_test_set' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtrainsetloader, testloader=adversarialtestloader,
                                    model_specific=True)
                                attack_weights_config(
                                    'weight_l2_gd_nonorm2%s%s_i%d_lr%s_mom%s_e%s_at%d_train_test_set' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtrainsetloader,
                                    testloader=adversarialtesttrainloader, model_specific=True)

                                attack = attacks.weights.GradientDescentAttack()
                                attack.epochs = iterations
                                attack.base_lr = learning_rate
                                attack.normalization = normalization
                                attack.backtrack = backtrack
                                attack.momentum = momentum
                                attack.lr_factor = lr_decay
                                attack.initialization = attacks.weights.initializations.FilterWiseL2UniformNormInitialization(adversarial_l2_error_rate)
                                attack.norm = attacks.weights.norms.L2Norm()
                                attack.projection = attacks.weights.SequentialProjections([
                                    attacks.weights.projections.FilterWiseL2Projection(adversarial_l2_error_rate)
                                ])
                                attack_weights_config(
                                    'weight_l2_gd_filter%s%s_i%d_lr%s_mom%s_e%s_at%d_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtrainbatchloader, testloader=adversarialtestloader,
                                    model_specific=True)
                                attack_weights_config(
                                    'weight_l2_gd_filter%s%s_i%d_lr%s_mom%s_e%s_at%d_train_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtraintrainbatchloader,
                                    testloader=adversarialtesttrainloader, model_specific=True)

                                attack = attacks.weights.GradientDescentAttack()
                                attack.epochs = iterations
                                attack.base_lr = learning_rate
                                attack.normalization = normalization
                                attack.backtrack = backtrack
                                attack.momentum = momentum
                                attack.lr_factor = lr_decay
                                attack.initialization = attacks.weights.initializations.FilterWiseL2UniformNormInitialization(adversarial_l2_error_rate)
                                attack.norm = attacks.weights.norms.L2Norm()
                                attack.projection = attacks.weights.SequentialProjections([
                                    attacks.weights.projections.FilterWiseL2Projection(adversarial_l2_error_rate)
                                ])
                                attack.get_layers = no_normalization
                                attack_weights_config(
                                    'weight_l2_gd_nonorm2_filter%s%s_i%d_lr%s_mom%s_e%s_at%d_test_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtestloader, testloader=adversarialtestloader,
                                    model_specific=True)
                                attack_weights_config(
                                    'weight_l2_gd_nonorm2_filter%s%s_i%d_lr%s_mom%s_e%s_at%d_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtrainbatchloader, testloader=adversarialtestloader,
                                    model_specific=True)
                                attack_weights_config(
                                    'weight_l2_gd_nonorm2_filter%s%s_i%d_lr%s_mom%s_e%s_at%d_train_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtraintrainbatchloader,
                                    testloader=adversarialtesttrainloader, model_specific=True)

                                attack = attacks.weights.GradientDescentAttack()
                                attack.epochs = iterations
                                attack.base_lr = learning_rate
                                attack.normalization = normalization
                                attack.backtrack = backtrack
                                attack.momentum = momentum
                                attack.lr_factor = lr_decay
                                attack.initialization = attacks.weights.initializations.L2UniformNormInitialization(relative_epsilon=adversarial_l2_error_rate)
                                attack.norm = attacks.weights.norms.L2Norm()
                                attack.projection = attacks.weights.SequentialProjections([
                                    attacks.weights.projections.L2Projection(relative_epsilon=adversarial_l2_error_rate)
                                ])
                                attack_weights_config(
                                    'weight_l2_gd_relative%s%s_i%d_lr%s_mom%s_e%s_at%d_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtrainbatchloader, testloader=adversarialtestloader,
                                    model_specific=True)
                                attack_weights_config(
                                    'weight_l2_gd_relative%s%s_i%d_lr%s_mom%s_e%s_at%d_train_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtraintrainbatchloader,
                                    testloader=adversarialtesttrainloader, model_specific=True)

                                attack = attacks.weights.GradientDescentAttack()
                                attack.epochs = iterations
                                attack.base_lr = learning_rate
                                attack.normalization = normalization
                                attack.backtrack = backtrack
                                attack.momentum = momentum
                                attack.lr_factor = lr_decay
                                attack.initialization = attacks.weights.initializations.L2UniformNormInitialization(relative_epsilon=adversarial_l2_error_rate)
                                attack.norm = attacks.weights.norms.L2Norm()
                                attack.projection = attacks.weights.SequentialProjections([
                                    attacks.weights.projections.L2Projection(relative_epsilon=adversarial_l2_error_rate)
                                ])
                                attack.get_layers = no_normalization
                                attack_weights_config(
                                    'weight_l2_gd_nonorm2_relative%s%s_i%d_lr%s_mom%s_e%s_at%d_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtrainbatchloader, testloader=adversarialtestloader,
                                    model_specific=True)
                                attack_weights_config(
                                    'weight_l2_gd_nonorm2_relative%s%s_i%d_lr%s_mom%s_e%s_at%d_train_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtraintrainbatchloader,
                                    testloader=adversarialtesttrainloader, model_specific=True)

for adversarial_l2_absolute_error_rate in adversarial_l2_absolute_error_rates:
    for attempts in [1, 3, 5, 10, 25]:
        for iterations in [7, 10, 20]:
            for learning_rate in [0.0001, 0.001, 0.01, 0.1, 1, 10]:
                for lr_decay in [2]:
                    for normalization, normalization_name in zip([
                        attacks.weights.normalizations.L2Normalization(),
                        attacks.weights.normalizations.RelativeL2Normalization(),
                        attacks.weights.normalizations.LayerWiseRelativeL2Normalization(),
                        None
                    ], [
                        '_l2normalized',
                        '_rl2normalized',
                        '_lwrl2normalized',
                        ''
                    ]):
                        for backtrack, backtrack_name in zip([True, False], ['_backtrack', '']):
                            for momentum in [0, 0.9]:
                                attack = attacks.weights.GradientDescentAttack()
                                attack.epochs = iterations
                                attack.base_lr = learning_rate
                                attack.normalization = normalization
                                attack.backtrack = backtrack
                                attack.momentum = momentum
                                attack.lr_factor = lr_decay
                                attack.initialization = attacks.weights.initializations.L2UniformNormInitialization(adversarial_l2_absolute_error_rate)
                                attack.norm = attacks.weights.norms.L2Norm()
                                attack.projection = attacks.weights.SequentialProjections([
                                    attacks.weights.projections.L2Projection(adversarial_l2_absolute_error_rate)
                                ])
                                attack_weights_config(
                                    'weight_l2_gd_absolute%s%s_i%d_lr%s_mom%s_e%s_at%d_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_absolute_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtrainbatchloader, testloader=adversarialtestloader,
                                    model_specific=True)
                                attack_weights_config(
                                    'weight_l2_gd_absolute%s%s_i%d_lr%s_mom%s_e%s_at%d_train_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_absolute_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtraintrainbatchloader,
                                    testloader=adversarialtesttrainloader, model_specific=True)

                                attack = attacks.weights.GradientDescentAttack()
                                attack.epochs = iterations
                                attack.base_lr = learning_rate
                                attack.normalization = normalization
                                attack.backtrack = backtrack
                                attack.momentum = momentum
                                attack.lr_factor = lr_decay
                                attack.initialization = attacks.weights.initializations.L2UniformNormInitialization(adversarial_l2_absolute_error_rate)
                                attack.norm = attacks.weights.norms.L2Norm()
                                attack.projection = attacks.weights.SequentialProjections([
                                    attacks.weights.projections.L2Projection(adversarial_l2_absolute_error_rate)
                                ])
                                attack.get_layers = no_normalization
                                attack_weights_config(
                                    'weight_l2_gd_nonorm2_absolute%s%s_i%d_lr%s_mom%s_e%s_at%d_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_absolute_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtrainbatchloader, testloader=adversarialtestloader,
                                    model_specific=True)
                                attack_weights_config(
                                    'weight_l2_gd_nonorm2_absolute%s%s_i%d_lr%s_mom%s_e%s_at%d_train_test' % (
                                        normalization_name,
                                        backtrack_name,
                                        iterations,
                                        gformat(learning_rate),
                                        gformat(momentum),
                                        gformat(adversarial_l2_absolute_error_rate),
                                        attempts,
                                    ), attack=attack, objective=attacks.weights.objectives.UntargetedF0Objective(),
                                    attempts=attempts,
                                    trainloader=adversarialtraintrainbatchloader,
                                    testloader=adversarialtesttrainloader, model_specific=True)

#
# linf input
#

for adversarial_linf_epsilon in adversarial_linf_epsilons:
    for attempts in [1, 3, 5, 10, 25]:
        for iterations in [7, 10, 20]:
            for learning_rate in [0.007]:
                for momentum in [0, 0.9]:
                    for backtrack, backtrack_name in zip([False], ['']):
                        for normalized, normalized_name in zip([True], ['_normalized']):
                            attack = attacks.BatchGradientDescent()
                            attack.max_iterations = iterations
                            attack.base_lr = learning_rate
                            attack.momentum = momentum
                            attack.lr_factor = 2
                            attack.backtrack = backtrack
                            attack.normalized = normalized
                            attack.c = 0
                            attack.initialization = attacks.initializations.LInfUniformNormInitialization(adversarial_linf_epsilon)
                            attack.projection = attacks.projections.SequentialProjections([
                                attacks.projections.BoxProjection(0, 1),
                                attacks.projections.LInfProjection(adversarial_linf_epsilon)
                            ])
                            attack.norm = attacks.norms.LInfNorm()
                            attack_config(
                                'input_linf_gd%s%s_lr%s_mom%s_i%d_e%s_at%d' % (
                                    normalized_name,
                                    backtrack_name,
                                    gformat(learning_rate),
                                    gformat(momentum),
                                    iterations,
                                    gformat(adversarial_linf_epsilon),
                                    attempts,
                                ), attack=attack, objective=attacks.objectives.UntargetedF0Objective(), attempts=attempts,
                                model_specific=True)
                            attack_config(
                                'input_linf_gd%s%s_lr%s_mom%s_i%d_e%s_at%d_es' % (
                                    normalized_name,
                                    backtrack_name,
                                    gformat(learning_rate),
                                    gformat(momentum),
                                    iterations,
                                    gformat(adversarial_linf_epsilon),
                                    attempts,
                                ), attack=attack, objective=attacks.objectives.UntargetedF0Objective(),
                                attempts=attempts, testloader=adversarialtrainsetloader, model_specific=True)
                            attack_config(
                                'input_linf_gd%s%s_lr%s_mom%s_i%d_e%s_at%d_train' % (
                                    normalized_name,
                                    backtrack_name,
                                    gformat(learning_rate),
                                    gformat(momentum),
                                    iterations,
                                    gformat(adversarial_linf_epsilon),
                                    attempts,
                                ), attack=attack, objective=attacks.objectives.UntargetedF0Objective(),
                                attempts=attempts, testloader=adversarialtesttrainloader,
                                model_specific=True)

    attack = attacks.BatchAutoAttack()
    attack.version = 'standard'
    attack.epsilon = adversarial_linf_epsilon
    attack_config('input_linf_aa_standard_e%s' % gformat(adversarial_linf_epsilon),
                  attack=attack, objective=attacks.objectives.UntargetedF0Objective(), attempts=1,
                  model_specific=True)
    attack_config('input_linf_aa_standard_e%s_train' % gformat(adversarial_linf_epsilon),
                  attack=attack, objective=attacks.objectives.UntargetedF0Objective(),
                  testloader=adversarialtesttrainloader, attempts=1, model_specific=True)

#
# l2 weight + linf input
#

for adversarial_linf_epsilon in adversarial_linf_epsilons:
    for attempts in [10]:
        for iterations in [7, 10, 20]:
            for input_learning_rate in [0.007]:
                for input_normalized, input_normalized_name in zip([True], ['_normalized']):
                    for adversarial_l2_error_rate in random_l2_error_rates:
                        input_attack = attacks.BatchGradientDescent()
                        input_attack.max_iterations = iterations
                        input_attack.base_lr = input_learning_rate
                        input_attack.momentum = 0
                        input_attack.c = 0
                        input_attack.lr_factor = 1
                        input_attack.backtrack = False
                        input_attack.normalized = input_normalized
                        input_attack.initialization = attacks.initializations.LInfUniformNormInitialization(adversarial_linf_epsilon)
                        input_attack.projection = attacks.projections.SequentialProjections([
                            attacks.projections.BoxProjection(0, 1),
                            attacks.projections.LInfProjection(adversarial_linf_epsilon),
                        ])
                        input_attack.norm = attacks.norms.LInfNorm()

                        weight_attack = attacks.weights.RandomAttack()
                        weight_attack.epochs = 1
                        weight_attack.initialization = attacks.weights.initializations.LayerWiseL2UniformSphereInitialization(adversarial_l2_error_rate)
                        weight_attack.norm = attacks.weights.norms.L2Norm()
                        weight_attack.projection = None
                        weight_attack.get_layers = no_normalization

                        attack = attacks.weights_inputs.SequentialAttack2(weight_attack, input_attack)
                        attack_weights_inputs_config(
                            'sequential2_weight_input_l2_random_nonorm2_e%s_linf_gd%s_lr%s_e%s_i%d_at%d_test' % (
                                gformat(adversarial_l2_error_rate),
                                input_normalized_name,
                                gformat(input_learning_rate),
                                gformat(adversarial_linf_epsilon),
                                iterations,
                                attempts
                            ), testloader=adversarialtestloader, attack=attack, attempts=attempts,
                        )
                        attack_weights_inputs_config(
                            'sequential2_weight_input_l2_random_nonorm2_e%s_linf_gd%s_lr%s_e%s_i%d_at%d_train_test' % (
                                gformat(adversarial_l2_error_rate),
                                input_normalized_name,
                                gformat(input_learning_rate),
                                gformat(adversarial_linf_epsilon),
                                iterations,
                                attempts
                            ), testloader=adversarialtesttrainloader, attack=attack, attempts=attempts,
                        )

                        weight_attack = attacks.weights.RandomAttack()
                        weight_attack.epochs = 1
                        weight_attack.initialization = attacks.weights.initializations.FilterWiseL2UniformSphereInitialization(adversarial_l2_error_rate)
                        weight_attack.norm = attacks.weights.norms.L2Norm()
                        weight_attack.projection = None
                        weight_attack.get_layers = no_normalization

                        attack = attacks.weights_inputs.SequentialAttack2(weight_attack, input_attack)
                        attack_weights_inputs_config(
                            'sequential2_weight_input_l2_random_nonorm2_filter_e%s_linf_gd%s_lr%s_e%s_i%d_at%d_test' % (
                                gformat(adversarial_l2_error_rate),
                                input_normalized_name,
                                gformat(input_learning_rate),
                                gformat(adversarial_linf_epsilon),
                                iterations,
                                attempts
                            ), testloader=adversarialtestloader, attack=attack, attempts=attempts,
                        )
                        attack_weights_inputs_config(
                            'sequential2_weight_input_l2_random_nonorm2_filter_e%s_linf_gd%s_lr%s_e%s_i%d_at%d_train_test' % (
                                gformat(adversarial_l2_error_rate),
                                input_normalized_name,
                                gformat(input_learning_rate),
                                gformat(adversarial_linf_epsilon),
                                iterations,
                                attempts
                            ), testloader=adversarialtesttrainloader, attack=attack, attempts=attempts,
                        )

                    for adversarial_l2_error_rate in adversarial_l2_error_rates:
                        for weight_learning_rate in [0.001, 0.01]:
                            for weight_normalization, weight_normalization_name in zip([
                                #attacks.weights.normalizations.L2Normalization(),
                                #attacks.weights.normalizations.RelativeL2Normalization(),
                                attacks.weights.normalizations.LayerWiseRelativeL2Normalization(),
                                #None
                            ], [
                                #'_l2normalized',
                                #'_rl2normalized',
                                '_lwrl2normalized',
                                #''
                            ]):
                                attack = attacks.weights_inputs.GradientDescentAttack()
                                attack.epochs = iterations
                                attack.weight_initialization = attacks.weights.initializations.LayerWiseL2UniformNormInitialization(adversarial_l2_error_rate)
                                attack.weight_projection = attacks.weights.projections.SequentialProjections([
                                    attacks.weights.projections.LayerWiseL2Projection(adversarial_l2_error_rate),
                                ])
                                attack.weight_norm = attacks.weights.norms.L2Norm()
                                attack.weight_normalization = weight_normalization
                                attack.input_initialization = attacks.initializations.LInfUniformNormInitialization(adversarial_linf_epsilon)
                                attack.input_projection = attacks.projections.SequentialProjections([
                                    attacks.projections.BoxProjection(0, 1),
                                    attacks.projections.LInfProjection(adversarial_linf_epsilon),
                                ])
                                attack.input_norm = attacks.norms.LInfNorm()
                                attack.input_normalized = True
                                attack.input_lr = input_learning_rate
                                attack.weight_lr = weight_learning_rate
                                attack_weights_inputs_config(
                                    'joint_weight_input_l2_gd%s_lr%s_e%s_linf_gd%s_lr%s_e%s_i%d_at%d_test' % (
                                        weight_normalization_name,
                                        gformat(weight_learning_rate),
                                        gformat(adversarial_l2_error_rate),
                                        input_normalized_name,
                                        gformat(input_learning_rate),
                                        gformat(adversarial_linf_epsilon),
                                        iterations,
                                        attempts
                                    ), testloader=adversarialtestloader, attack=attack, attempts=attempts,
                                )
                                attack_weights_inputs_config(
                                    'joint_weight_input_l2_gd%s_lr%s_e%s_linf_gd%s_lr%s_e%s_i%d_at%d_train_test' % (
                                        weight_normalization_name,
                                        gformat(weight_learning_rate),
                                        gformat(adversarial_l2_error_rate),
                                        input_normalized_name,
                                        gformat(input_learning_rate),
                                        gformat(adversarial_linf_epsilon),
                                        iterations,
                                        attempts
                                    ), testloader=adversarialtesttrainloader, attack=attack, attempts=attempts,
                                )

                                attack = attacks.weights_inputs.GradientDescentAttack()
                                attack.epochs = iterations
                                attack.weight_initialization = attacks.weights.initializations.LayerWiseL2UniformNormInitialization(adversarial_l2_error_rate)
                                attack.weight_projection = attacks.weights.projections.SequentialProjections([
                                    attacks.weights.projections.LayerWiseL2Projection(adversarial_l2_error_rate),
                                ])
                                attack.weight_norm = attacks.weights.norms.L2Norm()
                                attack.weight_normalization = weight_normalization
                                attack.input_initialization = attacks.initializations.LInfUniformNormInitialization(adversarial_linf_epsilon)
                                attack.input_projection = attacks.projections.SequentialProjections([
                                    attacks.projections.BoxProjection(0, 1),
                                    attacks.projections.LInfProjection(adversarial_linf_epsilon),
                                ])
                                attack.input_norm = attacks.norms.LInfNorm()
                                attack.input_normalized = True
                                attack.input_lr = input_learning_rate
                                attack.weight_lr = weight_learning_rate
                                attack.get_layers = no_normalization
                                attack_weights_inputs_config(
                                    'joint_weight_input_l2_gd_nonorm2%s_lr%s_e%s_linf_gd%s_lr%s_e%s_i%d_at%d_test_batch' % (
                                        weight_normalization_name,
                                        gformat(weight_learning_rate),
                                        gformat(adversarial_l2_error_rate),
                                        input_normalized_name,
                                        gformat(input_learning_rate),
                                        gformat(adversarial_linf_epsilon),
                                        iterations,
                                        attempts
                                    ), testloader=adversarialtrainbatchloader, attack=attack, attempts=attempts,
                                )
                                attack_weights_inputs_config(
                                    'joint_weight_input_l2_gd_nonorm2%s_lr%s_e%s_linf_gd%s_lr%s_e%s_i%d_at%d_test' % (
                                        weight_normalization_name,
                                        gformat(weight_learning_rate),
                                        gformat(adversarial_l2_error_rate),
                                        input_normalized_name,
                                        gformat(input_learning_rate),
                                        gformat(adversarial_linf_epsilon),
                                        iterations,
                                        attempts
                                    ), testloader=adversarialtestloader, attack=attack, attempts=attempts,
                                )
                                attack_weights_inputs_config(
                                    'joint_weight_input_l2_gd_nonorm2%s_lr%s_e%s_linf_gd%s_lr%s_e%s_i%d_at%d_train_test' % (
                                        weight_normalization_name,
                                        gformat(weight_learning_rate),
                                        gformat(adversarial_l2_error_rate),
                                        input_normalized_name,
                                        gformat(input_learning_rate),
                                        gformat(adversarial_linf_epsilon),
                                        iterations,
                                        attempts
                                    ), testloader=adversarialtesttrainloader, attack=attack, attempts=attempts,
                                )

                                attack = attacks.weights_inputs.GradientDescentAttack()
                                attack.epochs = iterations
                                attack.weight_initialization = attacks.weights.initializations.FilterWiseL2UniformNormInitialization(adversarial_l2_error_rate)
                                attack.weight_projection = attacks.weights.projections.SequentialProjections([
                                    attacks.weights.projections.FilterWiseL2Projection(adversarial_l2_error_rate),
                                ])
                                attack.weight_norm = attacks.weights.norms.L2Norm()
                                attack.weight_normalization = weight_normalization
                                attack.input_initialization = attacks.initializations.LInfUniformNormInitialization(adversarial_linf_epsilon)
                                attack.input_projection = attacks.projections.SequentialProjections([
                                    attacks.projections.BoxProjection(0, 1),
                                    attacks.projections.LInfProjection(adversarial_linf_epsilon),
                                ])
                                attack.input_norm = attacks.norms.LInfNorm()
                                attack.input_normalized = True
                                attack.input_lr = input_learning_rate
                                attack.weight_lr = weight_learning_rate
                                attack_weights_inputs_config(
                                    'joint_weight_input_l2_gd_filter%s_lr%s_e%s_linf_gd%s_lr%s_e%s_i%d_at%d_test' % (
                                        weight_normalization_name,
                                        gformat(weight_learning_rate),
                                        gformat(adversarial_l2_error_rate),
                                        input_normalized_name,
                                        gformat(input_learning_rate),
                                        gformat(adversarial_linf_epsilon),
                                        iterations,
                                        attempts
                                    ), testloader=adversarialtestloader, attack=attack, attempts=attempts,
                                )
                                attack_weights_inputs_config(
                                    'joint_weight_input_l2_gd_filter%s_lr%s_e%s_linf_gd%s_lr%s_e%s_i%d_at%d_train_test' % (
                                        weight_normalization_name,
                                        gformat(weight_learning_rate),
                                        gformat(adversarial_l2_error_rate),
                                        input_normalized_name,
                                        gformat(input_learning_rate),
                                        gformat(adversarial_linf_epsilon),
                                        iterations,
                                        attempts
                                    ), testloader=adversarialtesttrainloader, attack=attack, attempts=attempts,
                                )

                                attack = attacks.weights_inputs.GradientDescentAttack()
                                attack.epochs = iterations
                                attack.weight_initialization = attacks.weights.initializations.FilterWiseL2UniformNormInitialization(adversarial_l2_error_rate)
                                attack.weight_projection = attacks.weights.projections.SequentialProjections([
                                    attacks.weights.projections.FilterWiseL2Projection(adversarial_l2_error_rate),
                                ])
                                attack.weight_norm = attacks.weights.norms.L2Norm()
                                attack.weight_normalization = weight_normalization
                                attack.input_initialization = attacks.initializations.LInfUniformNormInitialization(adversarial_linf_epsilon)
                                attack.input_projection = attacks.projections.SequentialProjections([
                                    attacks.projections.BoxProjection(0, 1),
                                    attacks.projections.LInfProjection(adversarial_linf_epsilon),
                                ])
                                attack.input_norm = attacks.norms.LInfNorm()
                                attack.input_normalized = True
                                attack.input_lr = input_learning_rate
                                attack.weight_lr = weight_learning_rate
                                attack.get_layers = no_normalization
                                attack_weights_inputs_config(
                                    'joint_weight_input_l2_gd_nonorm2_filter%s_lr%s_e%s_linf_gd%s_lr%s_e%s_i%d_at%d_test' % (
                                        weight_normalization_name,
                                        gformat(weight_learning_rate),
                                        gformat(adversarial_l2_error_rate),
                                        input_normalized_name,
                                        gformat(input_learning_rate),
                                        gformat(adversarial_linf_epsilon),
                                        iterations,
                                        attempts
                                    ), testloader=adversarialtestloader, attack=attack, attempts=attempts,
                                )
                                attack_weights_inputs_config(
                                    'joint_weight_input_l2_gd_nonorm2_filter%s_lr%s_e%s_linf_gd%s_lr%s_e%s_i%d_at%d_train_test' % (
                                        weight_normalization_name,
                                        gformat(weight_learning_rate),
                                        gformat(adversarial_l2_error_rate),
                                        input_normalized_name,
                                        gformat(input_learning_rate),
                                        gformat(adversarial_linf_epsilon),
                                        iterations,
                                        attempts
                                    ), testloader=adversarialtesttrainloader, attack=attack, attempts=attempts,
                                )

                                attack = attacks.weights_inputs.GradientDescentAttack()
                                attack.epochs = iterations
                                attack.weight_initialization = attacks.weights.initializations.L2UniformNormInitialization(relative_epsilon=adversarial_l2_error_rate)
                                attack.weight_projection = attacks.weights.projections.SequentialProjections([
                                    attacks.weights.projections.L2Projection(relative_epsilon=adversarial_l2_error_rate)
                                ])
                                attack.weight_norm = attacks.weights.norms.L2Norm()
                                attack.weight_normalization = weight_normalization
                                attack.input_initialization = attacks.initializations.LInfUniformNormInitialization(adversarial_linf_epsilon)
                                attack.input_projection = attacks.projections.SequentialProjections([
                                    attacks.projections.BoxProjection(0, 1),
                                    attacks.projections.LInfProjection(adversarial_linf_epsilon),
                                ])
                                attack.input_norm = attacks.norms.LInfNorm()
                                attack.input_normalized = True
                                attack.input_lr = input_learning_rate
                                attack.weight_lr = weight_learning_rate
                                attack_weights_inputs_config(
                                    'joint_weight_input_l2_gd_relative%s_lr%s_e%s_linf_gd%s_lr%s_e%s_i%d_at%d_test' % (
                                        weight_normalization_name,
                                        gformat(weight_learning_rate),
                                        gformat(adversarial_l2_error_rate),
                                        input_normalized_name,
                                        gformat(input_learning_rate),
                                        gformat(adversarial_linf_epsilon),
                                        iterations,
                                        attempts
                                    ), testloader=adversarialtestloader, attack=attack, attempts=attempts,
                                )
                                attack_weights_inputs_config(
                                    'joint_weight_input_l2_gd_relative%s_lr%s_e%s_linf_gd%s_lr%s_e%s_i%d_at%d_train_test' % (
                                        weight_normalization_name,
                                        gformat(weight_learning_rate),
                                        gformat(adversarial_l2_error_rate),
                                        input_normalized_name,
                                        gformat(input_learning_rate),
                                        gformat(adversarial_linf_epsilon),
                                        iterations,
                                        attempts
                                    ), testloader=adversarialtesttrainloader, attack=attack, attempts=attempts,
                                )

                                attack = attacks.weights_inputs.GradientDescentAttack()
                                attack.epochs = iterations
                                attack.weight_initialization = attacks.weights.initializations.L2UniformNormInitialization(relative_epsilon=adversarial_l2_error_rate)
                                attack.weight_projection = attacks.weights.projections.SequentialProjections([
                                    attacks.weights.projections.L2Projection(relative_epsilon=adversarial_l2_error_rate)
                                ])
                                attack.weight_norm = attacks.weights.norms.L2Norm()
                                attack.weight_normalization = weight_normalization
                                attack.input_initialization = attacks.initializations.LInfUniformNormInitialization(adversarial_linf_epsilon)
                                attack.input_projection = attacks.projections.SequentialProjections([
                                    attacks.projections.BoxProjection(0, 1),
                                    attacks.projections.LInfProjection(adversarial_linf_epsilon),
                                ])
                                attack.input_norm = attacks.norms.LInfNorm()
                                attack.input_normalized = True
                                attack.input_lr = input_learning_rate
                                attack.weight_lr = weight_learning_rate
                                attack.get_layers = no_normalization
                                attack_weights_inputs_config(
                                    'joint_weight_input_l2_gd_nonorm2_relative%s_lr%s_e%s_linf_gd%s_lr%s_e%s_i%d_at%d_test' % (
                                        weight_normalization_name,
                                        gformat(weight_learning_rate),
                                        gformat(adversarial_l2_error_rate),
                                        input_normalized_name,
                                        gformat(input_learning_rate),
                                        gformat(adversarial_linf_epsilon),
                                        iterations,
                                        attempts
                                    ), testloader=adversarialtestloader, attack=attack, attempts=attempts,
                                )
                                attack_weights_inputs_config(
                                    'joint_weight_input_l2_gd_nonorm2_relative%s_lr%s_e%s_linf_gd%s_lr%s_e%s_i%d_at%d_train_test' % (
                                        weight_normalization_name,
                                        gformat(weight_learning_rate),
                                        gformat(adversarial_l2_error_rate),
                                        input_normalized_name,
                                        gformat(input_learning_rate),
                                        gformat(adversarial_linf_epsilon),
                                        iterations,
                                        attempts
                                    ), testloader=adversarialtesttrainloader, attack=attack, attempts=attempts,
                                )

log('set up attacks ...')


def simple_curriculum(attack, loss, perturbed_loss, epoch, threshold=2.1, population=1, epochs=5):
    p = 1

    return p, {
        'population': p,
        'epochs': attack.epochs,
    }


def threshold_curriculum(attack, loss, perturbed_loss, epoch, threshold=2.1, population=1, epochs=5):
    if loss < threshold:
        p = population
    else:
        p = 0

    return p, {
        'population': p,
        'epochs': attack.epochs,
    }


# import adversarial training trained with cifar10 (i.e., with AutoAugment) and cifar10_noaa_500k (i.e., no AutoAttack but unlabeled data)
for model_name in [
    'at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100',
]:
    for base_directory, base_directory_name in zip([
        'ICCV/Cifar10/', 'ICCV/Cifar10NoAA500k/'
    ], [
        'aa_', '500k_'
    ]):
        external_training_config(
            common.paths.experiment_file(base_directory + '/%s' % model_name, 'classifier', common.paths.STATE_EXT),
            base_directory_name + model_name,
            model_name,
            base_directory=base_directory,
        )
        external_training_config(
            common.paths.experiment_file(base_directory + '/%s' % model_name + '_es', 'classifier', common.paths.STATE_EXT),
            base_directory_name + model_name + '_es',
            model_name + '_es',
            base_directory=base_directory,
        )

normal_training_config(
    'normal_training',
    projection=None,
)

learning_rate = 0.007
momentum = 0
backtrack = False
normalized = True

for adversarial_linf_epsilon in adversarial_linf_epsilons:
    for iterations in [1, 3, 5, 7, 14]:
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = iterations
        attack.base_lr = learning_rate
        attack.momentum = momentum
        attack.lr_factor = 2
        attack.backtrack = backtrack
        attack.normalized = normalized
        attack.c = 0
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(adversarial_linf_epsilon)
        attack.projection = attacks.projections.SequentialProjections([
            attacks.projections.BoxProjection(0, 1),
            attacks.projections.LInfProjection(adversarial_linf_epsilon)
        ])
        attack.norm = attacks.norms.LInfNorm()

        # 50/50 adversarial training
        adversarial_training_config(
            'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s' % (
                gformat(learning_rate),
                gformat(momentum),
                iterations,
                gformat(adversarial_linf_epsilon),
            ),
            attack=attack,
            projection=None,
        )

        # 100% adversarial training
        adversarial_training_config(
            'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100' % (
                gformat(learning_rate),
                gformat(momentum),
                iterations,
                gformat(adversarial_linf_epsilon),
            ),
            attack=attack,
            projection=None,
            fraction=1,
        )

        for scale in [0.5, 2]:
            adversarial_training_config(
                'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100_s%s' % (
                    gformat(learning_rate),
                    gformat(momentum),
                    iterations,
                    gformat(adversarial_linf_epsilon),
                    gformat(scale),
                ),
                attack=attack,
                projection=None,
                fraction=1,
            )
            adversarial_training_config(
                'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100_slog%s' % (
                    gformat(learning_rate),
                    gformat(momentum),
                    iterations,
                    gformat(adversarial_linf_epsilon),
                    gformat(scale),
                ),
                attack=attack,
                projection=None,
                fraction=1,
            )

        for trainloader_bs, bs in zip([
            trainloader_4, trainloader_8, trainloader_16, trainloader_32, trainloader_64, trainloader_256, trainloader_512,
        ], [
            4, 8, 16, 32, 64, 256, 512
        ]):
            adversarial_training_config(
                'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100_b%s' % (
                    gformat(learning_rate),
                    gformat(momentum),
                    iterations,
                    gformat(adversarial_linf_epsilon),
                    bs,
                ),
                attack=attack,
                projection=None,
                fraction=1,
                trainloader=trainloader_bs,
            )

        for wa_weight, wa_weight_name in zip([0.98, 0.985, 0.99, 0.9975], ['098', '0985', '099', '']):
            adversarial_training_config(
                'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100_wa%s' % (
                    gformat(learning_rate),
                    gformat(momentum),
                    iterations,
                    gformat(adversarial_linf_epsilon),
                    wa_weight_name,
                ),
                attack=attack,
                projection=None,
                fraction=1,
                keep_average=True,
                keep_average_tau=0.9975,
            )
            adversarial_training_config(
                'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100_wa%s_extr' % (
                    gformat(learning_rate),
                    gformat(momentum),
                    iterations,
                    gformat(adversarial_linf_epsilon),
                    wa_weight_name,
                ),
                attack=attack,
                projection=None,
                fraction=1,
            )
            adversarial_training_config(
                'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100_wa%s_extr_uncal' % (
                    gformat(learning_rate),
                    gformat(momentum),
                    iterations,
                    gformat(adversarial_linf_epsilon),
                    wa_weight_name,
                ),
                attack=attack,
                projection=None,
                fraction=1,
            )

        # adversarial training with Adam
        adversarial_training_config(
            'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100_adam' % (
                gformat(learning_rate),
                gformat(momentum),
                iterations,
                gformat(adversarial_linf_epsilon),
            ),
            attack=attack,
            projection=None,
            fraction=1,
            get_optimizer=common.utils.partial(get_adam_optimizer, lr=helper.adam_lr),
        )

        # adversarial training with entropy SGD
        for entropy_epochs, entropy_L in zip([150, 75, 50, 30], [1, 2, 3, 5]):
            adversarial_training_config(
                'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100_esgd%d' % (
                    gformat(learning_rate),
                    gformat(momentum),
                    iterations,
                    gformat(adversarial_linf_epsilon),
                    entropy_L,
                ),
                attack=attack,
                projection=None,
                fraction=1,
                epochs=entropy_epochs,
                get_scheduler=common.utils.partial(get_multi_step_scheduler, milestones=[2 * entropy_epochs//5, 3 * entropy_epochs//5,  4 * entropy_epochs//5], lr_factor=0.1, batches_per_epoch=len(trainloader)),
                get_optimizer=common.utils.partial(get_entropy_optimizer, L=entropy_L, eps=1e-4, gamma=1e-4, scoping=1e-3),
                interface=common.experiments.EntropyAdversarialTrainingInterface,
            )

        # late learning rate schedule
        adversarial_training_config(
            'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100_late' % (
                gformat(learning_rate),
                gformat(momentum),
                iterations,
                gformat(adversarial_linf_epsilon),
            ),
            attack=attack,
            projection=None,
            fraction=1,
            get_scheduler=common.utils.partial(get_late_multi_step_scheduler, batches_per_epoch=len(trainloader)),
        )
        # learning rate factor 0.316
        adversarial_training_config(
            'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100_lrf0316' % (
                gformat(learning_rate),
                gformat(momentum),
                iterations,
                gformat(adversarial_linf_epsilon),
            ),
            attack=attack,
            projection=None,
            fraction=1,
            get_scheduler=common.utils.partial(get_multi_step_scheduler, batches_per_epoch=len(trainloader), lr_factor=0.316),
        )
        # constant learning rate
        for larger_learning_rate in [0.2, 0.1, 0.01]:
            adversarial_training_config(
                'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100_lr%s' % (
                    gformat(learning_rate),
                    gformat(momentum),
                    iterations,
                    gformat(adversarial_linf_epsilon),
                    gformat(larger_learning_rate),
                ),
                attack=attack,
                projection=None,
                fraction=1,
                get_optimizer=common.utils.partial(get_l2_optimizer, lr=larger_learning_rate),
            )
        for constant_learning_rate, constant_learning_rate_name in zip([0.1, 0.01, 0.005], ['01', '001', '0005']):
            adversarial_training_config(
                'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100_const%s' % (
                    gformat(learning_rate),
                    gformat(momentum),
                    iterations,
                    gformat(adversarial_linf_epsilon),
                    constant_learning_rate_name,
                ),
                attack=attack,
                projection=None,
                fraction=1,
                get_optimizer=common.utils.partial(get_l2_optimizer, lr=constant_learning_rate),
                get_scheduler=common.utils.partial(get_no_scheduler, batches_per_epoch=len(trainloader)),
            )
        # cyclic learning rate
        for cycles, cycles_name in zip([1, 2, 3, 4], ['', '2', '3', '4']):
            adversarial_training_config(
                'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100_cyc%s' % (
                    gformat(learning_rate),
                    gformat(momentum),
                    iterations,
                    gformat(adversarial_linf_epsilon),
                    cycles_name,
                ),
                attack=attack,
                projection=None,
                fraction=1,
                get_scheduler=common.utils.partial(get_cyclic_scheduler, batches_per_epoch=len(trainloader)),
                epochs=helper.cyclic_epochs,
            )

        # no weight decay
        adversarial_training_config(
            'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100_nowd' % (
                gformat(learning_rate),
                gformat(momentum),
                iterations,
                gformat(adversarial_linf_epsilon),
            ),
            attack=attack,
            projection=None,
            fraction=1,
            get_optimizer=common.utils.partial(get_l2_optimizer, weight_decay=0)
        )
        # different weight decay values
        for weight_decay_value, weight_decay_name in zip([0.05, 0.01, 0.005, 0.001, 0.0001], ['005', '001', '0005', '0001', '00001']):
            adversarial_training_config(
                'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100_wd%s' % (
                    gformat(learning_rate),
                    gformat(momentum),
                    iterations,
                    gformat(adversarial_linf_epsilon),
                    weight_decay_name,
                ),
                attack=attack,
                projection=None,
                fraction=1,
                get_optimizer=common.utils.partial(get_l2_optimizer, weight_decay=0.001)
            )

        # adversarial training with label smoothing
        for label_smoothing in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            adversarial_training_config(
                'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100_ls%s' % (
                    gformat(learning_rate),
                    gformat(momentum),
                    iterations,
                    gformat(adversarial_linf_epsilon),
                    gformat(label_smoothing),
                ),
                attack=attack,
                projection=None,
                fraction=1,
                loss=common.utils.partial(common.torch.smooth_classification_loss, epsilon=label_smoothing, K=10),
            )

        # adversarial training with label noise
        for label_noise in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            adversarial_training_config(
                'at_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100_ln%s' % (
                    gformat(learning_rate),
                    gformat(momentum),
                    iterations,
                    gformat(adversarial_linf_epsilon),
                    gformat(label_noise),
                ),
                attack=attack,
                projection=None,
                fraction=1,
                loss=common.utils.partial(common.torch.noisy_classification_loss, noise_rate=label_noise, K=10),
            )

        # adversarial training without label leaking for PGD
        adversarial_training_config(
            'at_pll_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100' % (
                gformat(learning_rate),
                gformat(momentum),
                iterations,
                gformat(adversarial_linf_epsilon),
            ),
            attack=attack,
            projection=None,
            fraction=1,
            prevent_label_leaking=True,
        )

        # adversarial training ignoring mis-classified examples
        adversarial_training_config(
            'at_ii_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100' % (
                gformat(learning_rate),
                gformat(momentum),
                iterations,
                gformat(adversarial_linf_epsilon),
            ),
            attack=attack,
            projection=None,
            fraction=1,
            ignore_incorrect=True,
        )
        # MART and TRADES
        for beta in [1, 3, 6, 9]:
            adversarial_training_config(
                'mart%s_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100' % (
                    gformat(beta),
                    gformat(learning_rate),
                    gformat(momentum),
                    iterations,
                    gformat(adversarial_linf_epsilon),
                ),
                attack=attack,
                projection=None,
                fraction=1. / beta,
                interface=common.experiments.MARTAdversarialTrainingInterface,
            )
            adversarial_training_config(
                'trades%s_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100' % (
                    gformat(beta),
                    gformat(learning_rate),
                    gformat(momentum),
                    iterations,
                    gformat(adversarial_linf_epsilon),
                ),
                attack=attack,
                projection=None,
                fraction=1. / beta,
                objective=attacks.objectives.UntargetedKLObjective(),
                interface=common.experiments.TRADESAdversarialTrainingInterface,
            )

        # adversarial training with unsupervised rotation prediction
        if rot_trainloader is not None:
            for unsup_weight in [0.5, 1, 2, 4, 8]:
                adversarial_semi_supervised_training_config(
                    'at_ssl%s_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100' % (
                        gformat(unsup_weight),
                        gformat(learning_rate),
                        gformat(momentum),
                        iterations,
                        gformat(adversarial_linf_epsilon),
                    ),
                    attack=attack,
                    trainloader=rot_trainloader,
                    projection=None,
                    unsup_weight=unsup_weight,
                )

    # PGD-7-tau
    for iterations in [7]:
        for tau in [0, 1, 2, 3, 4]:
            attack = attacks.BatchTauGradientDescent()
            attack.max_iterations = iterations
            # this line makes it PGD-7-tau
            attack.tau = tau
            attack.base_lr = learning_rate
            attack.momentum = momentum
            attack.lr_factor = 2
            attack.backtrack = backtrack
            attack.normalized = normalized
            attack.c = 0
            attack.initialization = attacks.initializations.LInfUniformNormInitialization(adversarial_linf_epsilon)
            attack.projection = attacks.projections.SequentialProjections([
                attacks.projections.BoxProjection(0, 1),
                attacks.projections.LInfProjection(adversarial_linf_epsilon)
            ])
            attack.norm = attacks.norms.LInfNorm()

            adversarial_training_config(
                'at_tau%d_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100' % (
                    tau,
                    gformat(learning_rate),
                    gformat(momentum),
                    iterations,
                    gformat(adversarial_linf_epsilon),
                ),
                attack=attack,
                projection=None,
                fraction=1,
            )

    # variants of adversarial training with adversarial weight perturbations
    for adversarial_l2_error_rate in adversarial_l2_error_rates:
        for weight_population in [1, 3]:
            for weight_iterations in [1, 3, 5]:
                for weight_learning_rate in [0.01, 1]:
                    for weight_normalization, weight_normalization_name in zip([
                        attacks.weights.normalizations.LayerWiseRelativeL2Normalization(),
                        attacks.weights.normalizations.RelativeL2Normalization(),
                        attacks.weights.normalizations.L2Normalization(),
                        None
                    ], [
                        '_lwrl2normalized',
                        '_rl2normalized',
                        '_l2normalized',
                        ''
                    ]):
                        for weight_curriculum_function, weight_curriculum_name in zip([
                            threshold_curriculum,
                            simple_curriculum,
                        ], [
                            'threshold',
                            'simple'
                        ]):
                            input_iterations = 7
                            input_momentum = 0
                            input_learning_rate = 0.007
                            weight_momentum = 0

                            weight_attack = attacks.weights.GradientDescentAttack()
                            weight_attack.epochs = weight_iterations
                            weight_attack.base_lr = weight_learning_rate
                            weight_attack.normalization = weight_normalization
                            weight_attack.backtrack = False
                            weight_attack.momentum = weight_momentum
                            weight_attack.lr_factor = 2
                            weight_attack.initialization = attacks.weights.initializations.LayerWiseL2UniformNormInitialization(adversarial_l2_error_rate)
                            weight_attack.norm = attacks.weights.norms.L2Norm()
                            weight_attack.projection = attacks.weights.SequentialProjections([
                                attacks.weights.projections.LayerWiseL2Projection(adversarial_l2_error_rate)
                            ])
                            weight_attack.get_layers = no_normalization_no_bias

                            input_attack = attacks.BatchGradientDescent()
                            input_attack.max_iterations = input_iterations
                            input_attack.base_lr = input_learning_rate
                            input_attack.momentum = input_momentum
                            input_attack.lr_factor = 2
                            input_attack.backtrack = False
                            input_attack.normalized = True
                            input_attack.c = 0
                            input_attack.initialization = attacks.initializations.LInfUniformNormInitialization(adversarial_linf_epsilon)
                            input_attack.projection = attacks.projections.SequentialProjections([
                                attacks.projections.BoxProjection(0, 1),
                                attacks.projections.LInfProjection(adversarial_linf_epsilon)
                            ])
                            input_attack.norm = attacks.norms.LInfNorm()

                            adversarial_weights_inputs_training_config(
                                '%s_awit_nonb_l2_gd%s_lr%s_mom%s_i%d_e%s_linf_gd_normalized_lr%s_mom%s_i%d_e%s_pop%d' % (
                                    weight_curriculum_name,
                                    weight_normalization_name,
                                    gformat(weight_learning_rate),
                                    gformat(weight_momentum),
                                    weight_iterations,
                                    gformat(adversarial_l2_error_rate),
                                    gformat(input_learning_rate),
                                    gformat(input_momentum),
                                    input_iterations,
                                    gformat(adversarial_linf_epsilon),
                                    weight_population,
                                ),
                                curriculum=common.utils.partial(weight_curriculum_function, population=weight_population, epochs=weight_iterations),
                                input_attack=input_attack,
                                weight_attack=weight_attack,
                                projection=None)
                            adversarial_weights_inputs_training_config(
                                '%s_awit_nonb2_l2_gd_normalized%s_lr%s_mom%s_i%d_e%s_linf_gd_normalized_lr%s_mom%s_i%d_e%s_pop%d' % (
                                    weight_curriculum_name,
                                    weight_normalization_name,
                                    gformat(weight_learning_rate),
                                    gformat(weight_momentum),
                                    weight_iterations,
                                    gformat(adversarial_l2_error_rate),
                                    gformat(input_learning_rate),
                                    gformat(input_momentum),
                                    input_iterations,
                                    gformat(adversarial_linf_epsilon),
                                    weight_population,
                                ),
                                curriculum=common.utils.partial(weight_curriculum_function, population=weight_population, epochs=weight_iterations),
                                input_attack=input_attack,
                                weight_attack=weight_attack,
                                projection=None)

    # adversarial training with adversarial weight perturbations
    for adversarial_l2_error_rate in adversarial_l2_error_rates:
        for weight_population in [1, 3]:
            for weight_normalization, weight_normalization_name in zip([
                attacks.weights.normalizations.LayerWiseRelativeL2Normalization(),
            ], [
                '_lwrl2normalized',
            ]):
                for weight_curriculum_function, weight_curriculum_name in zip([
                    threshold_curriculum,
                    simple_curriculum,
                ], [
                    'threshold',
                    'simple'
                ]):
                    input_iterations = 7
                    input_momentum = 0
                    input_learning_rate = 0.007
                    weight_momentum = 0

                    weight_attack = attacks.weights.GradientDescentAttack()
                    weight_attack.epochs = 1
                    weight_attack.base_lr = adversarial_l2_error_rate
                    weight_attack.normalization = weight_normalization
                    weight_attack.backtrack = False
                    weight_attack.momentum = 0
                    weight_attack.lr_factor = 2
                    weight_attack.initialization = None
                    weight_attack.norm = attacks.weights.norms.L2Norm()
                    weight_attack.projection = None
                    weight_attack.get_layers = no_normalization_no_bias

                    input_attack = attacks.BatchGradientDescent()
                    input_attack.max_iterations = input_iterations
                    input_attack.base_lr = input_learning_rate
                    input_attack.momentum = input_momentum
                    input_attack.lr_factor = 2
                    input_attack.backtrack = False
                    input_attack.normalized = True
                    input_attack.c = 0
                    input_attack.initialization = attacks.initializations.LInfUniformNormInitialization(adversarial_linf_epsilon)
                    input_attack.projection = attacks.projections.SequentialProjections([
                            attacks.projections.BoxProjection(0, 1),
                            attacks.projections.LInfProjection(
                                adversarial_linf_epsilon)
                        ])
                    input_attack.norm = attacks.norms.LInfNorm()

                    adversarial_weights_inputs_training_config(
                        '%s_awit_nonb_l2_gd_fgsm%s_e%s_linf_gd_normalized_lr%s_mom%s_i%d_e%s_pop%d' % (
                            weight_curriculum_name,
                            weight_normalization_name,
                            gformat(adversarial_l2_error_rate),
                            gformat(input_learning_rate),
                            gformat(input_momentum),
                            input_iterations,
                            gformat(adversarial_linf_epsilon),
                            weight_population,
                        ),
                        curriculum=common.utils.partial(weight_curriculum_function, population=weight_population, epochs=1),
                        input_attack=input_attack,
                        weight_attack=weight_attack,
                        projection=None)

    # adversarial training with _random_ weight perturbations
    for adversarial_l2_error_rate in random_l2_error_rates:
        for weight_population in [1, 3]:
            for weight_curriculum_function, weight_curriculum_name in zip([
                threshold_curriculum,
                simple_curriculum
            ], [
                'threshold',
                'simple'
            ]):
                input_iterations = 7
                input_momentum = 0
                input_learning_rate = 0.007
                weight_momentum = 0

                weight_attack = attacks.weights.RandomAttack()
                weight_attack.epochs = 1
                weight_attack.initialization = attacks.weights.initializations.LayerWiseL2UniformSphereInitialization(adversarial_l2_error_rate)
                weight_attack.norm = attacks.weights.norms.L2Norm()
                weight_attack.projection = None
                weight_attack.get_layers = no_normalization_no_bias

                input_attack = attacks.BatchGradientDescent()
                input_attack.max_iterations = input_iterations
                input_attack.base_lr = input_learning_rate
                input_attack.momentum = input_momentum
                input_attack.lr_factor = 2
                input_attack.backtrack = False
                input_attack.normalized = True
                input_attack.c = 0
                input_attack.initialization = attacks.initializations.LInfUniformNormInitialization(adversarial_linf_epsilon)
                input_attack.projection = attacks.projections.SequentialProjections([
                    attacks.projections.BoxProjection(0, 1),
                    attacks.projections.LInfProjection(adversarial_linf_epsilon)
                ])
                input_attack.norm = attacks.norms.LInfNorm()

                adversarial_weights_inputs_training_config(
                    '%s_awit_nonb_l2_random_e%s_linf_gd_normalized_lr%s_mom%s_i%d_e%s_pop%d' % (
                        weight_curriculum_name,
                        gformat(adversarial_l2_error_rate),
                        gformat(input_learning_rate),
                        gformat(input_momentum),
                        input_iterations,
                        gformat(adversarial_linf_epsilon),
                        weight_population,
                    ),
                    curriculum=common.utils.partial(weight_curriculum_function, population=weight_population, epochs=1),
                    input_attack=input_attack,
                    weight_attack=weight_attack,
                    projection=None)

# adversarial training with weight clipping
for projection, projection_name, clipped in zip([
        attacks.weights.projections.BoxProjection(-0.005, 0.005),
        attacks.weights.projections.BoxProjection(-0.01, 0.01),
        attacks.weights.projections.BoxProjection(-0.015, 0.015),
        attacks.weights.projections.BoxProjection(-0.02, 0.02),
    ], [
        '0005p_',
        '001p_',
        '0015p_',
        '002p_',
    ], [
        0.005,
        0.01,
        0.015,
        0.02,
    ]):

    for adversarial_linf_epsilon in adversarial_linf_epsilons:
        for iterations in [1, 3, 5, 7, 14]:
            attack = attacks.BatchGradientDescent()
            attack.max_iterations = iterations
            attack.base_lr = learning_rate
            attack.momentum = momentum
            attack.lr_factor = 2
            attack.backtrack = backtrack
            attack.normalized = normalized
            attack.c = 0
            attack.initialization = attacks.initializations.LInfUniformNormInitialization(adversarial_linf_epsilon)
            attack.projection = attacks.projections.SequentialProjections([
                attacks.projections.BoxProjection(0, 1),
                attacks.projections.LInfProjection(adversarial_linf_epsilon)
            ])
            attack.norm = attacks.norms.LInfNorm()

            # 100% adversarial training
            adversarial_training_config(
                '%sat_linf_gd_normalized_lr%s_mom%s_i%d_e%s_f100' % (
                    projection_name,
                    gformat(learning_rate),
                    gformat(momentum),
                    iterations,
                    gformat(adversarial_linf_epsilon),
                ),
                attack=attack,
                projection=projection,
                fraction=1,
            )

log('set up models ...')

cifar10_input_linf_benchmark = [#
    globals()['input_linf_aa_standard_e00314'],
    globals()['input_linf_gd_normalized_lr0007_mom0_i20_e00314_at10'],
    #
    globals()['input_linf_aa_standard_e00314_train'],
    globals()['input_linf_gd_normalized_lr0007_mom0_i20_e00314_at10_train'],
]
cifar10_weight_l2_benchmark = [
    globals()['weight_l2_random_nonorm2_e01_at50_test'],
    globals()['weight_l2_random_nonorm2_e025_at50_test'],
    globals()['weight_l2_random_nonorm2_e05_at50_test'],
    globals()['weight_l2_random_nonorm2_e075_at50_test'],
    globals()['weight_l2_random_nonorm2_e1_at50_test'],
    #
    globals()['weight_l2_gd_nonorm2_lwrl2normalized_i20_lr001_mom0_e000075_at10_test_test'],
    globals()['weight_l2_gd_nonorm2_lwrl2normalized_i20_lr001_mom0_e0001_at10_test_test'],
    globals()['weight_l2_gd_nonorm2_lwrl2normalized_i20_lr001_mom0_e0002_at10_test_test'],
    globals()['weight_l2_gd_nonorm2_lwrl2normalized_i20_lr001_mom0_e0003_at10_test_test'],
    globals()['weight_l2_gd_nonorm2_lwrl2normalized_i20_lr001_mom0_e0004_at10_test_test'],
    globals()['weight_l2_gd_nonorm2_lwrl2normalized_i20_lr001_mom0_e0005_at10_test_test'],
]
cifar10_weight_input_l2_linf_benchmark = [
    globals()['sequential2_weight_input_l2_random_nonorm2_e01_linf_gd_normalized_lr0007_e00314_i20_at10_test'],
    globals()['sequential2_weight_input_l2_random_nonorm2_e02_linf_gd_normalized_lr0007_e00314_i20_at10_test'],
    globals()['sequential2_weight_input_l2_random_nonorm2_e025_linf_gd_normalized_lr0007_e00314_i20_at10_test'],
    globals()['sequential2_weight_input_l2_random_nonorm2_e03_linf_gd_normalized_lr0007_e00314_i20_at10_test'],
    globals()['sequential2_weight_input_l2_random_nonorm2_e04_linf_gd_normalized_lr0007_e00314_i20_at10_test'],
    globals()['sequential2_weight_input_l2_random_nonorm2_e05_linf_gd_normalized_lr0007_e00314_i20_at10_test'],
    globals()['sequential2_weight_input_l2_random_nonorm2_e075_linf_gd_normalized_lr0007_e00314_i20_at10_test'],
    globals()['sequential2_weight_input_l2_random_nonorm2_e1_linf_gd_normalized_lr0007_e00314_i20_at10_test'],
    #
    globals()['joint_weight_input_l2_gd_nonorm2_lwrl2normalized_lr001_e000075_linf_gd_normalized_lr0007_e00314_i20_at10_test'],
    globals()['joint_weight_input_l2_gd_nonorm2_lwrl2normalized_lr001_e0001_linf_gd_normalized_lr0007_e00314_i20_at10_test'],
    globals()['joint_weight_input_l2_gd_nonorm2_lwrl2normalized_lr001_e0002_linf_gd_normalized_lr0007_e00314_i20_at10_test'],
    globals()['joint_weight_input_l2_gd_nonorm2_lwrl2normalized_lr001_e0003_linf_gd_normalized_lr0007_e00314_i20_at10_test'],
    globals()['joint_weight_input_l2_gd_nonorm2_lwrl2normalized_lr001_e0004_linf_gd_normalized_lr0007_e00314_i20_at10_test'],
    globals()['joint_weight_input_l2_gd_nonorm2_lwrl2normalized_lr001_e0005_linf_gd_normalized_lr0007_e00314_i20_at10_test'],
]
cifar10_early_stopping_benchmark = [
    globals()['input_linf_gd_normalized_lr0007_mom0_i7_e00314_at5_es'],
]
cifar10_epochs_benchmark = [
    globals()['input_linf_gd_normalized_lr0007_mom0_i20_e00314_at10'],
    globals()['input_linf_gd_normalized_lr0007_mom0_i20_e00314_at10_train'],
    globals()['sequential2_weight_input_l2_random_nonorm2_e05_linf_gd_normalized_lr0007_e00314_i20_at10_test'],
    globals()['joint_weight_input_l2_gd_nonorm2_lwrl2normalized_lr001_e000075_linf_gd_normalized_lr0007_e00314_i20_at10_test'],
]
cifar10_benchmark = cifar10_input_linf_benchmark + cifar10_weight_l2_benchmark + cifar10_weight_input_l2_linf_benchmark
cifar10_models = [
    globals()['at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100'],
    globals()['at_linf_gd_normalized_lr0007_mom0_i14_e00314_f100'],
    globals()['at_linf_gd_normalized_lr0007_mom0_i7_e00352_f100'],
    globals()['at_ii_linf_gd_normalized_lr0007_mom0_i7_e00314_f100'],
    globals()['at_pll_linf_gd_normalized_lr0007_mom0_i7_e00314_f100'],
    globals()['0005p_at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100'],
    globals()['at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ls01'],
    globals()['at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ls02'],
    globals()['at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ls03'],
    globals()['at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ls04'],
    globals()['at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ls05'],
    globals()['at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ln01'],
    globals()['at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ln02'],
    globals()['at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ln03'],
    globals()['at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ln04'],
    globals()['at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ln05'],
    globals()['at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_cyc'],
    globals()['at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_wd0001'],
    globals()['at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_wd001'],
    globals()['at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_wd005'],
    globals()['at_ssl05_linf_gd_normalized_lr0007_mom0_i7_e00314_f100'],
    globals()['at_ssl1_linf_gd_normalized_lr0007_mom0_i7_e00314_f100'],
    globals()['at_ssl2_linf_gd_normalized_lr0007_mom0_i7_e00314_f100'],
    globals()['at_ssl4_linf_gd_normalized_lr0007_mom0_i7_e00314_f100'],
    globals()['at_ssl8_linf_gd_normalized_lr0007_mom0_i7_e00314_f100'],
    globals()['trades1_linf_gd_normalized_lr0007_mom0_i7_e00314_f100'],
    globals()['trades3_linf_gd_normalized_lr0007_mom0_i7_e00314_f100'],
    globals()['trades6_linf_gd_normalized_lr0007_mom0_i7_e00314_f100'],
    globals()['trades9_linf_gd_normalized_lr0007_mom0_i7_e00314_f100'],
]