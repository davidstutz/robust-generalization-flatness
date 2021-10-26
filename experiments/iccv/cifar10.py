import common.datasets
import experiments.iccv.helper as helper

helper.lr = 0.05
helper.epochs = 150
helper.batch_size = 128
helper.base_directory = 'ICCV/Cifar10/'

helper.cyclic_min_lr = 0
helper.cyclic_max_lr = 0.2
helper.cyclic_epochs = 30
helper.adam_lr = 0.1

helper.autoaugment = True
helper.augment = False
helper.mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
helper.cutout = 16

helper.train_N = 50000
helper.test_N = 10000
helper.trainset = common.datasets.Cifar10TrainSet
helper.testset = common.datasets.Cifar10TestSet
helper.unsupset = False

from experiments.iccv.common import *