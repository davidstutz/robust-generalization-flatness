import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common.log import log
from common import paths
from common import utils
import numpy
import gzip


def check_dataset():
    filenames = [
        (paths.cifar10_train_images_file(), paths.cifar10_train_labels_file()),
        (paths.cifar10_test_images_file(), paths.cifar10_test_labels_file()),
    ]

    for files in filenames:
        data_file = files[0]
        label_file = files[1]
        data = utils.read_hdf5(data_file)
        log('read %s' % data_file)
        labels = utils.read_hdf5(label_file)
        log('read %s' % label_file)
        print(data.shape, numpy.max(data), numpy.min(data))
        print(labels[:50])


if __name__ == '__main__':
    check_dataset()