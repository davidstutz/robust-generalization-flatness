import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common.log import log
from common import paths
from common import utils
import numpy


def check_dataset():
    filenames = [
        paths.mnist_train_images_file(),
        paths.mnist_test_images_file(),
        paths.mnist_train_labels_file(),
        paths.mnist_test_labels_file()
    ]

    for filename in filenames:
        data = utils.read_hdf5(filename)
        log('read %s' % filename)
        print(data.shape, numpy.max(data), numpy.min(data))


if __name__ == '__main__':
    check_dataset()