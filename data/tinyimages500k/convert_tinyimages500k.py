import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common.log import log
from common import paths
from common import utils
import numpy
import pickle


def convert_dataset():
    filepath = paths.tinyimages500k_raw_file()
    f = open(filepath, 'rb')
    data = pickle.load(f)
    log('read %s') % filepath

    images = data['data']
    images = images.astype(numpy.float32)
    images /= 255.
    print(images.shape, numpy.min(images), numpy.max(images))

    labels = data['extrapolated_targets']
    labels = labels.reshape(-1, 1).astype(numpy.int)
    print(labels.shape, numpy.min(labels), numpy.max(labels))

    utils.write_hdf5(paths.tinyimages500k_train_images_file(), images)
    utils.write_hdf5(paths.tinyimages500k_train_labels_file(), labels)


if __name__ == '__main__':
    convert_dataset()