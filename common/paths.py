import os

# This file holds a bunch of specific paths used for experiments and
# data. The intention is to have all important paths at a central location, while
# allowing to easily prototype new experiments.

# Base directory for data and experiments.
BASE_DATA = os.getenv('BASE_DATA', False)
if BASE_DATA is False:
    BASE_DATA = 'data/'
    #log('[Warning] BASE_DATA environment variable not defined, using default', LogLevel.WARNING)

BASE_EXPERIMENTS = os.getenv('BASE_EXPERIMENTS', False)
if BASE_EXPERIMENTS is False:
    BASE_EXPERIMENTS = 'experiments/'
    #log('[Warning] BASE_EXPERIMENTS environment variable not defined, using default', LogLevel.WARNING)

BASE_LOGS = os.getenv('BASE_LOGS', False)
if BASE_LOGS is False:
    BASE_LOGS = 'logs/'
    #log('[Warning] BASE_DATE environment variable not defined, using default', LogLevel.WARNING)

# Common extension types used.
TXT_EXT = '.txt'
HDF5_EXT = '.h5'
STATE_EXT = '.pth.tar'
LOG_EXT = '.log'
PNG_EXT = '.png'
PICKLE_EXT = '.pkl'
TEX_EXT = '.tex'
MAT_EXT = '.mat'
GZIP_EXT = '.gz'


# Naming conventions.
def data_file(name, ext=HDF5_EXT):
    """
    Generate path to data file.

    :param name: name of file
    :type name: str
    :param ext: extension (including period)
    :type ext: str
    :return: filepath
    :rtype: str
    """

    return os.path.join(BASE_DATA, name) + ext


def faultmap_file(filename):
    """
    Faultmap file.

    :param filename: filename
    :type filename: str
    :return: filepath
    :rtype: str
    """

    return data_file('faultmaps/%s' % filename, TXT_EXT)


def raw_mnistc_dir():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist_c/', '')


def raw_mnist_train_images_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist/train-images-idx3-ubyte', GZIP_EXT)


def raw_mnist_test_images_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist/t10k-images-idx3-ubyte', GZIP_EXT)


def raw_mnist_train_labels_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist/train-labels-idx1-ubyte', GZIP_EXT)


def raw_mnist_test_labels_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist/t10k-labels-idx-ubyte', GZIP_EXT)


def mnist_train_images_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist/train_images', HDF5_EXT)


def mnist_test_images_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist/test_images', HDF5_EXT)


def mnist_train_labels_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist/train_labels', HDF5_EXT)


def mnist_test_labels_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('mnist/test_labels', HDF5_EXT)


def raw_cifar10_dir():
    """
    Raw Cifar10 training directory.

    :return: fielpath
    :rtype: str
    """

    return data_file('Cifar10/', '')


def cifar10_train_images_file():
    """
    Cifar10 train images.

    :return: filepath
    :rtype: str
    """

    return data_file('Cifar10/train_images', HDF5_EXT)


def cifar10_test_images_file():
    """
    Cifar10 test images.

    :return: filepath
    :rtype: str
    """

    return data_file('Cifar10/test_images', HDF5_EXT)


def cifar10_train_labels_file():
    """
    Cifar10 train labels.

    :return: filepath
    :rtype: str
    """

    return data_file('Cifar10/train_labels', HDF5_EXT)


def cifar10_test_labels_file():
    """
    Cifar10 test labels.

    :return: filepath
    :rtype: str
    """

    return data_file('Cifar10/test_labels', HDF5_EXT)


def tinyimages500k_raw_file():
    """
    500k file.

    :return: filepath
    :rtype: str
    """

    return data_file('500k_pseudolabeled.pickle', '')


def tinyimages500k_train_images_file():
    """
    500k file.

    :return: filepath
    :rtype: str
    """

    return data_file('tinyimages500k/train_images', HDF5_EXT)


def tinyimages500k_train_labels_file():
    """
    500k file.

    :return: filepath
    :rtype: str
    """

    return data_file('tinyimages500k/train_labels', HDF5_EXT)


def random_images_file(N, size):
    """
    Random train images.

    :return: filepath
    :rtype: str
    """

    return data_file('random/train_%d_%s' % (N, '_'.join(list(map(str, size)))), HDF5_EXT)


def random_labels_file(N, size):
    """
    Random train labels.

    :return: filepath
    :rtype: str
    """

    return data_file('random/train_%d_%s' % (N, '_'.join(list(map(str, size)))), HDF5_EXT)


def experiment_dir(directory):
    """
    Generate path to experiment directory.

    :param directory: directory
    :type directory: str
    """

    return os.path.join(BASE_EXPERIMENTS, directory)


def experiment_file(directory, name, ext=''):
    """
    Generate path to experiment file.

    :param directory: directory
    :type directory: str
    :param name: name of file
    :type name: str
    :return: filepath
    :rtype: str
    """

    return os.path.join(BASE_EXPERIMENTS, directory, name) + ext


def log_dir(directory):
    """
    Generate path to log directory.

    :param directory: directory
    :type directory: str
    """

    return os.path.join(BASE_LOGS, directory)


def log_file(directory, name, ext=''):
    """
    Generate path to log file.

    :param directory: directory
    :type directory: str
    :param name: name of file
    :type name: str
    :return: filepath
    :rtype: str
    """

    return os.path.join(BASE_LOGS, directory, name) + ext