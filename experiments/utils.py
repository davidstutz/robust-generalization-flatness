from common.log import log, LogLevel
import math
import models
import numpy
import torch


def training_arguments(parser):
    """
    Default training arguments.

    :param parser: argument parser
    :type parser: argparse.ArgumentParser
    """
    parser.add_argument('-n', '--normalization', type=str, dest='normalization', default='')
    parser.add_argument('-a', '--activation', type=str, dest='activation', default='relu')
    parser.add_argument('--whiten', action='store_true', default=False)
    parser.add_argument('--dropout', action='store_true', default=False)
    parser.add_argument('--init_scale', default=1, type=float)
    parser.add_argument('--scale', default=1, type=float)
    parser.add_argument('--channels', default=32, type=int)
    parser.add_argument('--clipping', default=None, type=float)


def training_argument_list(args):
    """
    Get default training parameters.

    :param args: arguments
    :type args: [str]
    :return: arguments
    :rtype: [str]
    """
    training_args = [
        '-n=%s' % str(args.normalization),
        '-a=%s' % str(args.activation),
        '--init_scale=%s' % str(args.init_scale),
        '--channels=%s' % str(args.channels),
        '--scale=%s' % str(args.scale),
    ]
    if args.clipping is not None:
        training_args += ['--clipping=%s' % str(args.clipping)]
    if args.whiten:
        training_args += ['--whiten']
    if args.dropout:
        training_args += ['--dropout']
    return training_args


def get_training_directory(training_config, args, suffix=''):
    """
    Get training directory based on training arguments.

    :param training_config: training configuration
    :type training_config: common.experiments.config.NormalTrainingConfig
    :param args: arguments
    :param suffix: suffix to use for directory
    :type suffix: str
    :return: directory name
    :rtype: str
    """
    init_scale = args.init_scale
    scale = args.scale
    clipping = args.clipping
    channels = args.channels
    whiten = args.whiten
    dropout = args.dropout
    architecture = args.architecture
    normalization = args.normalization
    if normalization == '':
        log('[Warning] no normalization', LogLevel.WARNING)
    activation = args.activation
    if activation == '':
        log('[Warning] no activation', LogLevel.WARNING)

    # just allows to call various scripts sequentially without caring about resetting the original directory
    if getattr(training_config, 'original_directory', None) is None:
        training_config.original_directory = training_config.directory
    directory = training_config.original_directory
    if suffix != '':
        directory += '_' + suffix
    directory += '_' + architecture
    if normalization != '':
        directory += '_' + normalization
    if activation != 'relu':
        directory += '_' + activation
    if whiten:
        directory += '_whiten'
    if dropout:
        directory += '_dropout'
    if scale != 1:
        directory += ('_scale%g' % scale).replace('.', '')
    if clipping is not None:
        directory += ('_clipping%g' % clipping).replace('.', '')
    if not math.isclose(init_scale, 1.):
        directory += ('_%g' % init_scale).replace('.', '')
    directory += '_%d' % channels
    return directory


def get_get_model(args, config):
    """
    Get a function to return and initialize the model.

    :param args: arguments
    :param config: training configuration
    :return: callable to get model
    """
    channels = args.channels
    whiten = args.whiten
    dropout = args.dropout
    init_scale = args.init_scale
    scale = args.scale
    clipping = args.clipping
    architecture = args.architecture
    normalization = args.normalization
    activation = args.activation

    def set_whiten(model, resolution):
        mean = numpy.zeros(resolution[0])
        std = numpy.zeros(resolution[0])
        for c in range(resolution[0]):
            mean[c] = numpy.mean(config.trainset.images[:, :, :, c])
            std[c] = numpy.std(config.trainset.images[:, :, :, c])
        model.whiten.weight.data = torch.from_numpy(std.astype(numpy.float32))
        model.whiten.bias.data = torch.from_numpy(mean.astype(numpy.float32))

    if architecture == 'resnet18':
        def get_model(N_class, resolution):
            model = models.ResNet(N_class, resolution, blocks=[2, 2, 2, 2], channels=channels,
                                  normalization=normalization, activation=activation, whiten=whiten, scale=scale, init_scale=init_scale,
                                  clipping=clipping, dropout=dropout)
            if whiten:
                set_whiten(model, resolution)
            print(model)
            return model
    else:
        assert False

    return get_model