import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse
import common.experiments
import common.utils
import common.eval
import common.paths
import common.imgaug
import common.datasets
from common.log import log, LogLevel
import utils
import importlib


class Test:
    def __init__(self, args=None):
        """
        Initialize.

        :param args: optional arguments if not to use sys.argv
        :type args: [str]
        """

        self.args = None
        """ Arguments of program. """

        parser = self.get_parser()
        if args is not None:
            self.args = parser.parse_args(args)
        else:
            self.args = parser.parse_args()

        self.config = importlib.import_module(self.args.config)

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser()
        parser.add_argument('config', type=str)
        parser.add_argument('architecture', type=str)
        parser.add_argument('model', type=str)
        parser.add_argument('attack', nargs='?', type=str) # to be compatible with attack interface
        utils.training_arguments(parser)
        parser.add_argument('--train', action='store_true', default=False)
        parser.add_argument('--epochs', action='store_true', default=False)
        parser.add_argument('--epoch', default=-1, type=int)

        return parser

    def main(self):
        """
        Main.
        """

        training_configs = getattr(self.config, self.args.model)
        if not isinstance(training_configs, list):
            training_configs = [training_configs]

        for training_config in training_configs:
            training_config.directory = utils.get_training_directory(training_config, self.args)

        def get_model_file(training_config):
            epoch = None
            model_file = common.paths.experiment_file(training_config.directory, 'classifier', common.paths.STATE_EXT)
            #log('checking %s' % model_file)
            if not os.path.exists(model_file):
                model_file, epoch = common.experiments.find_incomplete_file(model_file)

            return model_file, epoch

        def get_model_files(training_config):
            model_file = common.paths.experiment_file(training_config.directory, 'classifier', common.paths.STATE_EXT)
            model_files, model_epochs = common.experiments.find_incomplete_files(model_file)
            if os.path.exists(model_file):
                if model_files is None:
                    model_files = []
                    model_epochs = []
                model_files.insert(0, model_file)
                model_epochs.insert(0, None)
            return model_files, model_epochs

        for training_config in training_configs:
            # epoch might change in between attacks so check model for each attack anew
            if self.args.epochs:
                model_files, model_epochs = get_model_files(training_config)
            elif self.args.epoch >= 0:
                model_file, epoch = get_model_file(training_config)
                if epoch is not None:
                    model_file = model_file.replace('.%s' % str(epoch), '')
                print(model_file, self.args.epoch)
                model_files = [model_file + '.%d' % self.args.epoch]
                model_epochs = [self.args.epoch]
            else:
                model_file, epoch = get_model_file(training_config)
                model_files = [model_file]
                model_epochs = [epoch]

            for m in range(len(model_files)):
                model_file = model_files[m]
                epoch = model_epochs[m]
                log('%s %s' % (training_config.directory, str(epoch)))

                if model_file is None:
                    log('not found %s' % training_config.directory, LogLevel.WARNING)
                    continue

                if self.args.train:
                    testloader = self.config.testtrainloader
                    log('using testtrainloader')
                else:
                    testloader = self.config.testloader
                    log('using testloader')

                if self.args.train:
                    probabilities_file = common.paths.experiment_file(training_config.directory, 'train_probabilities', common.paths.HDF5_EXT)
                    if epoch is not None:
                        probabilities_file = common.paths.experiment_file(training_config.directory, 'train_probabilities%s.%d' % (common.paths.HDF5_EXT, epoch), '')
                else:
                    probabilities_file = common.paths.experiment_file(training_config.directory, 'probabilities', common.paths.HDF5_EXT)
                    if epoch is not None:
                        probabilities_file = common.paths.experiment_file(training_config.directory, 'probabilities%s.%d' % (common.paths.HDF5_EXT, epoch), '')

                if os.path.exists(probabilities_file):
                    probabilities = common.utils.read_hdf5(probabilities_file, 'probabilities')
                    if probabilities.shape[0] == testloader.dataset.labels.shape[0]:
                        log('found/skipping %s' % probabilities_file)
                        eval = common.eval.CleanEvaluation(probabilities, testloader.dataset.labels, validation=0)
                        log('epoch %s test error in %%: %g' % (str(epoch), eval.test_error() * 100))
                        continue;

                model = common.state.State.load(model_file).model
                model.eval()

                cuda = True
                if cuda:
                    model = model.cuda()

                probabilities = common.test.test(model, testloader, cuda=cuda)
                common.utils.write_hdf5(probabilities_file, probabilities, 'probabilities')
                log('wrote %s' % probabilities_file)

                eval = common.eval.CleanEvaluation(probabilities, testloader.dataset.labels, validation=0)
                log('epoch %s test error in %%: %g' % (str(epoch), eval.test_error() * 100))


if __name__ == '__main__':
    program = Test()
    program.main()