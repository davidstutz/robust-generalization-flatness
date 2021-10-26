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
import shutil


class ExtractWA:
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
        parser.add_argument('--suffix', type=str, default='_extr')
        parser.add_argument('--calibrated', default=False, action='store_true')
        utils.training_arguments(parser)

        return parser

    def main(self):
        """
        Main.
        """

        training_config = getattr(self.config, self.args.model)
        assert not isinstance(training_config, list)
        training_config.directory = utils.get_training_directory(training_config, self.args)

        model_file = common.paths.experiment_file(training_config.directory, 'classifier', common.paths.STATE_EXT)
        model_files, model_epochs = common.experiments.find_incomplete_files(model_file)
        assert model_files is not None and model_epochs is not None
        model_epochs.append(None)
        model_files.append(model_file)

        if not os.path.exists(model_file):
            log('training not finished', LogLevel.ERROR)
            exit()

        target_config = getattr(self.config, self.args.model + self.args.suffix)
        target_config.directory = utils.get_training_directory(target_config, self.args)
        common.utils.makedir(target_config.directory)
        log('copying to %s' % target_config.directory, LogLevel.WARNING)

        cuda = True
        for m in range(len(model_epochs)):
            model_epoch = model_epochs[m]
            average_model_file = model_file + 'average'
            if model_epoch is not None:
                average_model_file += '.%d' % model_epoch

            if not os.path.exists(average_model_file):
                log('not found %s' % average_model_file, LogLevel.WARNING)
                continue;

            target_model_file = common.paths.experiment_file(target_config.directory, 'classifier', common.paths.STATE_EXT)
            if model_epoch is not None:
                target_model_file += '.%d' % model_epoch
            else:
                target_model_file += '.uncalibrated'
            target_probabilities_file = common.paths.experiment_file(target_config.directory, 'probabilities', common.paths.HDF5_EXT)
            if model_epoch is not None:
                target_probabilities_file += '.%d' % model_epoch
            else:
                target_probabilities_file += '.uncalibrated'

            if os.path.exists(target_model_file) and os.path.exists(target_probabilities_file):
                log('%s skipping, already done' % (str(model_epoch)))
                continue

            model = common.state.State.load(average_model_file).model
            if cuda:
                model = model.cuda()

            model.eval()
            probabilities = common.test.test(model, self.config.testloader, cuda=cuda)
            evaluation = common.eval.CleanEvaluation(probabilities, self.config.testloader.dataset.labels)
            log('epoch %s, test error %g' % (str(model_epoch), evaluation.test_error()))

            common.utils.write_hdf5(target_probabilities_file, probabilities, 'probabilities')
            log('wrote %s' % target_probabilities_file)
            shutil.copy(average_model_file, target_model_file)
            log('copy %s -> %s' % (average_model_file, target_model_file))

        if self.args.calibrated:
            calibrated_model_file = model_file + 'average.calibrated'
            model = common.state.State.load(calibrated_model_file).model
            if cuda:
                model = model.cuda()

            model.eval()
            probabilities = common.test.test(model, self.config.testloader, cuda=cuda)
            evaluation = common.eval.CleanEvaluation(probabilities, self.config.testloader.dataset.labels)
            log('calibrated test error %g' % (evaluation.test_error()))

            target_probabilities_file = common.paths.experiment_file(target_config.directory, 'probabilities', common.paths.HDF5_EXT)
            common.utils.write_hdf5(target_probabilities_file, probabilities, 'probabilities')
            log('wrote %s' % target_probabilities_file)

            target_model_file = common.paths.experiment_file(target_config.directory, 'classifier', common.paths.STATE_EXT)
            shutil.copy(calibrated_model_file, target_model_file)
            log('copy %s -> %s' % (calibrated_model_file, target_model_file))
        else:
            calibrated_model_file = model_file + 'average'
            model = common.state.State.load(calibrated_model_file).model
            if cuda:
                model = model.cuda()

            model.eval()
            probabilities = common.test.test(model, self.config.testloader, cuda=cuda)
            evaluation = common.eval.CleanEvaluation(probabilities, self.config.testloader.dataset.labels)
            log('uncalibrated test error %g' % (evaluation.test_error()))

            target_probabilities_file = common.paths.experiment_file(target_config.directory, 'probabilities', common.paths.HDF5_EXT)
            common.utils.write_hdf5(target_probabilities_file, probabilities, 'probabilities')
            log('wrote %s' % target_probabilities_file)

            target_model_file = common.paths.experiment_file(target_config.directory, 'classifier', common.paths.STATE_EXT)
            shutil.copy(calibrated_model_file, target_model_file)
            log('copy %s -> %s' % (calibrated_model_file, target_model_file))


if __name__ == '__main__':
    program = ExtractWA()
    program.main()