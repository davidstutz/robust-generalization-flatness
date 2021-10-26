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


class EarlyStopping:
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

        metric = 'test_error'
        pickle_file = common.paths.experiment_file(training_config.directory, 'early_stopping', common.paths.PICKLE_EXT)
        if os.path.exists(pickle_file):
            data = common.utils.read_pickle(pickle_file)
            if 'evaluations' in data.keys():
                evaluations = data['evaluations']
                log('early stopping found in %s' % pickle_file, LogLevel.WARNING)

                log('epoch: te')
                for e in range(len(evaluations)):
                    model_epoch = model_epochs[e]
                    evaluation = evaluations[e]
                    log('%s: %g' % (str(model_epoch), getattr(evaluation, metric)()))

                if not 'epochs' in data.keys():
                    data['epochs'] = model_epochs
                    common.utils.write_pickle(pickle_file, data)
                    log('wrote %s' % pickle_file)
                return

        data = {
            'training_config': training_config.directory,
        }
        common.utils.write_pickle(pickle_file, data)
        log('wrote %s' % pickle_file)

        evaluations = []
        min_test_error = 1e12
        min_epoch = None

        cuda = True
        for m in range(len(model_epochs)):
            model_epoch = model_epochs[m]
            model_file = model_files[m]

            model = common.state.State.load(model_file).model
            if cuda:
                model = model.cuda()
            model.eval()
            clean_probabilities = common.test.test(model, self.config.adversarialtrainsetloader, cuda=cuda)

            evaluation = common.eval.CleanEvaluation(clean_probabilities, self.config.adversarialtrainset.labels)
            test_error = getattr(evaluation, metric)()
            log('epoch %s, test error %g' % (str(model_epoch), test_error))
            evaluations.append(evaluation)

            if test_error < min_test_error:
                min_test_error = test_error
                min_epoch = model_epoch

        log('epoch: te')
        for e in range(len(evaluations)):
            model_epoch = model_epochs[e]
            evaluation = evaluations[e]
            log('%s: %g' % (str(model_epoch), getattr(evaluation, metric)()))
        log('best epoch: %s' % str(min_epoch))

        backup_model_file = model_file + '.bak'
        if not os.path.exists(model_file) and os.path.exists(backup_model_file):
            shutil.copy(backup_model_file, model_file)
            log('copy %s -> %s' % (model_file, backup_model_file))
        if not os.path.exists(backup_model_file):
            shutil.copy(model_file, backup_model_file)
            log('copy %s -> %s' % (model_file, backup_model_file))

        probabilities_file = common.paths.experiment_file(training_config.directory, 'probabilities', common.paths.HDF5_EXT)
        backup_probabilities_file = probabilities_file + '.bak'
        shutil.move(probabilities_file, backup_probabilities_file)
        log('move %s -> %s' % (probabilities_file, backup_probabilities_file))

        if min_epoch is not None:
            snapshot_model_file = common.paths.experiment_file(training_config.directory, 'classifier', common.paths.STATE_EXT) + '.%d' % min_epoch
        else:
            snapshot_model_file = common.paths.experiment_file(training_config.directory, 'classifier', common.paths.STATE_EXT)
        shutil.copy(snapshot_model_file, model_file)
        log('copy %s -> %s' % (snapshot_model_file, model_file))

        if min_epoch is not None:
            snapshot_probabilities_file = common.paths.experiment_file(training_config.directory, 'probabilities', common.paths.HDF5_EXT) + '.%d' % min_epoch
        else:
            snapshot_probabilities_file = common.paths.experiment_file(training_config.directory, 'probabilities', common.paths.HDF5_EXT)
        shutil.copy(snapshot_probabilities_file, probabilities_file)
        log('copy %s -> %s' % (snapshot_probabilities_file, probabilities_file))

        data = {
            'evaluations': evaluations,
            'epochs': model_epochs,
            'epoch': min_epoch,
            'training_config': training_config.directory,
        }
        common.utils.write_pickle(pickle_file, data)
        log('wrote %s' % pickle_file)


if __name__ == '__main__':
    program = EarlyStopping()
    program.main()