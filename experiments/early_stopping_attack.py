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
import numpy
import utils
import importlib
import shutil


class EarlyStoppingAttack:
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
        parser.add_argument('attack', type=str)
        parser.add_argument('--suffix', type=str, default='')
        parser.add_argument('--wa_suffix', type=str, default='')
        parser.add_argument('--force', action='store_true', default=False, dest='force')
        utils.training_arguments(parser)

        return parser

    def main(self):
        """
        Main.
        """

        training_config = getattr(self.config, self.args.model)
        assert not isinstance(training_config, list)
        training_config.directory = utils.get_training_directory(training_config, self.args)

        attack_configs = getattr(self.config, self.args.attack)
        if not isinstance(attack_configs, list):
            attack_configs = [attack_configs]

        model_file = common.paths.experiment_file(training_config.directory, 'classifier', common.paths.STATE_EXT)
        model_files, model_epochs = common.experiments.find_incomplete_files(model_file)
        assert model_files is not None and model_epochs is not None, 'no model files found for %s' % training_config.directory
        model_epochs.append(None)
        model_files.append(model_file)
        print(model_epochs)

        if not os.path.exists(model_file):
            log('training not finished', LogLevel.ERROR)
            exit()

        if self.args.suffix != '':
            target_config = getattr(self.config, self.args.model + self.args.suffix)
            target_config.directory = utils.get_training_directory(target_config, self.args)
            common.utils.makedir(target_config.directory)
            log('copying to %s' % target_config.directory, LogLevel.WARNING)
        else:
            target_config = training_config
            log('overwriting files in place', LogLevel.ERROR)
            exit()

        cuda = True
        metric = 'test_error'
        robust_metric = 'robust_test_error'
        pickle_file = common.paths.experiment_file(target_config.directory, 'early_stopping', common.paths.PICKLE_EXT)

        if os.path.exists(pickle_file):
            data = common.utils.read_pickle(pickle_file)
            if 'evaluations' in data.keys():
                evaluations = data['evaluations']
                log('early stopping found in %s' % pickle_file, LogLevel.WARNING)

                log('epoch: rte, te')
                for e in range(len(evaluations)):
                    model_epoch = model_epochs[e]
                    evaluation = evaluations[e]
                    log('%s: %g %g' % (str(model_epoch), getattr(evaluation, robust_metric)(), getattr(evaluation, metric)()))

                if not 'epochs' in data.keys():
                    data['epochs'] = model_epochs
                    common.utils.write_pickle(pickle_file, data)
                    log('wrote %s' % pickle_file)

                if self.args.suffix != '':
                    snapshot_model_file = common.paths.experiment_file(training_config.directory, 'classifier', common.paths.STATE_EXT)
                    target_model_file = common.paths.experiment_file(target_config.directory, 'classifier', common.paths.STATE_EXT)
                    if data['epoch'] is not None:
                        snapshot_model_file += '.%d' % data['epoch']
                    shutil.copy(snapshot_model_file, target_model_file)
                    log('copy %s -> %s' % (snapshot_model_file, target_model_file))

                    snapshot_probabilities_file = common.paths.experiment_file(training_config.directory, 'probabilities', common.paths.HDF5_EXT)
                    target_probabilities_file = common.paths.experiment_file(target_config.directory, 'probabilities', common.paths.HDF5_EXT)
                    if data['epoch'] is not None:
                        snapshot_probabilities_file += '.%d' % data['epoch']
                    shutil.copy(snapshot_probabilities_file, target_probabilities_file)
                    log('copy %s -> %s' % (snapshot_probabilities_file, target_probabilities_file))

                if self.args.wa_suffix != '':
                    wa_config = getattr(self.config, self.args.model + self.args.wa_suffix)
                    wa_config.directory = utils.get_training_directory(wa_config, self.args)
                    common.utils.makedir(wa_config.directory)
                    log('copying to %s' % wa_config.directory, LogLevel.WARNING)

                    wa_model_file = model_file + 'average'
                    if data['epoch'] is not None:
                        wa_model_file += '.%d' % data['epoch']

                    model = common.state.State.load(wa_model_file).model
                    if cuda:
                        model = model.cuda()

                    model.eval()
                    probabilities = common.test.test(model, self.config.testloader, cuda=cuda)
                    evaluation = common.eval.CleanEvaluation(probabilities, self.config.testloader.dataset.labels)
                    log('uncalibrated test error %g' % (evaluation.test_error()))

                    target_probabilities_file = common.paths.experiment_file(wa_config.directory, 'probabilities', common.paths.HDF5_EXT)
                    common.utils.write_hdf5(target_probabilities_file, probabilities, 'probabilities')
                    log('wrote %s' % target_probabilities_file)

                    target_model_file = common.paths.experiment_file(wa_config.directory, 'classifier', common.paths.STATE_EXT)
                    shutil.copy(wa_model_file, target_model_file)
                    log('copy %s -> %s' % (wa_model_file, target_model_file))

                if not self.args.force:
                    return

        data = {
            'training_config': training_config.directory,
            'attack_configs': [attack_config.directory for attack_config in attack_configs],
        }
        common.utils.write_pickle(pickle_file, data)
        log('wrote %s' % pickle_file)

        evaluations = []
        min_robust_test_error = 1e12
        min_epoch = None

        for m in range(len(model_epochs)):
            model_epoch = model_epochs[m]
            model_file = model_files[m]

            model = common.state.State.load(model_file).model
            if cuda:
                model = model.cuda()
            model.eval()
            clean_probabilities = common.test.test(model, self.config.adversarialtrainsetloader, cuda=cuda)

            adversarial_probabilities = None
            adversarial_errors = None

            for attack_config in attack_configs:

                attack_config.snapshot = model_epoch
                log('epoch %s, attack %s' % (str(model_epoch), attack_config.directory))

                program = attack_config.interface(training_config, attack_config)
                program.main()

                if model_epoch is not None:
                    adversarial_probabilities_file = common.paths.experiment_file('%s/%s' % (training_config.directory, attack_config.directory), 'perturbations%d' % model_epoch, common.paths.HDF5_EXT)
                else:
                    adversarial_probabilities_file = common.paths.experiment_file('%s/%s' % (training_config.directory, attack_config.directory), 'perturbations', common.paths.HDF5_EXT)
                assert os.path.exists(adversarial_probabilities_file)

                adversarial_probabilities_ = common.utils.read_hdf5(adversarial_probabilities_file, 'probabilities')
                adversarial_errors_ = numpy.copy(adversarial_probabilities_)

                adversarial_errors_[
                    :,
                    numpy.arange(adversarial_errors_.shape[1]),
                    self.config.adversarialtrainset.labels[:adversarial_errors_.shape[1]],
                ] = 0
                assert len(adversarial_errors_.shape) == 3
                adversarial_errors_ = -numpy.max(adversarial_errors_, axis=2)

                adversarial_probabilities = common.numpy.concatenate(adversarial_probabilities, adversarial_probabilities_, axis=0)
                adversarial_errors = common.numpy.concatenate(adversarial_errors, adversarial_errors_, axis=0)

            evaluation = common.eval.AdversarialEvaluation(clean_probabilities, adversarial_probabilities, self.config.adversarialtrainset.labels, errors=adversarial_errors, validation=0)
            robust_test_error = getattr(evaluation, robust_metric)()
            log('epoch %s, robust test error %g' % (str(model_epoch), robust_test_error))
            evaluations.append(evaluation)

            if robust_test_error < min_robust_test_error:
                min_robust_test_error = robust_test_error
                min_epoch = model_epoch

        log('epoch: rte')
        for e in range(len(evaluations)):
            model_epoch = model_epochs[e]
            evaluation = evaluations[e]
            log('%s: %g %g' % (str(model_epoch), getattr(evaluation, robust_metric)(), getattr(evaluation, metric)()))
        log('best epoch: %s' % str(min_epoch))

        if self.args.suffix == '':
            # do it in place
            log ('overwriting files in place', LogLevel.WARNING)

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
                shutil.copy(snapshot_model_file, model_file)
                log('copy %s -> %s' % (snapshot_model_file, model_file))

            if min_epoch is not None:
                snapshot_probabilities_file = common.paths.experiment_file(training_config.directory, 'probabilities', common.paths.HDF5_EXT) + '.%d' % min_epoch
                shutil.copy(snapshot_probabilities_file, probabilities_file)
                log('copy %s -> %s' % (snapshot_probabilities_file, probabilities_file))
        else:
            snapshot_model_file = common.paths.experiment_file(training_config.directory, 'classifier', common.paths.STATE_EXT)
            target_model_file = common.paths.experiment_file(target_config.directory, 'classifier', common.paths.STATE_EXT)
            if min_epoch is not None:
                snapshot_model_file += '.%d' % min_epoch
            shutil.copy(snapshot_model_file, target_model_file)
            log('copy %s -> %s' % (snapshot_model_file, target_model_file))

            snapshot_probabilities_file = common.paths.experiment_file(training_config.directory, 'probabilities', common.paths.HDF5_EXT)
            target_probabilities_file = common.paths.experiment_file(target_config.directory, 'probabilities', common.paths.HDF5_EXT)
            if min_epoch is not None:
                snapshot_probabilities_file += '.%d' % min_epoch
            shutil.copy(snapshot_probabilities_file, target_probabilities_file)
            log('copy %s -> %s' % (snapshot_probabilities_file, target_probabilities_file))

        if self.args.wa_suffix != '':
            wa_config = getattr(self.config, self.args.model + self.args.wa_suffix)
            wa_config.directory = utils.get_training_directory(wa_config, self.args)
            common.utils.makedir(wa_config.directory)
            log('copying to %s' % wa_config.directory, LogLevel.WARNING)

            wa_model_file = model_file + 'average'
            if min_epoch is not None:
                wa_model_file += '.%d' % min_epoch

            model = common.state.State.load(wa_model_file).model
            if cuda:
                model = model.cuda()

            model.eval()
            probabilities = common.test.test(model, self.config.testloader, cuda=cuda)
            evaluation = common.eval.CleanEvaluation(probabilities, self.config.testloader.dataset.labels)
            log('wa test error %g' % (evaluation.test_error()))

            target_probabilities_file = common.paths.experiment_file(wa_config.directory, 'probabilities', common.paths.HDF5_EXT)
            common.utils.write_hdf5(target_probabilities_file, probabilities, 'probabilities')
            log('wrote %s' % target_probabilities_file)

            target_model_file = common.paths.experiment_file(wa_config.directory, 'classifier', common.paths.STATE_EXT)
            shutil.copy(wa_model_file, target_model_file)
            log('copy %s -> %s' % (wa_model_file, target_model_file))

        data = {
            'evaluations': evaluations,
            'epochs': model_epochs,
            'epoch': min_epoch,
            'training_config': training_config.directory,
            'attack_configs': [attack_config.directory for attack_config in attack_configs],
        }
        common.utils.write_pickle(pickle_file, data)
        log('wrote %s' % pickle_file)


if __name__ == '__main__':
    program = EarlyStoppingAttack()
    program.main()