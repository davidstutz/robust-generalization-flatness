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


class Attack:
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
        utils.training_arguments(parser)
        parser.add_argument('--force', action='store_true', default=False)
        parser.add_argument('--epochs', action='store_true', default=False)
        parser.add_argument('--epoch', default=-1, type=int)
        parser.add_argument('--skip', type=int, default=1)

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

        attack_configs = getattr(self.config, self.args.attack)
        if not isinstance(attack_configs, list):
            attack_configs = [attack_configs]

        def get_model_file(training_config):
            epoch = None
            model_file = common.paths.experiment_file(training_config.directory, 'classifier', common.paths.STATE_EXT)
            #log('checking %s' % model_file)
            if not os.path.exists(model_file):
                model_file, epoch = common.experiments.find_incomplete_file(model_file)

            return model_file, epoch

        def get_model_files(training_config):
            assert training_config.directory != ''
            model_file = common.paths.experiment_file(training_config.directory, 'classifier', common.paths.STATE_EXT)
            model_files, model_epochs = common.experiments.find_incomplete_files(model_file)

            #assert model_files is not None, '%s does not exist' % os.path.dirname(model_file)
            if model_files is None:
                model_files = []
                model_epochs = []

            if os.path.exists(model_file):
                model_files.insert(0, model_file)
                model_epochs.insert(0, None)
            return model_files, model_epochs

        for training_config in training_configs:
            # epoch might change in between attacks so check model for each attack anew
            model_file, epoch = get_model_file(training_config)

            if model_file is None:
                log('not found %s' % training_config.directory, LogLevel.WARNING)
                continue

            if self.args.epochs:
                model_files, model_epochs = get_model_files(training_config)
            elif self.args.epoch >= 0:
                model_files = [model_file + '.%d' % self.args.epoch]
                model_epochs = [self.args.epoch]
            else:
                model_files = [model_file]
                model_epochs = [epoch]

            for m in range(0, len(model_epochs), self.args.skip):
                for attack_config in attack_configs:
                    log(training_config.directory)
                    log(attack_config.directory)
                    attack_config.snapshot = model_epochs[m]
                    log('epoch %s' % str(attack_config.snapshot))

                    # for clipped_x attacks, the clipping is determined by the clipping used for training the model
                    attack_projection = None
                    if isinstance(attack_config, common.experiments.AttackWeightsConfig):
                        attack_projection = getattr(attack_config.attack, 'projection', False)

                    # projection is set in interface
                    assert getattr(attack_config, 'interface', None) is not None
                    program = attack_config.interface(training_config, attack_config)
                    program.main(self.args.force)

                    # reset projection
                    if isinstance(attack_config, common.experiments.AttackWeightsConfig):
                        attack_config.attack.projection = attack_projection


if __name__ == '__main__':
    program = Attack()
    program.main()