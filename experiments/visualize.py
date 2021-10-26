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
import common.visualization


class Visualize:
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
        parser.add_argument('attacks', type=str)
        parser.add_argument('-d', '--direction_normalization', dest='direction_normalization', type=str, default='')
        utils.training_arguments(parser)
        parser.add_argument('--epochs', action='store_true', default=False)
        parser.add_argument('--force', action='store_true', default=False)
        parser.add_argument('--force_attack', action='store_true', default=False)
        parser.add_argument('-l', '--loss_attack', type=str,  dest='loss_attack', default=None)
        parser.add_argument('-s', '--steps', type=int, dest='steps', default=51)
        parser.add_argument('--hessian', type=int, default=0)

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

        attack_configs = self.args.attacks
        attack_configs = [attack_config for attack_config in attack_configs.split(',') if attack_config != '']
        for a in range(len(attack_configs)):
            attack_configs[a] = getattr(self.config, attack_configs[a])
        assert len(attack_configs) <= 2
        assert len(attack_configs) > 0

        loss_attack = None
        if self.args.loss_attack is not None:
            loss_attack = getattr(self.config, self.args.loss_attack)
            assert not isinstance(loss_attack, list)

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

        normalizations = [normalization for normalization in self.args.direction_normalization.split(',') if normalization != '']
        assert len(normalizations) > 0

        for training_config in training_configs:
            # epoch might change in between attacks so check model for each attack anew
            if self.args.epochs:
                model_files, model_epochs = get_model_files(training_config)
                if model_files is None:
                    log('no epochs found', LogLevel.WARNING)
                    model_file, epoch = get_model_file(training_config)
                    model_files = [model_file]
                    model_epochs = [epoch]
            else:
                model_file, epoch = get_model_file(training_config)
                model_files = [model_file]
                model_epochs = [epoch]

            for m in range(0, len(model_files), 2):
                model_file = model_files[m]
                model_epoch = model_epochs[m]
                log('%s %s' % (training_config.directory, str(model_epoch)))

                if model_file is None or not os.path.exists(model_file):
                    log('not found %s' % training_config.directory, LogLevel.WARNING)
                    continue

                if len(attack_configs) > 1:
                    attack_config_a = attack_configs[0]
                    attack_config_b = attack_configs[1]

                    if isinstance(attack_config_a, common.experiments.AttackConfig):
                        assert isinstance(attack_config_b, common.experiments.AttackConfig)
                    if isinstance(attack_config_a, common.experiments.AttackWeightsConfig):
                        assert isinstance(attack_config_b, common.experiments.AttackWeightsConfig)

                    attack_config_a.snapshot = model_epoch
                    attack_config_b.snapshot = model_epoch

                    if isinstance(attack_config_a, common.experiments.AttackConfig):
                        for normalization in normalizations:
                            program = common.experiments.Visualize2DInterface(training_config, attack_config_a, attack_config_b,
                                                                              self.config.adversarialtrainbatchloader, normalization,
                                                                              steps=self.args.steps)
                            program.main(self.args.force, self.args.force_attack)
                    elif isinstance(attack_config_a, common.experiments.AttackWeightsConfig):
                        for normalization in normalizations:
                            program = common.experiments.VisualizeWeights2DInterface(training_config, attack_config_a, attack_config_b,
                                                                                     self.config.adversarialtrainbatchloader, normalization,
                                                                                     input_attack_config=loss_attack, steps=self.args.steps,
                                                                                     hessian=self.args.hessian)
                            program.main(self.args.force, self.args.force_attack)
                    else:
                        assert False

                else:
                    attack_config = attack_configs[0]
                    attack_config.snapshot = model_epoch
                    log(attack_config.directory)

                    if isinstance(attack_config, common.experiments.AttackWeightsInputsConfig):
                        # input normalization can be done automatically
                        for normalization in normalizations:
                            program = common.experiments.VisualizeWeightsInputs2DInterface(training_config, attack_config,
                                                                                           attack_config.testloader,
                                                                                           normalization, None, steps=self.args.steps)
                            program.main(self.args.force, self.args.force_attack)
                    elif isinstance(attack_config, common.experiments.AttackConfig):
                        for normalization in normalizations:
                            program = common.experiments.Visualize1DInterface(training_config, attack_config,
                                                                              attack_config.trainloader, normalization,
                                                                              steps=self.args.steps)
                            program.main(self.args.force, self.args.force_attack)
                    elif isinstance(attack_config, common.experiments.AttackWeightsConfig):
                        for normalization in normalizations:
                            log('batches: %d' % len(attack_config.trainloader))
                            if loss_attack is not None:
                                program = common.experiments.VisualizeAdversarialWeights1DInterface(training_config, attack_config,
                                                                                                    loss_attack,
                                                                                                    attack_config.trainloader,
                                                                                                    normalization,
                                                                                                    steps=self.args.steps,
                                                                                                    hessian=self.args.hessian)
                                program.main(self.args.force, self.args.force_attack)
                            else:
                                program = common.experiments.VisualizeWeights1DInterface(training_config, attack_config,
                                                                                         attack_config.trainloader, normalization,
                                                                                         input_attack_config=loss_attack, steps=self.args.steps,
                                                                                         hessian=self.args.hessian)
                                program.main(self.args.force, self.args.force_attack)
                    else:
                        assert False


if __name__ == '__main__':
    program = Visualize()
    program.main()