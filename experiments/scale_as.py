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
import common.scaling
from common.log import log, LogLevel
import utils
import importlib


class ScaleAs:
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
        parser.add_argument('model_as', type=str)
        parser.add_argument('suffix', type=str)
        utils.training_arguments(parser)
        parser.add_argument('--per_layer', action='store_true', default=False)
        parser.add_argument('-d', '--norm', default='linf', type=str)
        parser.add_argument('--ignore', default='', type=str)

        return parser

    def main(self):
        """
        Main.
        """

        def get_model_file(directory):
            epoch = None
            model_file = common.paths.experiment_file(directory, 'classifier', common.paths.STATE_EXT)
            if not os.path.exists(model_file):
                model_file, epoch = common.experiments.find_incomplete_file(model_file)

            return model_file, epoch

        reference_training_config = getattr(self.config, self.args.model_as, None)
        assert reference_training_config is not None
        reference_training_config.directory = utils.get_training_directory(reference_training_config, self.args)

        training_configs = getattr(self.config, self.args.model)
        if not isinstance(training_configs, list):
            training_configs = [training_configs]

        reference_model_file, _ = get_model_file(reference_training_config.directory)
        if reference_model_file is None:
            log('not found %s' % reference_training_config.directory, LogLevel.ERROR)
            exit()

        reference_model = common.state.State.load(reference_model_file).model
        #reference_model.train()
        reference_model.eval()

        reference_model = reference_model.cuda()

        reference_probabilities = common.test.test(reference_model, self.config.testloader, cuda=True, eval=True)
        reference_evaluation = common.eval.CleanEvaluation(reference_probabilities, self.config.testset.labels)
        log('test error (reference): %g' % reference_evaluation.test_error())

        reference_statistics = common.scaling.statistics(reference_model)
        reference_norms = common.scaling.norms(reference_model)
        log('reference - min, mean, max: %g, %g, %g' % (reference_statistics[-1][0], reference_statistics[-1][1], reference_statistics[-1][2]))
        log('reference - l2, l1, linf: %g, %g, %g' % (reference_norms[0], reference_norms[1], reference_norms[2]))

        for training_config in training_configs:
            training_config_directory = utils.get_training_directory(training_config, self.args)
            log(training_config_directory)
            model_file, _ = get_model_file(training_config_directory)

            if model_file is None:
                log('not found %s' % training_config.directory, LogLevel.WARNING)
                continue

            state = common.state.State.load(model_file)
            model = state.model
            #model.train()
            model.eval()
            model = model.cuda()

            original_probabilities = common.test.test(model, self.config.testloader, cuda=True)
            original_evaluation = common.eval.CleanEvaluation(original_probabilities, self.config.testset.labels)
            log('test error (before): %g' % original_evaluation.test_error())

            statistics = common.scaling.statistics(model)
            norms = common.scaling.norms(model)
            log('UNscaled - min, mean, max: %g, %g, %g' % (statistics[-1][0], statistics[-1][1], statistics[-1][2]))
            log('UNscaled - l2, l1, linf: %g, %g, %g' % (norms[0], norms[1], norms[2]))

            ignore = self.args.ignore.split(',')
            log('ignoring layers: %s' % ','.join(ignore))
            if self.args.per_layer:
                raise NotImplementedError
            else:
                factor = common.scaling.scaling_factor(reference_model, model, norm=self.args.norm, names=['weight', 'bias'], ignore=ignore)
                log('factor %g' % factor)
                common.scaling.scale(model, factor, ignore=ignore)

            statistics = common.scaling.statistics(model)
            norms = common.scaling.norms(model)
            log('scaled - min, mean, max: %g, %g, %g' % (statistics[-1][0], statistics[-1][1], statistics[-1][2]))
            log('scaled - l2, l1, linf: %g, %g, %g' % (norms[0], norms[1], norms[2]))

            assert self.args.suffix != ''
            scaling_directory = utils.get_training_directory(training_config, self.args, suffix=self.args.suffix)
            scaled_model_file = common.paths.experiment_file(scaling_directory, 'classifier', ext=common.paths.STATE_EXT)
            common.state.State.checkpoint(scaled_model_file, model, optimizer=state.optimizer, scheduler=state.scheduler, epoch=state.epoch)
            log('wrote %s' % scaled_model_file)

            scaled_probabilities = common.test.test(model, self.config.testloader, cuda=True, eval=True)
            scaled_probabilities_file = common.paths.experiment_file(scaling_directory, 'probabilities', ext=common.paths.HDF5_EXT)
            common.utils.write_hdf5(scaled_probabilities_file, scaled_probabilities, 'probabilities')

            evaluation = common.eval.CleanEvaluation(scaled_probabilities, self.config.testset.labels)
            log('test error: %g (from %g, reference: %g)' % (evaluation.test_error(), original_evaluation.test_error(), reference_evaluation.test_error()))


if __name__ == '__main__':
    program = ScaleAs()
    program.main()