import os
import sys
import torch
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


class Hessian:
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
        parser.add_argument('-k', type=int, default=2)
        parser.add_argument('--epochs', action='store_true', default=False)
        utils.training_arguments(parser)

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
            else:
                model_file, epoch = get_model_file(training_config)
                model_files = [model_file]
                model_epochs = [epoch]

            for m in range(len(model_files)):
                model_file = model_files[m]
                model_epoch = model_epochs[m]
                log('%s %s' % (training_config.directory, str(model_epoch)))

                if model_file is None or not os.path.exists(model_file):
                    log('not found %s' % training_config.directory, LogLevel.WARNING)
                    continue

                eigenvalues_file = os.path.join(os.path.dirname(model_file), 'eigenvalues_%d%s' % (self.args.k, common.paths.PICKLE_EXT))
                if model_epoch is not None:
                    eigenvalues_file = os.path.join(os.path.dirname(model_file), 'eigenvalues_%d%s.%d' % (self.args.k, common.paths.PICKLE_EXT, model_epoch))

                if os.path.exists(eigenvalues_file):
                    log('found %s' % eigenvalues_file)
                    continue;

                model = common.state.State.load(model_file).model
                model = model.cuda()

                if model.auxiliary is not None:
                    model.auxiliary = None

                criterion = torch.nn.CrossEntropyLoss()
                eigs, _ = common.hessian.min_max_k_hessian_eigs(model, self.config.adversarialtrainsetloader, criterion, k=self.args.k, use_cuda=True, verbose=True)

                common.utils.write_pickle(eigenvalues_file, eigs)
                log('wrote %s' % eigenvalues_file)


if __name__ == '__main__':
    program = Hessian()
    program.main()