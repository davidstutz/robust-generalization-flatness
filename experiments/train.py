import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse
import utils
import importlib


class Train:
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
        parser.add_argument('--summary', action='store_true', default=False)
        parser.add_argument('--dontask', action='store_true', default=False)

        return parser

    def main(self):
        """
        Main.
        """

        training_config = getattr(self.config, self.args.model)
        training_config.get_model = utils.get_get_model(self.args, self.config)
        training_config.directory = utils.get_training_directory(training_config, self.args)

        if self.args.summary:
            #training_config.summary_histograms = True
            training_config.summary_weights = True

        # fix path for finetune model
        if training_config.finetune is not None:
            training_config.finetune = utils.get_training_directory(training_config, self.args)

        training_config.validate()
        program = training_config.interface(training_config)
        program.main()


if __name__ == '__main__':
    program = Train()
    program.main()