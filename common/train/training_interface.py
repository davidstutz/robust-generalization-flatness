import common.numpy
from common.progress import ProgressBar


class TrainingInterface:
    """
    Training interface.

    Will only take care of the training and test step. This means that loading of model, optimizer,
    scheduler as well as data (including data augmentation in form of a dataset) needs to be
    taken care of separately.
    """

    def __init__(self, writer=common.summary.SummaryWriter()):
        """
        Constructor.

        :param writer: summary writer
        :type writer: torch.utils.tensorboard.SummaryWriter or TensorboardX equivalent
        """

        self.writer = writer
        """ (torch.util.tensorboard.SummaryWriter or equivalent) Summary writer. """

        self.progress = ProgressBar()
        """ (Timer) """

    def train(self, epoch):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """

        raise NotImplementedError()

    def test(self, epoch):
        """
        Test step.

        :param epoch: epoch
        :type epoch: int
        :return: probabilities of test set, model
        :rtype: numpy.array, torch.nn.Module
        """

        raise NotImplementedError()

    def step(self, epoch):
        """
        Training + test step.

        :param epoch: epoch
        :type epoch: int
        :return: probabilities of test set
        :rtype: numpy.array
        """

        self.train(epoch)
        return self.test(epoch)