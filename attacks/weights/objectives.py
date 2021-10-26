import torch
import common.torch


class Objective:
    def __call__(self, logits, targets):
        """
        Objective function.

        :param logits: logit output of the network
        :type logits: torch.autograd.Variable
        :param targets: perturbations
        :type targets: torch.autograd.Variable or None
        :return: error
        :rtype: torch.autograd.Variable
        """

        raise NotImplementedError()

    def reset(self):
        pass

    def success(self, logits, targets):
        """
        Success.

        :param logits: logit output of the network
        :type logits: torch.autograd.Variable
        :param targets: perturbations
        :type targets: torch.autograd.Variable or None
        :return: error
        :rtype: torch.autograd.Variable
        """

        return common.torch.classification_error(logits, targets)

    def true_confidence(self, logits, targets):
        """
        True confidence.

        :param logits: logit output of the network
        :type logits: torch.autograd.Variable
        :param targets: perturbations
        :type targets: torch.autograd.Variable or None
        :return: true confidence
        :rtype: float
        """

        probabilities = common.torch.softmax(logits, dim=1)
        return torch.mean(probabilities[torch.arange(logits.size(0)).long(), targets])

    def target_confidence(self, logits, targets):
        """
        True confidence.

        :param logits: logit output of the network
        :type logits: torch.autograd.Variable
        :param targets: perturbations
        :type targets: torch.autograd.Variable or None
        :return: true confidence
        :rtype: float
        """

        probabilities = common.torch.softmax(logits, dim=1)
        probabilities[torch.arange(logits.size(0)).long(), targets] = 0
        target_classes = torch.max(probabilities, dim=1)[1]
        return torch.mean(probabilities[torch.arange(logits.size(0)).long(), target_classes])


class UntargetedF0Objective(Objective):
    def __init__(self, loss=common.torch.classification_loss):
        """
        Constructor.

        :param loss: loss function to use
        :type loss: callable
        """

        super(UntargetedF0Objective, self).__init__()

        self.loss = loss
        """ (callable) Loss. """

    def __call__(self, logits, targets):
        """
        Objective function.

        :param logits: logit output of the network
        :type logits: torch.autograd.Variable
        :param targets: perturbations
        :type targets: torch.autograd.Variable or None
        :return: error
        :rtype: torch.autograd.Variable
        """

        assert self.loss is not None

        return -self.loss(logits, targets)
