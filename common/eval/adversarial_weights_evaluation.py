import numpy
import torch


class AdversarialWeightsEvaluation:
    """
    Evaluation on adversarial and clean examples.
    """

    def __init__(self, clean_probabilities, adversarial_probabilities, labels):
        """
        Constructor.

        :param clean_probabilities: probabilities on clean examles
        :type clean_probabilities: numpy.ndarray
        :param adversarial_probabilities: probabilities on adversarial examples
        :type adversarial_probabilities: numpy.ndarray
        :param labels: labels
        :type labels: numpy.ndarray
        """

        labels = numpy.squeeze(labels)
        assert len(labels.shape) == 1
        assert len(clean_probabilities.shape) == 2
        assert clean_probabilities.shape[0] == labels.shape[0]
        assert clean_probabilities.shape[1] == numpy.max(labels) + 1
        assert len(adversarial_probabilities.shape) == len(clean_probabilities.shape)
        assert adversarial_probabilities.shape[1] == clean_probabilities.shape[1]
        assert adversarial_probabilities.shape[0] <= clean_probabilities.shape[0]

        self.reference_N = adversarial_probabilities.shape[0]
        """ (int) Reference N. """

        self.reference_probabilities = clean_probabilities[:self.reference_N]
        """ (numpy.ndarray) Test probabilities. """

        self.reference_labels = labels[:self.reference_N]
        """ (numpy.ndarray) Test labels. """

        self.reference_predictions = numpy.argmax(self.reference_probabilities, axis=1)
        """ (numpy.ndarray) Test predicted labels. """

        self.reference_errors = (self.reference_predictions != self.reference_labels)
        """ (numpy.ndarray) Test errors. """

        #
        self.test_N = clean_probabilities.shape[0]
        """ (int) Test N. """

        self.test_probabilities = clean_probabilities
        """ (numpy.ndarray) Test probabilities. """

        self.test_labels = labels
        """ (numpy.ndarray) Test labels. """

        self.test_predictions = numpy.argmax(self.test_probabilities, axis=1)
        """ (numpy.ndarray) Test predicted labels. """

        self.test_errors = (self.test_predictions != self.test_labels)
        """ (numpy.ndarray) Test errors. """

        #
        self.test_adversarial_probabilities = adversarial_probabilities
        """ (numpy.ndarray) Test probabilities. """

        self.test_adversarial_predictions = numpy.argmax(self.test_adversarial_probabilities, axis=1)
        """ (numpy.ndarray) Test predicted labels. """

        self.test_adversarial_errors = (self.test_adversarial_predictions != self.reference_labels)
        """ (numpy.ndarray) Test errors. """

    def test_error(self):
        """
        Test error.

        :return: test error
        :rtype: float
        """

        return numpy.sum(self.test_errors.astype(int)) / float(self.test_N)

    def reference_test_error(self):
        """
        Reference test error.

        :return: reference test error
        :rtype: float
        """

        return numpy.sum(self.reference_errors.astype(int)) / float(self.reference_N)

    def robust_test_error(self):
        """
        Robust test error.

        :return: robust test error
        :rtype: float
        """

        return numpy.sum(numpy.logical_or(self.test_adversarial_errors, self.reference_errors).astype(int)) / float(self.reference_N)

    def loss(self):
        """
        Loss.

        :param loss: loss to use
        :type loss: callable
        :return: loss
        :rtype: float
        """

        probabilities = torch.from_numpy(self.test_probabilities)
        labels = torch.from_numpy(self.test_labels)
        return torch.nn.functional.nll_loss(torch.log(probabilities), labels).item()

    def reference_loss(self):
        """
        Loss.

        :param loss: loss to use
        :type loss: callable
        :return: loss
        :rtype: float
        """

        probabilities = torch.from_numpy(self.reference_probabilities)
        labels = torch.from_numpy(self.reference_labels)
        return torch.nn.functional.nll_loss(torch.log(probabilities), labels).item()

    def robust_correct_loss(self):
        """
        Loss.

        :param loss: loss to use
        :type loss: callable
        :return: loss
        :rtype: float
        """

        probabilities = torch.from_numpy(self.test_adversarial_probabilities[numpy.logical_not(self.reference_errors)])
        labels = torch.from_numpy(self.reference_labels[numpy.logical_not(self.reference_errors)])
        return torch.nn.functional.nll_loss(torch.log(probabilities + 1e-12), labels).item()

    def robust_loss(self):
        """
        Loss.

        :param loss: loss to use
        :type loss: callable
        :return: loss
        :rtype: float
        """

        probabilities = torch.from_numpy(self.test_adversarial_probabilities)
        labels = torch.from_numpy(self.reference_labels)
        return torch.nn.functional.nll_loss(torch.log(probabilities + 1e-12), labels).item()

    def confidence(self):
        """
        Loss.

        :param loss: loss to use
        :type loss: callable
        :return: loss
        :rtype: float
        """

        return numpy.mean(numpy.max(self.test_probabilities, axis=1))

    def reference_confidence(self):
        """
        Loss.

        :param loss: loss to use
        :type loss: callable
        :return: loss
        :rtype: float
        """

        return numpy.mean(numpy.max(self.reference_probabilities, axis=1))

    def robust_confidence(self):
        """
        Loss.

        :param loss: loss to use
        :type loss: callable
        :return: loss
        :rtype: float
        """

        return numpy.mean(numpy.max(self.test_adversarial_probabilities, axis=1))
