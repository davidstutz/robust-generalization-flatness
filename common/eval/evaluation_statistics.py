import numpy


class EvaluationStatistics:
    """
    Statistics over evaluations.
    """

    def __init__(self, evaluations):
        """
        Constructor.

        :param evaluations: evaluations
        :type evaluations: [AdversarialWeightsEvaluation]
        """

        assert isinstance(evaluations, list)
        assert len(evaluations) > 0

        not_none = False
        first = None
        for t in range(len(evaluations)):
            if first is None and evaluations[t] is not None:
                first = t
            not_none = (evaluations[t] is not None) or not_none
        assert not_none

        self.evaluations = evaluations
        """ ([AdversarialWeightsEvaluation]) Evaluations. """

        self.first = first
        """ (int) First not-none element. """

    def __call__(self, metric, statistic='mean', factor=1, **kwargs):
        """
        Get statistic.

        :param metric: metric name
        :type metric: string
        :param statistic: statistic name
        :type statistic: str
        :return: statistic
        :rtype: float
        """

        metric_function = getattr(self.evaluations[self.first], metric, None)
        statistic_function = getattr(numpy, statistic, None)

        assert metric_function is not None
        assert statistic_function is not None

        values = []
        for evaluation in self.evaluations:
            if evaluation is not None:
                values.append(getattr(evaluation, metric)()*factor)

        return statistic_function(numpy.array(values), **kwargs), len(values)