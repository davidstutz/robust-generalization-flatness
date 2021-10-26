import unittest
import numpy
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import common.numpy


class TestNumpy(unittest.TestCase):
    def testOneHot(self):
        labels = numpy.random.randint(0, 9, size=(1000))
        classes = 10
        one_hot = common.numpy.one_hot(labels, classes)
        self.assertEqual(one_hot.shape[0], labels.shape[0])
        self.assertEqual(one_hot.shape[1], classes)
        for i in range(labels.shape[0]):
            for d in range(classes):
                if d == labels[i]:
                    self.assertEqual(1, one_hot[i, d])
                else:
                    self.assertEqual(0, one_hot[i, d])

    def testUniformNorm(self):
        N = 10000
        epsilons = [10, 5, 25, 0.5]
        ords = [0, 2, 1, float('inf')]
        for o in range(len(ords)):
            ord = ords[o]
            epsilon = epsilons[o]
            for D in [50, 100, 1000]:
                samples = common.numpy.uniform_norm(N, D, ord=ord, epsilon=epsilon)
                self.assertEqual(samples.shape[0], N)
                self.assertEqual(samples.shape[1], D)
                norms = numpy.linalg.norm(samples, ord=ord, axis=1)
                self.assertTrue(numpy.all(norms < epsilon))

                bins = 10
                histogram = [0]*bins
                for b in range(bins):
                    histogram[b] = numpy.sum(numpy.logical_and(norms < epsilon/bins*(b + 1), norms >= epsilon/bins*b if b > 0 else 0))/float(norms.shape[0])

                for b in range(bins):
                    self.assertGreaterEqual(0.125, histogram[b])

    def testUniformBall(self):
        N = 1000
        epsilon = 0.3
        ords = [2, float('inf')]
        for ord in ords:
            for D in [2, 3, 5, 10, 100]:
                samples = common.numpy.uniform_ball(N, D, ord=ord, epsilon=epsilon)
                self.assertEqual(samples.shape[0], N)
                self.assertEqual(samples.shape[1], D)
                norms = numpy.linalg.norm(samples, ord=ord, axis=1)
                self.assertTrue(numpy.all(norms < epsilon))

    def testUniformSphere(self):
        N = 10000
        epsilons = [10, 5, 25, 0.5]
        ords = [0, 2, 1, float('inf')]
        for o in range(len(ords)):
            ord = ords[o]
            epsilon = epsilons[o]
            for D in [50, 100, 1000]:
                samples = common.numpy.uniform_sphere(N, D, ord=ord, epsilon=epsilon)
                self.assertEqual(samples.shape[0], N)
                self.assertEqual(samples.shape[1], D)
                norms = numpy.linalg.norm(samples, ord=ord, axis=1)
                self.assertTrue(numpy.allclose(norms, epsilon))

    def testProjectBall(self):
        N = 1000
        epsilons = [4, 5, 15, 0.3]
        ords = [1, 2, float('inf')]
        for i in range(len(ords)):
            ord = ords[i]
            epsilon = epsilons[i]
            for D in [2, 3, 5, 10, 100]:
                original_data = numpy.random.randn(N, D)
                original_norms = numpy.linalg.norm(original_data, ord=ord, axis=1)

                projected_data = common.numpy.project_ball(original_data, ord=ord, epsilon=epsilon)
                projected_norms = numpy.linalg.norm(projected_data, ord=ord, axis=1)

                evaluated1 = numpy.logical_and(original_norms <= epsilon, numpy.abs(original_norms - projected_norms) <= 0.001)
                evaluated2 = numpy.logical_and(original_norms > epsilon, numpy.abs(projected_norms - epsilon) <= 0.0001)
                self.assertTrue(numpy.all(numpy.logical_or(evaluated1, evaluated2)))

    def testProjectSphere(self):
        N = 1000
        epsilon = 0.3
        ords = [2, float('inf')]
        for ord in ords:
            for D in [2, 3, 5, 10, 100]:
                original_data = numpy.random.randn(N, D) * 2
                projected_data = common.numpy.project_sphere(original_data, ord=ord, epsilon=epsilon)
                projected_norms = numpy.linalg.norm(projected_data, ord=ord, axis=1)
                self.assertTrue(numpy.allclose(projected_norms, epsilon))


if __name__ == '__main__':
    unittest.main()
