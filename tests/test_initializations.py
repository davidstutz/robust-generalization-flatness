import unittest
import numpy
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import attacks
import torch
import torch.utils.data


class TestInitializations(unittest.TestCase):
    def testZeroInitialization(self):
        images = None
        perturbations = torch.from_numpy(numpy.random.uniform(0, 1, size=[100, 1, 32, 32]))
        perturbations = torch.autograd.Variable(perturbations, requires_grad=True)

        initializer = attacks.initializations.ZeroInitialization()
        initializer(images, perturbations)

        perturbations = perturbations.detach().cpu().numpy()
        norms = numpy.linalg.norm(perturbations.reshape(100, -1), ord=float('inf'), axis=1)
        self.assertTrue(numpy.allclose(perturbations, 0))
        self.assertAlmostEqual(numpy.sum(perturbations), 0)
        self.assertTrue(numpy.all(norms <= 0))

    def testLInfInitializations(self):
        epsilon = 0.3
        images = None
        perturbations = torch.from_numpy(numpy.zeros([100, 1, 32, 32]))
        perturbations = torch.autograd.Variable(perturbations, requires_grad=True)

        initializer = attacks.initializations.LInfUniformInitialization(epsilon)
        initializer(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=float('inf'), axis=1)
        self.assertTrue(numpy.all(norms <= epsilon))

        initializer = attacks.initializations.LInfUniformNormInitialization(epsilon)
        initializer(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=float('inf'), axis=1)
        self.assertTrue(numpy.all(norms <= epsilon))

        initializer = attacks.initializations.LInfUniformSphereInitialization(epsilon)
        initializer(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=float('inf'), axis=1)
        self.assertTrue(numpy.allclose(norms, epsilon))

    def testL2Initializations(self):
        epsilon = 0.3
        images = None
        perturbations = torch.from_numpy(numpy.zeros([100, 1, 32, 32]))
        perturbations = torch.autograd.Variable(perturbations, requires_grad=True)

        initializer = attacks.initializations.L2UniformNormInitialization(epsilon)
        initializer(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=2, axis=1)
        self.assertTrue(numpy.all(norms <= epsilon))

        initializer = attacks.initializations.L2UniformSphereInitialization(epsilon)
        initializer(images, perturbations)

        norms = numpy.linalg.norm(perturbations.detach().cpu().numpy().reshape(100, -1), ord=2, axis=1)
        self.assertTrue(numpy.allclose(norms, epsilon))


if __name__ == '__main__':
    unittest.main()
