import unittest
import torch
import numpy
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import models
import common.test
import common.train
import common.datasets
import common.torch
import common.numpy
import common.state
import common.eval
import common.utils
import attacks
import torch
import torch.utils.data
import math 


class TestNet(torch.nn.Module):
    def __init__(self, D=100, K=10, L=10):
        super(TestNet, self).__init__()
        self.L = L
        Ds = []
        for l in range(self.L):
            Ds.append(D + l)
        for l in range(self.L):
            linear = torch.nn.Linear(Ds[l], Ds[l], bias=False)
            torch.nn.init.uniform_(linear.weight, -1, 1)
            setattr(self, 'linear%d' % l, linear)
        self.logits = torch.nn.Linear(Ds[-1], K, bias=False)
        torch.nn.init.uniform_(self.logits.weight, -1, 1)
        # !
        self.linear0.weight.requires_grad = False

    def forward(self, inputs):
        for l in range(self.L):
            linear = getattr(self, 'linear%d' % l)
            inputs = linear(inputs)
        return self.logits(inputs)


class TestInitializationsWeights(unittest.TestCase):
    def testLInfUniformNormInitialization(self):
        model = TestNet()

        N = 5
        epsilon = 0.01
        for i in range(N):
            perturbed_model = common.torch.clone(model)
            layers = list(range(len(list(perturbed_model.parameters()))))
            norm = attacks.weights.norms.L2Norm()
            dist = norm(model, perturbed_model, layers)
            self.assertAlmostEqual(0, dist)
            initialization = attacks.weights.initializations.L2UniformNormInitialization(epsilon)
            initialization(model, perturbed_model, layers)
            dist = norm(model, perturbed_model, layers)
            self.assertGreaterEqual(epsilon, dist)
            self.assertGreater(dist, 0)


if __name__ == '__main__':
    unittest.main()
