import unittest
import torch
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import models


class TestModels(unittest.TestCase):
    def testResNet(self):
        resolutions = [
            [1, 28, 28],
            [3, 32, 32],
        ]
        blocks = [
            [3],
            [3, 3],
            [3, 3, 3],
            [3, 3, 3, 3],
        ]
        normalizations = [
            'bn',
            'fixedbn',
            'rebn',
        ]
        biases = [
            True,
            False,
        ]

        classes = 10
        batch_size = 100
        for resolution in resolutions:
            for block in blocks:
                for normalization in normalizations:
                    for bias in biases:
                        model = models.ResNet(classes, resolution, clamp=True, blocks=block, normalization=normalization, bias=bias)
                        output = model(torch.autograd.Variable(torch.zeros([batch_size] + resolution)))
                        self.assertEqual(output.size()[0], batch_size)
                        self.assertEqual(output.size()[1], classes)

                        if not bias:
                            for name, parameter in model.named_parameters():
                                if not (name.find('bn') or name.find('gn')):
                                    assert name.find('bias') < 0, name


if __name__ == '__main__':
    unittest.main()
