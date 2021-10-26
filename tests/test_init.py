import unittest
import torch
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import models


class TestInit(unittest.TestCase):
    def testLeNet(self):
        def test(scale, places=2):
            model = models.LeNet(10, [3, 32, 32], clamp=True, channels=256,
                                 activation='relu', normalization='bn', init_scale=scale)
 
            self.assertAlmostEqual(torch.std(model.conv0.weight).item(), 0.16*scale, places=places)
            self.assertAlmostEqual(torch.std(model.conv1.weight).item(), 0.017*scale, places=places)
            self.assertAlmostEqual(torch.std(model.conv2.weight).item(), 0.012*scale, places=places)
            self.assertAlmostEqual(torch.std(model.fc2.weight).item(), 0.011*scale, places=places)
            self.assertAlmostEqual(torch.std(model.logits.weight).item(), 0.043*scale, places=places)
        test(1, places=2)
        test(0.5, places=2)
        test(0.1, places=3)

    def testMLP(self):
        def test(scale, places=2):
            model = models.MLP(10, [3, 32, 32], units=[1024, 1024, 1024], clamp=True,
                                 activation='relu', normalization='bn', init_scale=scale)

            self.assertAlmostEqual(torch.std(model.lin1.weight).item(), 0.025 * scale, places=places)
            self.assertAlmostEqual(torch.std(model.lin2.weight).item(), 0.044 * scale, places=places)
            self.assertAlmostEqual(torch.std(model.lin3.weight).item(), 0.044 * scale, places=places)
            self.assertAlmostEqual(torch.std(model.logits.weight).item(), 0.044 * scale, places=places)

        test(1, places=2)
        test(0.5, places=2)
        test(0.1, places=3)


if __name__ == '__main__':
    unittest.main()
