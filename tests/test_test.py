import unittest
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import models
import common.test
import common.train
import common.datasets
import common.torch
import common.state
import common.imgaug
import common.eval
import common.summary
import torch
import torch.utils.data


class TestTest(unittest.TestCase):
    def setUp(self):
        self.trainset = torch.utils.data.DataLoader(common.datasets.MNISTTrainSet(indices=range(10000)), batch_size=100, shuffle=True, num_workers=4)
        self.testset = torch.utils.data.DataLoader(common.datasets.MNISTTestSet(), batch_size=100, num_workers=4, shuffle=False)
        self.adversarialset = torch.utils.data.DataLoader(common.datasets.MNISTTestSet(indices=range(1000)), batch_size=100, num_workers=4, shuffle=False)
        self.cuda = True

    def testTest(self):
        model = models.LeNet(10, [1, 28, 28], channels=12)

        if self.cuda:
            model = model.cuda()

        model.eval()
        probabilities = common.test.test(model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels)
        self.assertGreaterEqual(0.05, abs(0.9 - eval.test_error()))


if __name__ == '__main__':
    unittest.main()
