import unittest
import numpy
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import models
import common.test
import common.train
import common.datasets
import common.torch
import common.state
import common.eval
import common.hessian
from common.log import Timer
import torch
import torch.utils.data


class TestAttacksNormalMNISTMLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 100
        cls.cuda = True
        cls.setDatasets()

        if os.path.exists(cls.getModelFile()):
            state = common.state.State.load(cls.getModelFile())
            cls.model = state.model

            if cls.cuda:
                cls.model = cls.model.cuda()
        else:
            cls.model = cls.getModel()
            if cls.cuda:
                cls.model = cls.model.cuda()
            print(cls.model)

            optimizer = torch.optim.SGD(cls.model.parameters(), lr=0.1, momentum=0.9)
            scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(cls.trainloader))
            writer = common.summary.SummaryWriter()
            augmentation = None

            trainer = common.train.NormalTraining(cls.model, cls.trainloader, cls.testloader, optimizer, scheduler, augmentation=augmentation, writer=writer,
                                                  cuda=cls.cuda)
            for e in range(10):
                trainer.step(e)

            common.state.State.checkpoint(cls.getModelFile(), cls.model, optimizer, scheduler, e)

            cls.model.eval()
            probabilities = common.test.test(cls.model, cls.testloader, cuda=cls.cuda)
            eval = common.eval.CleanEvaluation(probabilities, cls.testloader.dataset.labels, validation=0)
            assert 0.075 > eval.test_error(), '0.05 !> %g' % eval.test_error()
            assert numpy.mean(numpy.max(probabilities, axis=1)) > 0.9

        cls.model.eval()

    @classmethod
    def setDatasets(cls):
        cls.trainset = common.datasets.MNISTTrainSet()
        cls.testset = common.datasets.MNISTTestSet()
        cls.adversarialset = common.datasets.MNISTTestSet(indices=range(100))
        cls.trainloader = torch.utils.data.DataLoader(cls.trainset, batch_size=cls.batch_size, shuffle=True, num_workers=0)
        cls.testloader = torch.utils.data.DataLoader(cls.testset, batch_size=cls.batch_size, shuffle=False, num_workers=0)
        cls.adversarialloader = torch.utils.data.DataLoader(cls.adversarialset, batch_size=cls.batch_size, shuffle=False, num_workers=0)

    @classmethod
    def getModelFile(cls):
        return 'test_hessian.pth.tar'

    @classmethod
    def getModel(cls):
        return models.MLP(10, [1, 28, 28], units=[64, 64], action=torch.nn.Sigmoid)

    def testMinMaxHessianEigs(self):
        # Mainly about testing that it runs, and not what the values are:
        criterion = torch.nn.CrossEntropyLoss()
        timer = Timer()
        eigs, count = common.hessian.min_max_k_hessian_eigs(self.model, self.adversarialloader, criterion, k=2, use_cuda=True, verbose=True)
        maxeig = eigs[-1]
        mineig = eigs[0]
        print(timer.elapsed(), maxeig, mineig)
        self.assertGreaterEqual(maxeig, 0)
        self.assertGreaterEqual(0, mineig)


if __name__ == '__main__':
    unittest.main()
