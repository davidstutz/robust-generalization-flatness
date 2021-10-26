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
import common.progress
import attacks
import attacks.weights
import torch
import torch.utils.data


class TestAttacksWeightsMNISTMLP(unittest.TestCase):
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
            assert 0.1 > eval.test_error(), '0.1 !> %g' % eval.test_error()
            assert numpy.mean(numpy.max(probabilities, axis=1)) > 0.9

        cls.model.eval()

    @classmethod
    def setDatasets(cls):
        cls.trainset = common.datasets.MNISTTrainSet(indices=range(10000))
        cls.testset = common.datasets.MNISTTestSet(indices=range(1000))
        cls.adversarialset = common.datasets.MNISTTestSet(indices=range(100))
        cls.trainloader = torch.utils.data.DataLoader(cls.trainset, batch_size=cls.batch_size, shuffle=True, num_workers=0)
        cls.testloader = torch.utils.data.DataLoader(cls.testset, batch_size=cls.batch_size, shuffle=False, num_workers=0)
        cls.adversarialloader = torch.utils.data.DataLoader(cls.adversarialset, batch_size=cls.batch_size, shuffle=False, num_workers=0)

    @classmethod
    def getModelFile(cls):
        return 'test_attacks_weights_mnist_mlp.pth.tar'

    @classmethod
    def getModel(cls):
        return models.MLP(10, [1, 28, 28], units=[64, 64], action=torch.nn.ReLU)

    def _testAttackPerformance(self, attack, attempts=5, objective=attacks.weights.objectives.UntargetedF0Objective()):
        error_rate = 0
        for t in range(attempts):
            perturbed_model = attack.run(self.model, self.adversarialloader, objective)
            perturbed_model = perturbed_model.cuda()

            probabilities = common.test.test(perturbed_model, self.adversarialloader, cuda=self.cuda)
            eval = common.eval.CleanEvaluation(probabilities, self.adversarialset.labels)
            error_rate += eval.test_error()

        error_rate /= attempts
        return error_rate

    def testGradientDescentAttackL2(self):
        epsilon = 100
        attack = attacks.weights.GradientDescentAttack()
        attack.progress = common.progress.ProgressBar()
        attack.epochs = 10
        attack.base_lr = 0.1
        attack.momentum = 0.9
        attack.lr_factor = 1
        attack.normalized = False
        attack.backtrack = False
        attack.initialization = attacks.weights.initializations.L2UniformNormInitialization(epsilon)
        attack.projection = attacks.weights.projections.L2Projection(epsilon)
        attack.norm = attacks.weights.norms.L2Norm()

        error_rate = self._testAttackPerformance(attack)
        self.assertGreaterEqual(error_rate, 0.8)

    def testGradientDescentAttackBacktrackL2(self):
        epsilon = 10
        attack = attacks.weights.GradientDescentAttack()
        attack.progress = common.progress.ProgressBar()
        attack.epochs = 100
        attack.base_lr = 0.05
        attack.momentum = 0.9
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.weights.initializations.L2UniformNormInitialization(epsilon)
        attack.projection = attacks.weights.projections.L2Projection(epsilon)
        attack.norm = attacks.weights.norms.L2Norm()

        error_rate = self._testAttackPerformance(attack)
        self.assertGreaterEqual(error_rate, 0.8)

    def testRandomAttackL2(self):
        error_rates = []
        for epsilon in [10, 100]:
            attack = attacks.weights.RandomAttack()
            attack.progress = common.progress.ProgressBar()
            attack.epochs = 10
            attack.initialization = attacks.weights.initializations.L2UniformSphereInitialization(epsilon)
            attack.projection = attacks.weights.projections.LayerWiseL2Projection(epsilon)
            attack.norm = attacks.weights.norms.L2Norm()
            error_rates.append(self._testAttackPerformance(attack))

        for error_rate in error_rates:
            self.assertGreaterEqual(error_rate, 0.01)
        for i in range(1, len(error_rates)):
            self.assertGreaterEqual(error_rates[i], error_rates[i - 1])


class TestAttacksWeightsMNISTResNet(TestAttacksWeightsMNISTMLP):
    @classmethod
    def getModelFile(cls):
        return 'test_attacks_weights_mnist_resnet.pth.tar'

    @classmethod
    def getModel(cls):
        return models.ResNet(10, [1, 28, 28], blocks=[1, 1, 1], channels=4)


class TestAttacksWeightsMNISTNormalizedResNet(TestAttacksWeightsMNISTMLP):
    @classmethod
    def getModelFile(cls):
        return 'test_attacks_weights_mnist_normalized_resnet.pth.tar'

    @classmethod
    def getModel(cls):
        model = models.ResNet(10, [1, 28, 28], blocks=[1, 1, 1], whiten=True, channels=8)

        mean = numpy.zeros(1)
        std = numpy.zeros(1)
        mean[0] = numpy.mean(cls.trainset.images[:, :, :, 0])
        std[0] = numpy.std(cls.trainset.images[:, :, :, 0])

        model.whiten.weight.data = torch.from_numpy(std.astype(numpy.float32))
        model.whiten.bias.data = torch.from_numpy(mean.astype(numpy.float32))

        return model


class TestAttacksWeightsMNISTScaledResNet(TestAttacksWeightsMNISTMLP):
    @classmethod
    def getModelFile(cls):
        return 'test_attacks_weights_mnist_scaled_resnet.pth.tar'

    @classmethod
    def getModel(cls):
        return models.ResNet(10, [1, 28, 28], blocks=[1, 1, 1], scale=True, channels=4)


if __name__ == '__main__':
    unittest.main()
