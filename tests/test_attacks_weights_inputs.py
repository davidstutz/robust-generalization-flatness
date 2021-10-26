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
import attacks.weights_inputs
import torch
import torch.utils.data
from common.progress import ProgressBar


class TestAttacksWeightsInputsMNISTMLP(unittest.TestCase):
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
        cls.adversarialset = common.datasets.MNISTTestSet(indices=range(500))
        cls.trainloader = torch.utils.data.DataLoader(cls.trainset, batch_size=cls.batch_size, shuffle=True, num_workers=0)
        cls.testloader = torch.utils.data.DataLoader(cls.testset, batch_size=cls.batch_size, shuffle=False, num_workers=0)
        cls.adversarialloader = torch.utils.data.DataLoader(cls.adversarialset, batch_size=cls.batch_size, shuffle=False, num_workers=0)

    @classmethod
    def getModelFile(cls):
        return 'test_attacks_weights_inputs_mnist_mlp.pth.tar'

    @classmethod
    def getModel(cls):
        return models.MLP(10, [1, 28, 28], units=[64, 64], action=torch.nn.Sigmoid)

    def _testAttackPerformance(self, attack, attempts=5, weight_objective=attacks.weights.objectives.UntargetedF0Objective(), input_objective=attacks.objectives.UntargetedF0Objective()):
        error_rate = 0
        for t in range(attempts):
            attack.progress = ProgressBar()
            perturbed_model, perturbations, errors = attack.run(self.model, self.adversarialloader, weight_objective, input_objective)

            adversarialset = common.datasets.AdversarialDataset(self.adversarialloader.dataset.images, numpy.expand_dims(perturbations, axis=0), self.adversarialloader.dataset.labels)
            adversarialloader = torch.utils.data.DataLoader(adversarialset, batch_size=self.adversarialloader.batch_size, shuffle=False)

            perturbed_model = perturbed_model.cuda()
            probabilities = common.test.test(perturbed_model, adversarialloader, cuda=self.cuda)
            eval = common.eval.CleanEvaluation(probabilities, adversarialloader.dataset.labels)
            error_rate += eval.test_error()

        error_rate /= attempts
        return error_rate

    def testSequentialAttack2(self):
        weight_attack = attacks.weights.GradientDescentAttack()
        weight_attack.epochs = 5
        weight_attack.base_lr = 0.1
        weight_attack.normalization = None
        weight_attack.backtrack = False
        weight_attack.momentum = 0
        weight_attack.lr_factor = 1
        weight_attack.initialization = attacks.weights.initializations.L2UniformSphereInitialization(relative_epsilon=0.02)
        weight_attack.norm = attacks.weights.norms.L2Norm()
        weight_attack.projection = attacks.weights.SequentialProjections([
            attacks.weights.projections.L2Projection(relative_epsilon=0.02)
        ])

        input_attack = attacks.BatchGradientDescent()
        input_attack.max_iterations = 5
        input_attack.base_lr = 0.1
        input_attack.momentum = 0
        input_attack.lr_factor = 1
        input_attack.backtrack = False
        input_attack.normalized = True
        input_attack.c = 0
        input_attack.initialization = attacks.initializations.LInfUniformNormInitialization(0.3)
        input_attack.projection = attacks.projections.SequentialProjections([
            attacks.projections.BoxProjection(0, 1),
            attacks.projections.LInfProjection(0.3)
        ])
        input_attack.norm = attacks.norms.LInfNorm()

        sequential_attack = attacks.weights_inputs.SequentialAttack2(weight_attack, input_attack)
        test_error = self._testAttackPerformance(sequential_attack)
        print(test_error)

    def testGradientDescentAttack(self):
        attack = attacks.weights_inputs.GradientDescentAttack()
        attack.epochs = 10
        attack.weight_initialization = attacks.weights.initializations.LInfUniformNormInitialization(relative_epsilon=0.02)
        attack.weight_projection = attacks.weights.projections.L2Projection(relative_epsilon=0.02)
        attack.weight_norm = attacks.weights.norms.L2Norm()
        attack.weight_normalization = attacks.weights.normalizations.L2Normalization()
        attack.input_initialization = attacks.initializations.LInfUniformNormInitialization(0.0002)
        attack.input_projection = attacks.projections.SequentialProjections([
            attacks.projections.BoxProjection(0, 1),
            attacks.projections.LInfProjection(0.0002),
        ])
        attack.input_norm = attacks.norms.LInfNorm()
        attack.input_normalized = True
        attack.input_lr = 0.1
        attack.weight_lr = 1
        test_error = self._testAttackPerformance(attack)
        print(test_error)


if __name__ == '__main__':
    unittest.main()
