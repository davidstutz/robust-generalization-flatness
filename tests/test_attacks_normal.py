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
import attacks
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
        return 'test_attacks_normal_mnist_mlp.pth.tar'

    @classmethod
    def getModel(cls):
        return models.MLP(10, [1, 28, 28], units=[64, 64], action=torch.nn.Sigmoid)

    def successRate(self, images, perturbations, labels):
        adversarialloader = torch.utils.data.DataLoader(common.datasets.AdversarialDataset(images, perturbations, labels), batch_size=100)
        testloader = torch.utils.data.DataLoader(self.adversarialset, batch_size=100, shuffle=False)
        self.assertEqual(len(adversarialloader), len(testloader))

        # assumes one attempt only!
        clean_probabilities = common.test.test(self.model, testloader, cuda=self.cuda)
        adversarial_probabilities = numpy.array([common.test.test(self.model, adversarialloader, cuda=self.cuda)])

        eval = common.eval.AdversarialEvaluation(clean_probabilities, adversarial_probabilities, labels, validation=0)
        return eval.success_rate()

    def runTestAttackProjections(self, attack, epsilon, objective=attacks.objectives.UntargetedF0Objective()):
        for b, (images, labels) in enumerate(self.testloader):
            break

        images = common.torch.as_variable(images, self.cuda).permute(0, 3, 1, 2)
        labels = common.torch.as_variable(labels, self.cuda)

        objective.set(labels)
        perturbations, errors = attack.run(self.model, images, objective)
        norms = numpy.linalg.norm(perturbations.reshape(self.batch_size, -1), ord=float('inf'), axis=1)

        perturbed_images = images.cpu().numpy() + perturbations
        self.assertTrue(numpy.all(perturbed_images <= 1 + 1e-6), 'max value=%g' % numpy.max(perturbed_images))
        self.assertTrue(numpy.all(perturbed_images >= 0 - 1e-6), 'min value=%g' % numpy.min(perturbed_images))
        self.assertTrue(numpy.all(norms <= epsilon + 1e-6), 'max norm=%g epsilon=%g' % (numpy.max(norms), epsilon))

    def runTestAttackPerformance(self, attack, attempts=5, objective=attacks.objectives.UntargetedF0Objective()):
        for b, (images, labels) in enumerate(self.adversarialloader):
            break

        images = common.torch.as_variable(images, self.cuda).permute(0, 3, 1, 2)
        labels = common.torch.as_variable(labels, self.cuda)

        success_rate = 0
        for t in range(attempts):
            objective.set(labels)
            perturbations, errors = attack.run(self.model, images, objective)

            perturbations = numpy.array([numpy.transpose(perturbations, (0, 2, 3, 1))])
            success_rate += self.successRate(numpy.transpose(images.cpu().numpy(), (0, 2, 3, 1)), perturbations, labels.cpu().numpy())

        success_rate /= attempts
        return success_rate

    def testBatchGradientDescentStep(self):
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 1
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.normalized = False
        attack.backtrack = False
        attack.initialization = None
        attack.projection = None
        attack.norm = attacks.norms.LInfNorm()

        for i in range(10):
            image, label = self.testloader.dataset[i]

            images = common.torch.as_variable(torch.from_numpy(numpy.expand_dims(image, axis=0)), self.cuda)
            images = images.permute(0, 3, 1, 2)
            labels = common.torch.as_variable(torch.from_numpy(numpy.array([label])), self.cuda)
            objective = attacks.objectives.UntargetedF0Objective()
            objective.set(labels)

            # pre-compute step
            manual_perturbations = torch.zeros_like(images)
            manual_perturbations.requires_grad = True
            manual_logits = self.model(images + manual_perturbations)
            manual_error = objective(manual_logits)
            manual_error.backward()

            manual_gradients = manual_perturbations.grad
            next_perturbations = manual_perturbations - attack.base_lr * manual_gradients

            next_logits = self.model(images + next_perturbations)
            next_error = objective(next_logits)
            if next_error.item() < manual_error.item():
                manual_perturbations = next_perturbations
            manual_perturbations = manual_perturbations.detach().cpu().numpy()

            # now run attack
            attack_perturbations, _ = attack.run(self.model, images, objective)

            numpy.testing.assert_allclose(manual_perturbations, attack_perturbations, atol=1e-4)

    def testBatchGradientDescentStepNormalized(self):
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 1
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.normalized = True
        attack.backtrack = False
        attack.initialization = None
        attack.projection = None
        attack.norm = attacks.norms.LInfNorm()

        for i in range(10):
            image, label = self.testloader.dataset[i]

            images = common.torch.as_variable(torch.from_numpy(numpy.expand_dims(image, axis=0)), self.cuda)
            images = images.permute(0, 3, 1, 2)
            labels = common.torch.as_variable(torch.from_numpy(numpy.array([label])), self.cuda)
            objective = attacks.objectives.UntargetedF0Objective()
            objective.set(labels)

            # pre-compute step
            manual_perturbations = torch.zeros_like(images)
            manual_perturbations.requires_grad = True
            manual_logits = self.model(images + manual_perturbations)
            manual_error = objective(manual_logits)
            manual_error.backward()

            manual_gradients = manual_perturbations.grad
            next_perturbations = manual_perturbations - attack.base_lr*torch.sign(manual_gradients)

            next_logits = self.model(images + next_perturbations)
            next_error = objective(next_logits)
            if next_error.item() < manual_error.item():
                manual_perturbations = next_perturbations
            manual_perturbations = manual_perturbations.detach().cpu().numpy()

            # now run attack
            attack_perturbations, _ = attack.run(self.model, images, objective)

            numpy.testing.assert_allclose(manual_perturbations, attack_perturbations, atol=1e-4)

    def testBatchGradientDescentNormalized(self):
        epsilon = 0.3
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 10
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.normalized = True
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.9)

    def testBatchGradientDescentNormalizedZero(self):
        epsilon = 0.3
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 10
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.normalized = True
        attack.initialization = None
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.85)

    def testBatchGradientDescentNormalizedBacktrack(self):
        epsilon = 0.3
        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 5
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        self.runTestAttackProjections(attack, epsilon)
        success_rate = self.runTestAttackPerformance(attack)
        self.assertGreaterEqual(success_rate, 0.9)

    def testBatchGradientDescentOptimizationRelative(self):
        success_rates = []
        epsilon = 0.3

        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 20
        attack.base_lr = 100
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        success_rates.append(self.runTestAttackPerformance(attack))

        attack = attacks.batch_gradient_descent.BatchGradientDescent()
        attack.max_iterations = 10
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.backtrack = False
        attack.normalized = True
        attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        success_rates.append(self.runTestAttackPerformance(attack))

        for i in range(len(success_rates) - 1):
            self.assertGreaterEqual(success_rates[i + 1], success_rates[i])
        self.assertGreaterEqual(success_rates[-1], 0.95)


class TestAttacksNormalMNISTLeNet(TestAttacksNormalMNISTMLP):
    @classmethod
    def getModelFile(cls):
        return 'test_attacks_normal_mnist_lenet.pth.tar'

    @classmethod
    def getModel(cls):
        return models.LeNet(10, [1, 28, 28], channels=12)


class TestAttacksNormalMNISTResNet(TestAttacksNormalMNISTMLP):
    @classmethod
    def getModelFile(cls):
        return 'test_attacks_normal_mnist_resnet.pth.tar'

    @classmethod
    def getModel(cls):
        return models.ResNet(10, [1, 28, 28], blocks=[1, 1, 1], channels=4)


class TestAttacksNormalMNISTNormalizedResNet(TestAttacksNormalMNISTMLP):
    @classmethod
    def getModelFile(cls):
        return 'test_attacks_normal_mnist_normalized_resnet.pth.tar'

    @classmethod
    def getModel(cls):
        model = models.ResNet(10, [1, 28, 28], blocks=[1, 1, 1], whiten=True, channels=4)

        mean = numpy.zeros(1)
        std = numpy.zeros(1)
        mean[0] = numpy.mean(cls.trainset.images[:, :, :, 0])
        std[0] = numpy.std(cls.trainset.images[:, :, :, 0])

        model.whiten.weight.data = torch.from_numpy(std.astype(numpy.float32))
        model.whiten.bias.data = torch.from_numpy(mean.astype(numpy.float32))

        return model


if __name__ == '__main__':
    unittest.main()
