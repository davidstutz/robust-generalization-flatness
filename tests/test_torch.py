import unittest
import numpy
import torch
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import models
import common.torch
import common.datasets
import common.train
import common.test
import common.eval
from common.log import Timer


class TestNet2(torch.nn.Module):
    def __init__(self, D=100, K=10, L=5):
        super(TestNet2, self).__init__()
        self.L = L
        for l in range(self.L):
            linear = torch.nn.Linear(D, D)
            torch.nn.init.uniform_(linear.weight, -1, 1)
            setattr(self, 'linear%d' % l, linear)
        self.logits = torch.nn.Linear(D, K)
        torch.nn.init.uniform_(self.logits.weight, -1, 1)

    def forward(self, inputs):
        for l in range(self.L):
            linear = getattr(self, 'linear%d' % l)
            inputs = linear(inputs)
        return self.logits(inputs)


class TestTorch(unittest.TestCase):
    def testTopKSpeed(self):
        N = 1000000
        total_time = 0
        for i in range(50):
            tensor = torch.Tensor(N).uniform_(0, 1).cuda()
            sorted_tensor, sorted_indices = torch.sort(tensor, descending=True)
            timer = Timer()
            common.torch.topk(sorted_tensor, int(0.01*N))
            total_time += timer.elapsed()
        print(total_time/50)

    def testClone(self):
        model = TestNet2()
        model.logits.weight.requires_grad = False
        model.logits.bias.requires_grad = False

        requires_grad = [parameter.requires_grad for parameter in model.parameters()]
        assert not numpy.all(requires_grad)

        clone_model = common.torch.clone(model)
        clone_requires_grad = [parameter.requires_grad for parameter in clone_model.parameters()]
        assert not numpy.all(clone_requires_grad)
        print(requires_grad, clone_requires_grad)
        numpy.testing.assert_array_equal(requires_grad, clone_requires_grad)

    def testProjectBall(self):
        N = 1000
        epsilons = [4, 5, 15, 0.3]
        ords = [0, 1, 2, float('inf')]
        for i in range(len(ords)):
            ord = ords[i]
            epsilon = epsilons[i]
            for D in [10, 100]:
                original_data = torch.from_numpy(numpy.random.randn(N, D))
                original_norms = torch.norm(original_data, p=ord, dim=1)

                projected_data = common.torch.project_ball(original_data, ord=ord, epsilon=epsilon)
                projected_norms = torch.norm(projected_data, p=ord, dim=1)

                original_norms = original_norms.numpy()
                projected_norms = projected_norms.numpy()

                evaluated1 = numpy.logical_and(original_norms <= epsilon, numpy.abs(original_norms - projected_norms) <= 0.001)
                evaluated2 = numpy.logical_and(original_norms > epsilon, numpy.abs(projected_norms - epsilon) <= 0.0001)
                self.assertTrue(numpy.all(numpy.logical_or(evaluated1, evaluated2)))

    def testProjectBallL0Large(self):
        N = 10
        epsilon = 500
        D = 6000000
        ord = 0

        original_data = torch.from_numpy(numpy.random.randn(N, D)*0.01)
        original_norms = torch.norm(original_data, p=ord, dim=1)

        projected_data = common.torch.project_ball(original_data, ord=ord, epsilon=epsilon)
        projected_norms = torch.norm(projected_data, p=ord, dim=1)

        original_norms = original_norms.numpy()
        projected_norms = projected_norms.numpy()

        evaluated1 = numpy.logical_and(original_norms <= epsilon, numpy.abs(original_norms - projected_norms) <= 0.001)
        evaluated2 = numpy.logical_and(original_norms > epsilon, numpy.abs(projected_norms - epsilon) <= 0.0001)
        print(original_norms, projected_norms)
        self.assertTrue(numpy.all(numpy.logical_or(evaluated1, evaluated2)))

    def testUniformNorm(self):
        N = 10000
        epsilons = [10, 5, 25, 0.5]
        ords = [0, 2, 1, float('inf')]
        for o in range(len(ords)):
            ord = ords[o]
            epsilon = epsilons[o]
            for D in [50, 100, 1000]:
                samples = common.torch.uniform_norm(N, D, ord=ord, epsilon=epsilon)
                self.assertEqual(samples.shape[0], N)
                self.assertEqual(samples.shape[1], D)
                norms = numpy.linalg.norm(samples, ord=ord, axis=1)
                self.assertTrue(numpy.all(norms <= epsilon))

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
                samples = common.torch.uniform_ball(N, D, ord=ord, epsilon=epsilon)
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
                samples = common.torch.uniform_sphere(N, D, ord=ord, epsilon=epsilon)
                self.assertEqual(samples.shape[0], N)
                self.assertEqual(samples.shape[1], D)
                norms = numpy.linalg.norm(samples, ord=ord, axis=1)
                self.assertTrue(numpy.allclose(norms, epsilon))

    def testReparameterizedBatchNorm(self):
        batch_size = 100
        trainset = torch.utils.data.DataLoader(common.datasets.MNISTTrainSet(indices=range(10000)), batch_size=batch_size, shuffle=True, num_workers=4)
        testset = torch.utils.data.DataLoader(common.datasets.MNISTTestSet(), batch_size=batch_size, shuffle=False, num_workers=4)

        cuda = True
        model = models.MLP(10, [28, 28, 1], [64, 64], normalization='rebn')

        if cuda:
            model = model.cuda()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        trainer = common.train.NormalTraining(model, trainset, testset, optimizer, scheduler, augmentation=augmentation, writer=writer, cuda=cuda)

        epochs = 10
        for e in range(epochs):
            trainer.step(e)
            for name, parameter in model.named_parameters():
                if name.find('bn') >= 0:
                    print(name, torch.mean(parameter))

        probabilities = common.test.test(model, testset, cuda=cuda)
        eval = common.eval.CleanEvaluation(probabilities, testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.1, eval.test_error())
        print(eval.test_error())


if __name__ == '__main__':
    unittest.main()
