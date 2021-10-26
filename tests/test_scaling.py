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
import common.eval
import common.hessian
import torch
import torch.utils.data
import numpy
import common.scaling


class TestScaling(unittest.TestCase):
    def train(self, model, model_file, cuda, batch_size=100):
        trainset = common.datasets.MNISTTrainSet(list(range(10000)))
        testset = common.datasets.MNISTTestSet(list(range(1000)))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

        if os.path.exists(model_file):
            state = common.state.State.load(model_file)
            model = state.model

            if cuda:
                model = model.cuda()
        else:
            if cuda:
                model = model.cuda()
            print(model)

            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(trainloader))
            writer = common.summary.SummaryWriter()
            augmentation = None

            trainer = common.train.NormalTraining(model, trainloader, testloader, optimizer, scheduler, augmentation=augmentation, writer=writer,
                                                  cuda=cuda)
            for e in range(5):
                trainer.step(e)

            common.state.State.checkpoint(model_file, model, optimizer, scheduler, e)

            model.eval()
            probabilities = common.test.test(model, testloader, cuda=cuda)
            eval = common.eval.CleanEvaluation(probabilities, testloader.dataset.labels, validation=0)
            #assert 0.075 > eval.test_error(), '0.05 !> %g' % eval.test_error()
            #assert numpy.mean(numpy.max(probabilities, axis=1)) > 0.9

        model.eval()
        return model

    def testMLPScaling(self):
        batch_size = 100
        cuda = True
        testset = common.datasets.MNISTTestSet(indices=list(range(1000)))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
        model = models.MLP(10, [1, 28, 28], units=[100, 100, 100], activation='relu', normalization='')
        model = self.train(model, 'test_scaling_mlp.pth.tar', cuda=cuda)
        print(model)

        for factor in [2]:
            logits = common.test.logits(model, testloader, cuda=cuda)
            probabilities = common.test.test(model, testloader, cuda=cuda)
            predictions = numpy.argmax(probabilities, axis=1)

            scaled_model = common.torch.clone(model)
            common.scaling.scale(scaled_model, factor)

            scaled_logits = common.test.logits(scaled_model, testloader, cuda=cuda)
            scaled_probabilities = common.test.test(scaled_model, testloader, cuda=cuda)
            scaled_predictions = numpy.argmax(scaled_probabilities, axis=1)

            indices = numpy.where(predictions != scaled_predictions)
            indices = indices[0]
            print(indices)
            print(factor, numpy.sum(numpy.sum(logits > 0, axis=1) > 1), numpy.sum(predictions != scaled_predictions))

    def testMLPScalingBN(self):
        batch_size = 100
        cuda = True
        testset = common.datasets.MNISTTestSet(indices=list(range(1000)))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
        model = models.MLP(10, [1, 28, 28], units=[100, 100, 100], activation='relu', normalization='bn')
        model = self.train(model, 'test_scaling_mlp_bn.pth.tar', cuda=cuda)
        print(model)

        for factor in [2]:
            logits = common.test.logits(model, testloader, cuda=cuda)
            probabilities = common.test.test(model, testloader, cuda=cuda)
            predictions = numpy.argmax(probabilities, axis=1)

            scaled_model = common.torch.clone(model)
            common.scaling.scale(scaled_model, factor, ignore=['bn'])

            scaled_logits = common.test.logits(scaled_model, testloader, cuda=cuda)
            scaled_probabilities = common.test.test(scaled_model, testloader, cuda=cuda)
            scaled_predictions = numpy.argmax(scaled_probabilities, axis=1)

            indices = numpy.where(predictions != scaled_predictions)
            indices = indices[0]
            print(indices)
            print(factor, numpy.sum(numpy.sum(logits > 0, axis=1) > 1), numpy.sum(predictions != scaled_predictions))

    def testMLPScalingFixedBN(self):
        batch_size = 100
        cuda = True
        testset = common.datasets.MNISTTestSet(indices=list(range(1000)))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
        model = models.MLP(10, [1, 28, 28], units=[100, 100, 100], activation='relu', normalization='fixedbn')
        model = self.train(model, 'test_scaling_mlp_fixedbn.pth.tar', cuda=cuda)
        print(model)

        for factor in [2]:
            logits = common.test.logits(model, testloader, cuda=cuda)
            probabilities = common.test.test(model, testloader, cuda=cuda)
            predictions = numpy.argmax(probabilities, axis=1)

            scaled_model = common.torch.clone(model)
            common.scaling.scale(scaled_model, factor)

            scaled_logits = common.test.logits(scaled_model, testloader, cuda=cuda)
            scaled_probabilities = common.test.test(scaled_model, testloader, cuda=cuda)
            scaled_predictions = numpy.argmax(scaled_probabilities, axis=1)

            indices = numpy.where(predictions != scaled_predictions)
            indices = indices[0]
            print(indices)
            print(factor, numpy.sum(numpy.sum(logits > 0, axis=1) > 1), numpy.sum(predictions != scaled_predictions))
            print(logits[indices[:5], :])
            print(scaled_logits[indices[:5], :])

    def testMLPScalingReBN(self):
        batch_size = 100
        cuda = True
        testset = common.datasets.MNISTTestSet(indices=list(range(1000)))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
        model = models.MLP(10, [1, 28, 28], units=[100, 100, 100], activation='relu', normalization='rebn')
        model = self.train(model, 'test_scaling_mlp_rebn.pth.tar', cuda=cuda)
        print(model)

        for factor in [2]:
            logits = common.test.logits(model, testloader, cuda=cuda)
            probabilities = common.test.test(model, testloader, cuda=cuda)
            predictions = numpy.argmax(probabilities, axis=1)

            scaled_model = common.torch.clone(model)
            common.scaling.scale(scaled_model, factor, ignore=['rebn'])

            scaled_logits = common.test.logits(scaled_model, testloader, cuda=cuda)
            scaled_probabilities = common.test.test(scaled_model, testloader, cuda=cuda)
            scaled_predictions = numpy.argmax(scaled_probabilities, axis=1)

            indices = numpy.where(predictions != scaled_predictions)
            indices = indices[0]
            print(indices)
            print(factor, numpy.sum(numpy.sum(logits > 0, axis=1) > 1), numpy.sum(predictions != scaled_predictions))
            print(logits[indices[:5], :])
            print(scaled_logits[indices[:5], :])

    def testLeNetScalingReBN(self):
        batch_size = 100
        cuda = True
        testset = common.datasets.MNISTTestSet(indices=list(range(1000)))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
        model = models.LeNet(10, [1, 28, 28], channels=32, activation='relu', linear=False, normalization='rebn')
        model = self.train(model, 'test_scaling_lenet_rebn.pth.tar', cuda=cuda)
        print(model)

        for factor in [2]:
            logits = common.test.logits(model, testloader, cuda=cuda)
            probabilities = common.test.test(model, testloader, cuda=cuda)
            predictions = numpy.argmax(probabilities, axis=1)

            scaled_model = common.torch.clone(model)
            common.scaling.scale(scaled_model, factor, ignore=['rebn'])

            scaled_logits = common.test.logits(scaled_model, testloader, cuda=cuda)
            scaled_probabilities = common.test.test(scaled_model, testloader, cuda=cuda)
            scaled_predictions = numpy.argmax(scaled_probabilities, axis=1)

            indices = numpy.where(predictions != scaled_predictions)
            indices = indices[0]
            print(indices)
            print(factor, numpy.sum(numpy.sum(logits > 0, axis=1) > 1), numpy.sum(predictions != scaled_predictions))
            print(logits[indices[:5], :])
            print(scaled_logits[indices[:5], :])

    def testResNetScalingReBN(self):
        batch_size = 100
        cuda = True
        testset = common.datasets.MNISTTestSet(indices=list(range(1000)))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
        model = models.ResNet(10, [1, 28, 28], blocks=[2, 2, 2, 2], channels=32, activation='relu', linear=False, normalization='rebn', bias=False)
        model = self.train(model, 'test_scaling_simplenet_rebn.pth.tar', cuda=cuda)
        print(model)

        for factor in [0.5, 2, 10]:
            logits = common.test.logits(model, testloader, cuda=cuda)
            probabilities = common.test.test(model, testloader, cuda=cuda)
            predictions = numpy.argmax(probabilities, axis=1)

            scaled_model = common.torch.clone(model)
            common.scaling.scale(scaled_model, factor, ignore=['whiten', 'rebn', 'norm', 'downsample.1'])

            scaled_logits = common.test.logits(scaled_model, testloader, cuda=cuda)
            scaled_probabilities = common.test.test(scaled_model, testloader, cuda=cuda)
            scaled_predictions = numpy.argmax(scaled_probabilities, axis=1)

            indices = numpy.where(predictions != scaled_predictions)
            indices = indices[0]
            print(indices)
            print(factor, numpy.sum(numpy.sum(logits > 0, axis=1) > 1), numpy.sum(predictions != scaled_predictions))
            print(logits[indices[:5], :])
            print(scaled_logits[indices[:5], :])



if __name__ == '__main__':
    unittest.main()
