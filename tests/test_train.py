import unittest
import numpy
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import models
import common.test
import common.train
import common.datasets
import common.calibration
import common.torch
import common.state
import common.imgaug
import common.eval
import attacks
import attacks.weights
import attacks.weights_inputs
import torch
import torch.utils.data
from imgaug import augmenters as iaa


class TestTrainMNIST(unittest.TestCase):
    def setUp(self):
        self.batch_size = 100
        self.trainset = torch.utils.data.DataLoader(common.datasets.MNISTTrainSet(indices=range(10000)), batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.testset = torch.utils.data.DataLoader(common.datasets.MNISTTestSet(indices=range(5000)), batch_size=self.batch_size, shuffle=False, num_workers=4)

        self.cuda = True
        self.model = models.LeNet(10, [1, 28, 28], channels=32, normalization='bn', linear=256)

        if self.cuda:
            self.model = self.model.cuda()

    def testNormalTraining(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        trainer = common.train.NormalTraining(self.model, self.trainset, self.testset, optimizer, scheduler, augmentation=augmentation, writer=writer, cuda=self.cuda)

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval.test_error())

    def testNormalTrainingAverage(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        trainer = common.train.NormalTraining(self.model, self.trainset, self.testset, optimizer, scheduler, augmentation=augmentation, writer=writer, cuda=self.cuda)
        trainer.keep_average = True
        trainer.keep_average_tau = 0.98

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        self.model.eval()
        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval1 = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval1.test_error())

        trainer.average.eval()
        probabilities = common.test.test(trainer.average, self.testset, cuda=self.cuda)
        eval2 = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval2.test_error())

        common.calibration.reset(trainer.average)
        common.calibration.calibrate(trainer.average, trainer.trainset, trainer.testset, trainer.augmentation, epochs=1, cuda=self.cuda)

        trainer.average.eval()
        probabilities = common.test.test(trainer.average, self.testset, cuda=self.cuda)
        eval3 = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        print(eval1.test_error())
        print(eval2.test_error())
        print(eval3.test_error())
        self.assertGreaterEqual(0.05, eval3.test_error())

    def testSmoothNormalTraining(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        trainer = common.train.NormalTraining(self.model, self.trainset, self.testset, optimizer, scheduler, augmentation=augmentation,
                                              loss=common.torch.smooth_classification_loss, writer=writer, cuda=self.cuda)

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval.test_error())

    def testNormalTrainingAugmentation(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=(0, 0.025)),
            iaa.Add((-0.075, 0.075)),
            common.imgaug.Clip(0, 1)
        ])

        trainer = common.train.NormalTraining(self.model, self.trainset, self.testset, optimizer, scheduler, augmentation=augmentation, writer=writer, cuda=self.cuda)

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval.test_error())

    def testAdversarialTraining(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        epsilon = 0.1
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 2
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = False
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.norm = attacks.norms.LInfNorm()
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        objective = attacks.objectives.UntargetedF0Objective()

        trainer = common.train.AdversarialTraining(self.model, self.trainset, self.testset, optimizer, scheduler, attack, objective, fraction=0.5, augmentation=augmentation, writer=writer, cuda=self.cuda)
        trainer.summary_gradients = False

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval.test_error())

        adversarial_perturbations, adversarial_probabilities, _ = common.test.attack(self.model, self.testset, attack, objective, attempts=1, writer=writer, cuda=self.cuda)
        eval = common.eval.AdversarialEvaluation(probabilities[:len(self.testset.dataset)], adversarial_probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.2, eval.robust_test_error())

    def testAdversarialTrainingAverage(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        epsilon = 0.1
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 2
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = False
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.norm = attacks.norms.LInfNorm()
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        objective = attacks.objectives.UntargetedF0Objective()

        trainer = common.train.AdversarialTraining(self.model, self.trainset, self.testset, optimizer, scheduler, attack, objective, fraction=0.5, augmentation=augmentation, writer=writer, cuda=self.cuda)
        trainer.summary_gradients = False
        trainer.keep_average = True
        trainer.keep_average_tau = 0.98

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval1 = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval1.test_error())

        adversarial_perturbations, adversarial_probabilities, _ = common.test.attack(self.model, self.testset, attack, objective, attempts=1, writer=writer, cuda=self.cuda)
        eval2 = common.eval.AdversarialEvaluation(probabilities[:len(self.testset.dataset)], adversarial_probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.2, eval2.robust_test_error())

        trainer.average.eval()
        probabilities = common.test.test(trainer.average, self.testset, cuda=self.cuda)
        eval3 = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval3.test_error())

        adversarial_perturbations, adversarial_probabilities, _ = common.test.attack(trainer.average, self.testset, attack, objective, attempts=1, writer=writer, cuda=self.cuda)
        eval4 = common.eval.AdversarialEvaluation(probabilities[:len(self.testset.dataset)], adversarial_probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.2, eval4.robust_test_error())

        common.calibration.reset(trainer.average)
        common.calibration.calibrate(trainer.average, trainer.trainset, trainer.testset, trainer.augmentation, epochs=1, cuda=self.cuda)

        trainer.average.eval()
        probabilities = common.test.test(trainer.average, self.testset, cuda=self.cuda)
        eval5 = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval5.test_error())

        adversarial_perturbations, adversarial_probabilities, _ = common.test.attack(trainer.average, self.testset, attack, objective, attempts=1, writer=writer, cuda=self.cuda)
        eval6 = common.eval.AdversarialEvaluation(probabilities[:len(self.testset.dataset)], adversarial_probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.2, eval6.robust_test_error())

        print(eval1.test_error())
        print(eval2.robust_test_error())
        print(eval3.test_error())
        print(eval4.robust_test_error())
        print(eval5.test_error())
        print(eval6.robust_test_error())

    def testAdversarialTrainingLabelLeaking(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        epsilon = 0.1
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 3
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.norm = attacks.norms.LInfNorm()
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        objective = attacks.objectives.UntargetedF0Objective()

        trainer = common.train.AdversarialTraining(self.model, self.trainset, self.testset, optimizer, scheduler, attack, objective, fraction=0.5, augmentation=augmentation, writer=writer, cuda=self.cuda)
        trainer.summary_gradients = False
        trainer.prevent_label_leaking = True

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval.test_error())

        adversarial_perturbations, adversarial_probabilities, _ = common.test.attack(self.model, self.testset, attack, objective, attempts=1, writer=writer, cuda=self.cuda)
        eval = common.eval.AdversarialEvaluation(probabilities[:len(self.testset.dataset)], adversarial_probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.2, eval.robust_test_error())

    def testAdversarialTrainingFraction(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        epsilon = 0.1
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 3
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.norm = attacks.norms.LInfNorm()
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        objective = attacks.objectives.UntargetedF0Objective()

        trainer = common.train.AdversarialTraining(self.model, self.trainset, self.testset, optimizer, scheduler, attack, objective, fraction=1, augmentation=augmentation, writer=writer, cuda=self.cuda)
        trainer.summary_gradients = False

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval.test_error())

        adversarial_perturbations, adversarial_probabilities, _ = common.test.attack(self.model, self.testset, attack, objective, attempts=1, writer=writer, cuda=self.cuda)
        eval = common.eval.AdversarialEvaluation(probabilities[:len(self.testset.dataset)], adversarial_probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.15, eval.robust_test_error())

    def testAdversarialWeightsInputsTraining(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        weight_objective = attacks.weights.UntargetedF0Objective()
        weight_epsilon = 10
        clipping = 1
        weight_attack = attacks.weights.GradientDescentAttack()
        weight_attack.base_lr = 0.01
        weight_attack.momentum = 0.9
        weight_attack.epochs = 1
        weight_attack.normalization = attacks.weights.normalizations.L2Normalization()
        weight_attack.backtrack = False
        weight_attack.initialization = attacks.weights.initializations.L2UniformNormInitialization(weight_epsilon)
        weight_attack.projection = attacks.weights.projections.SequentialProjections([
            attacks.weights.projections.BoxProjection(-clipping, clipping),
            attacks.weights.projections.L2Projection(weight_epsilon),
        ])
        weight_attack.norm = attacks.weights.norms.L2Norm()

        input_objective = attacks.objectives.UntargetedF0Objective()
        input_epsilon = 0.1
        input_attack = attacks.BatchGradientDescent()
        input_attack.max_iterations = 1
        input_attack.base_lr = 0.1
        input_attack.momentum = 0
        input_attack.lr_factor = 1
        input_attack.c = 0
        input_attack.normalized = True
        input_attack.backtrack = False
        input_attack.initialization = attacks.initializations.LInfUniformInitialization(input_epsilon)
        input_attack.projection = attacks.projections.SequentialProjections([
            attacks.projections.BoxProjection(0, 1),
            attacks.projections.LInfProjection(input_epsilon),
        ])
        input_attack.norm = attacks.norms.LInfNorm()

        trainer = common.train.AdversarialWeightsInputsTraining(self.model, self.trainset, self.testset, optimizer, scheduler,
                                                          weight_attack, weight_objective, input_attack, input_objective,
                                                          augmentation=augmentation, writer=writer, cuda=self.cuda)
        trainer.projection = attacks.weights.projections.BoxProjection(-clipping, clipping)

        def simple_curriculum(attack, loss, perturbed_loss, epoch):
            if perturbed_loss < 0.5:
                attack.epochs = 20
            elif perturbed_loss < 1:
                attack.epochs = 15
            elif perturbed_loss < 1.5:
                attack.epochs = 10
            elif perturbed_loss < 1.75:
                attack.epochs = 7
            elif perturbed_loss < 2:
                attack.epochs = 5
            elif perturbed_loss < 2.15:
                attack.epochs = 3
            else:
                attack.epochs = 1
            population = 1

            return population, {
                'population': population,
                'epochs': attack.epochs,
            }
        trainer.curriculum = simple_curriculum

        epochs = 10
        for e in range(epochs):
            trainer.step(e)
            common.state.State.checkpoint('test_train.pth.tar', self.model)

        self.model.eval()
        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        #self.assertGreaterEqual(0.05, eval.test_error())
        print(eval.test_error())

        weight_probabilities = None
        for b, (inputs, targets) in enumerate(self.testset):
            batchset = [(inputs, targets)]
            perturbed_model = weight_attack.run(self.model, batchset, weight_objective)
            if self.cuda:
                perturbed_model = perturbed_model.cuda()
            weight_probabilities = common.numpy.concatenate(weight_probabilities, common.test.test(perturbed_model, batchset, cuda=self.cuda))

        eval = common.eval.CleanEvaluation(weight_probabilities, self.testset.dataset.labels, validation=0)
        print(eval.test_error())

        self.model.eval()
        _, input_probabilities, _ = common.test.attack(self.model, self.testset, input_attack, input_objective, attempts=1, cuda=True)
        eval = common.eval.CleanEvaluation(input_probabilities[0], self.testset.dataset.labels, validation=0)
        print(eval.test_error())

    def testSemiSupervisedTraining(self):
        auxiliary_model = torch.nn.Linear(256, 4)
        self.model = models.LeNet(10, [1, 28, 28], channels=8, normalization='bn', linear=256, auxiliary=auxiliary_model)
        if self.cuda:
            self.model = self.model.cuda()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        supervisedset = common.datasets.MNISTTrainSet(indices=range(1000))
        unsupervisedset = common.datasets.MNISTTrainSet(indices=1000 + numpy.array(list(range(59000))))
        trainset = common.datasets.RotatedSemiSupervisedDataset(supervisedset, unsupervisedset)
        trainsampler = common.datasets.SemiSupervisedSampler(trainset.sup_indices, trainset.unsup_indices, self.batch_size, 0.5, num_batches=int(numpy.ceil(len(trainset.sup_indices) / self.batch_size)))
        trainset = torch.utils.data.DataLoader(trainset, batch_sampler=trainsampler)
        trainer = common.train.SemiSupervisedTraining(self.model, trainset, self.testset, optimizer, scheduler, augmentation=augmentation, writer=writer, cuda=self.cuda)

        epochs = 50
        for e in range(epochs):
            trainer.train(e)

        self.model.eval()
        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        # self.assertGreaterEqual(0.05, eval.test_error())
        print(eval.test_error())

    def testAdversarialSemiSupervisedTraining(self):
        auxiliary_model = torch.nn.Linear(256, 4)
        self.model = models.LeNet(10, [1, 28, 28], channels=8, normalization='bn', linear=256, auxiliary=auxiliary_model)
        if self.cuda:
            self.model = self.model.cuda()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        supervisedset = common.datasets.MNISTTrainSet(indices=range(1000))
        unsupervisedset = common.datasets.MNISTTrainSet(indices=1000 + numpy.array(list(range(59000))))
        trainset = common.datasets.RotatedSemiSupervisedDataset(supervisedset, unsupervisedset)
        trainsampler = common.datasets.SemiSupervisedSampler(trainset.sup_indices, trainset.unsup_indices, self.batch_size, 0.5, num_batches=int(numpy.ceil(len(trainset.sup_indices) / self.batch_size)))
        trainset = torch.utils.data.DataLoader(trainset, batch_sampler=trainsampler)

        epsilon = 0.3
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = 2
        attack.base_lr = 0.1
        attack.momentum = 0
        attack.c = 0
        attack.lr_factor = 1.5
        attack.normalized = True
        attack.backtrack = True
        attack.initialization = attacks.initializations.LInfUniformInitialization(epsilon)
        attack.norm = attacks.norms.LInfNorm()
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        objective = attacks.objectives.UntargetedF0Objective()

        trainer = common.train.AdversarialSemiSupervisedTraining(self.model, trainset, self.testset, optimizer, scheduler,
                                                            attack, objective, fraction=0.5, augmentation=augmentation,
                                                            writer=writer, cuda=self.cuda)
        trainer.summary_gradients = False

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        #self.assertGreaterEqual(0.05, eval.test_error())
        print('test error', eval.test_error())

        adversarial_perturbations, adversarial_probabilities, _ = common.test.attack(self.model, self.testset, attack, objective, attempts=1, writer=writer, cuda=self.cuda)
        eval = common.eval.AdversarialEvaluation(probabilities[:len(self.testset.dataset)], adversarial_probabilities, self.testset.dataset.labels, validation=0)
        print('robust test error', eval.robust_test_error())


if __name__ == '__main__':
    unittest.main()
