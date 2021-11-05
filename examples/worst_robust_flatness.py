import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import attacks
import attacks.weights
import attacks.weights_inputs
import common.test
import common.state
import common.eval
import common.datasets
import numpy
import torch.utils.data


cuda = True
model_file = 'AT.pth.tar'
if not os.path.exists(model_file):
    print('Download the adversarially trained model as outlined in the README and store it in the directory ./examples as %s!' % model_file)
    exit()

# load model
state = common.state.State.load(model_file)
model = state.model
model.eval()
if cuda:
    model = model.cuda()

# setup set to compute flatness on
# use only one batch for simplicity, in the paper we use 1000 examples instead
testset = common.datasets.Cifar10TestSet(indices=numpy.array(list(range(128))))
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# the PGD attack to compute adversarial examples with
epsilon = 0.0314
input_attack = attacks.BatchGradientDescent()
input_attack.max_iterations = 20
input_attack.base_lr = 0.007
input_attack.momentum = 0
input_attack.c = 0
input_attack.lr_factor = 1
input_attack.backtrack = False
input_attack.normalized = True
input_attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
input_attack.projection = attacks.projections.SequentialProjections([
    attacks.projections.BoxProjection(0, 1),
    attacks.projections.LInfProjection(epsilon),
])
input_attack.norm = attacks.norms.LInfNorm()


# make sure that no normalization layers are attacked:
def no_normalization(model):
    exclude_layers = ['whiten', 'rebn', 'norm', 'downsample.1', 'bn', 'shortcut.1']
    n_parameters = len(list(model.parameters()))
    parameters = dict(model.named_parameters())
    names = list(parameters.keys())
    assert len(names) == n_parameters
    layers = []
    for i in range(len(names)):
        if parameters[names[i]].requires_grad:
            exclude = False
            for e in exclude_layers:
                if names[i].find(e) >= 0:
                    exclude = True
            if not exclude:
                layers.append(i)
    return layers


# the weight attack to jointly compute weight and input perturbations
error_rate = 0.003
attack = attacks.weights_inputs.GradientDescentAttack()
attack.epochs = 20
attack.weight_initialization = attacks.weights.initializations.LayerWiseL2UniformNormInitialization(error_rate)
attack.weight_projection = attacks.weights.projections.SequentialProjections([
    attacks.weights.projections.LayerWiseL2Projection(error_rate),
])
attack.weight_norm = attacks.weights.norms.L2Norm()
attack.weight_normalization = attacks.weights.normalizations.LayerWiseRelativeL2Normalization()
attack.input_initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
attack.input_projection = attacks.projections.SequentialProjections([
    attacks.projections.BoxProjection(0, 1),
    attacks.projections.LInfProjection(epsilon),
])
attack.input_norm = attacks.norms.LInfNorm()
# this matches the PGD attack above
attack.input_normalized = True
attack.input_lr = 0.007
attack.weight_lr = 0.01
attack.get_layers = no_normalization

# objectives: maximize cross-entropy loss
input_objective = attacks.objectives.UntargetedF0Objective()
weight_objective = attacks.weights.objectives.UntargetedF0Objective()

# compute clean probabilities on test set
clean_probabilities = common.test.test(model, testloader, cuda=cuda)

# run just adversarial examples, i.e., robust loss without weight perturbation
writer = common.summary.SummaryWriter()
_, reference_probabilities, _ = common.test.attack(
    model, testloader, input_attack, input_objective, attempts=1, writer=writer, cuda=cuda)

# run the weight attack, i.e., compute robust loss with weight perturbations
# in the paper, we use 10 attempts instead
_, _, probabilities, _ = common.test.attack_weights_inputs(
    model, testloader, attack, weight_objective, input_objective, attempts=5, writer=writer, cuda=cuda)

# for weight perturbations, we take the mean across random weight perturbation
# the reference loss is just the robust loss on the separately computed adversarial examples
evaluations = []
for i in range(probabilities.shape[0]):
    evaluations.append(common.eval.AdversarialWeightsEvaluation(clean_probabilities, probabilities[i], testset.labels))
input_evaluation = common.eval.AdversarialEvaluation(clean_probabilities, reference_probabilities, testset.labels, validation=0)
weight_evaluation = common.eval.EvaluationStatistics(evaluations)

reference_loss = input_evaluation.robust_loss()
perturbed_loss, _ = weight_evaluation('robust_loss', 'max')
print('reference loss:', reference_loss)
print('perturbed loss:', perturbed_loss)
print('average-case robust flatness:', perturbed_loss - reference_loss)

