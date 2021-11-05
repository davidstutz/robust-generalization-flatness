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
attack = attacks.weights.GradientDescentAttack()
attack.epochs = 20
attack.base_lr = 0.01
attack.normalization = attacks.weights.normalizations.LayerWiseRelativeL2Normalization()
attack.backtrack = False
attack.momentum = 0
attack.lr_factor = 1
attack.initialization = attacks.weights.initializations.LayerWiseL2UniformNormInitialization(error_rate)
attack.norm = attacks.weights.norms.L2Norm()
attack.projection = attacks.weights.SequentialProjections([
    attacks.weights.projections.LayerWiseL2Projection(error_rate)
])
attack.get_layers = no_normalization

# objectives: maximize cross-entropy loss
input_objective = attacks.objectives.UntargetedF0Objective()
weight_objective = attacks.weights.objectives.UntargetedF0Objective()

# compute clean probabilities on test set
reference_probabilities = common.test.test(model, testloader, cuda=cuda)
reference_evaluation = common.eval.CleanEvaluation(reference_probabilities, testset.labels)

# run the weight attack, i.e., compute loss with weight perturbations
writer = common.summary.SummaryWriter()
objective = attacks.weights.objectives.UntargetedF0Objective()
perturbed_models = common.test.attack_weights(
    model, testloader, attack, objective, attempts=10, writer=writer, cuda=cuda)
evaluations = []
for perturbed_model in perturbed_models:
    perturbed_model.eval()
    if cuda:
        perturbed_model = perturbed_model.cuda()
    probabilities = common.test.test(perturbed_model, testloader, cuda=cuda)
    evaluations.append(common.eval.AdversarialWeightsEvaluation(reference_probabilities, probabilities, testset.labels))
weight_evaluation = common.eval.EvaluationStatistics(evaluations)

reference_loss = reference_evaluation.loss()
perturbed_loss, _ = weight_evaluation('robust_loss', 'max')
print('reference loss:', reference_loss)
print('perturbed loss:', perturbed_loss)
print('average-case robust flatness:', perturbed_loss - reference_loss)
