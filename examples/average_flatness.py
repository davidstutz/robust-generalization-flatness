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


# the "weight attack", i.e., random weight perturbations
error_rate = 0.5
weight_attack = attacks.weights.RandomAttack()
weight_attack.epochs = 1
weight_attack.initialization = attacks.weights.initializations.LayerWiseL2UniformSphereInitialization(error_rate)
weight_attack.norm = attacks.weights.norms.L2Norm()
weight_attack.projection = None
weight_attack.get_layers = no_normalization

# compute clean probabilities on test set and evaluate
reference_probabilities = common.test.test(model, testloader, cuda=cuda)
reference_evaluation = common.eval.CleanEvaluation(reference_probabilities, testset.labels)

# run the weight attack, i.e., compute loss with weight perturbations
# we need to manually evaluate each perturbed model
writer = common.summary.SummaryWriter()
weight_objective = attacks.weights.objectives.UntargetedF0Objective()
perturbed_models = common.test.attack_weights(
    model, testloader, weight_attack, weight_objective, attempts=10, writer=writer, cuda=cuda)
evaluations = []
for perturbed_model in perturbed_models:
    perturbed_model.eval()
    if cuda:
        perturbed_model = perturbed_model.cuda()
    probabilities = common.test.test(perturbed_model, testloader, cuda=cuda)
    evaluations.append(common.eval.AdversarialWeightsEvaluation(reference_probabilities, probabilities, testset.labels))
weight_evaluation = common.eval.EvaluationStatistics(evaluations)

reference_loss = reference_evaluation.loss()
perturbed_loss, _ = weight_evaluation('robust_loss', 'mean')
print('reference loss:', reference_loss)
print('perturbed loss:', perturbed_loss)
print('average-case robust flatness:', perturbed_loss - reference_loss)

