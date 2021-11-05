# Relating Adversarially Robust Generalization to Flat Minima

This repository contains code corresponding to the MLSys'21 paper:

D. Stutz, M. Hein, B. Schiele. **Relating Adversarially Robust Generalization to Flat Minima**. ICCV, 2021.

Please cite as:

    @article{Stutz2021ICCV,
        author    = {David Stutz and Matthias Hein and Bernt Schiele},
        title     = {Relating Adversarially Robust Generalization to Flat Minima},
        booktitle = {IEEE International Conference on Computer Vision (ICCV)},
        publisher = {IEEE Computer Society},
        year      = {2021}
    }

Also check the [project page](https://davidstutz.de/projects/robust-generalization-and-flatness/).

This repository allows to reproduce experiments reported in the paper or use the correspondsing quantization,
weight clipping or training procedures as standalone components.

![Relating Adversarially Robust Generalization to Flat Minima.](poster.jpg?raw=true "Robust Flatness.")

## Overview

* [Installation](#installation)
* [Setup](#setup)
* [Standalone Usage](#standalone-usage)
  * [Clean Flatness](#clean-flatness)
  * [Robust Flatness](#robust-flatness)
* [Reproduce Experiments](#reproduce-experiments)
  * [Training](#training)
  * [Evaluation](#evaluation)
  * [Visualization](#visualization)
* [License](#license)

## Installation

The following list includes all Python packages required

* torch (including `torch.utils.tensorboard`)
* torchvision
* tensorflow
* tensorboard
* h5py
* json
* numpy
* zipfile
* umap
* sklearn
* imageio
* scipy
* imgaug

The requirements can be checked using `python3 tests/test_installation.py`. If everything works correctly, all
tests in `tests/` should run without failure.

Code tested with the following versions:

* Debain 9
* Python 3.5.3
* torch 1.3.1+cu92 (with CUDA 9.2)
* torchvision 0.4.2+cu92
* tensorflow 1.14.0
* tensorboard 1.14.0
* h5py 2.9.0
* numpy 1.18.2
* scipy 1.4.1
* sklearn 0.22.1
* imageio 2.5.0
* imgaug 0.2.9
* gcc 6.3.0

Also see `environment.yml` for a (not minimal) export of the used environment.

## Download Datasets

To prepare experiments, datasets need to be downloaded and their paths need to be specified:

Check `common/paths.py` and adapt the following variables appropriately:

    # Absolute path to the data directory:
    # BASE_DATA/mnist will contain MNIST
    # BASE_DATA/Cifar10 (capitlization!) will contain Cifar10
    # BASE_DATA/Cifar100 (capitlization!) will contain Cifar100
    BASE_DATA = '/absolute/path/to/data/directory/'
    # Absolute path to experiments directory, experimental results will be written here (i.e., models, perturbed models ...)
    BASE_EXPERIMENTS = '/absolute/path/to/experiments/directory/'
    # Absolute path to log directory (for TensorBoard logs).
    BASE_LOGS = '/absolute/path/to/log/directory/'
    # Absolute path to code directory (this should point to the root directory of this repository)
    BASE_CODE = '/absolute/path/to/root/of/this/repository/'

Download datasets and copy to the appropriate places. Note that MNIST is only needed for tests and is not used in
the paper's experiments.

**Note that MNIST was not used in the paper, but will be required when running some tests in `tests/`!**

| Dataset | Download |
|---------|----------|
| MNIST | [mnist.zip](https://nextcloud.mpi-klsb.mpg.de/index.php/s/SYGrssFDciF8st8) |
| CIFAR10 | [cifar10.zip](https://nextcloud.mpi-klsb.mpg.de/index.php/s/ik2yFHPCyiA74td) |
| TinyImages 500k| [tinyimages500k.zip](https://nextcloud.mpi-klsb.mpg.de/index.php/s/AYxXdwSXxn3y6AX) |

### Manual Conversion of Datasets

Download MNIST and 500k tiny images from the original sources [1,2]. Then, use the scripts in `data` to convert and check the datasets.
For the code to run properly, the datasets are converted to HDF5 format. Cifar is downloaded automatically.

    [1] http://yann.lecun.com/exdb/mnist/
    [2] https://github.com/yaircarmon/semisup-adv
    
The final dataset directory structure should look as follows:

    BASE_DATE/mnist
    |- t10k-images-idx3-ubyte.gz (downloaded)
    |- t10k-labels-idx-ubyte.gz (downloaded)
    |- train-images-idx3-ubyte.gz (downloaded)
    |- train-labels-idx1-ubyte.gz (downloaded)
    |- train_images.h5 (from data/mnist/convert_mnist.py)
    |- test_images.h5 (from data/mnist/convert_mnist.py)
    |- train_labels.h5 (from data/mnist/convert_mnist.py)
    |- test_labels.h5 (from data/mnist/convert_mnist.py)
    BASE_DATA/Cifar10
    |- cifar-10-batches-py (from torchvision)
    |- cifar-10-python.tar.gz (from torchvision)
    |- train_images.h5 (from data/cifar10/convert_cifar.py)
    |- test_images.h5 (from data/cifar10/convert_cifar.py)
    |- train_labels.h5 (from data/cifar10/convert_cifar.py)
    |- test_labels.h5 (from data/cifar10/convert_cifar.py)
    BASE_DATA/500k_pseudolabeled.pickle
    BASE_DATA/tinyimages500k
    |- train_images.h5
    |- train_labels.h5

## Standalone Components

There are various components that can be used in a standalone fashion.
To highlight a few of them:

* Training procedures for adversarial training variants:
  * Vanilla adversarial training - `common/train/adversarial_training.py`
  * Adversarial training with (adversarial) weight perturbations - `common/train/adversarial_weights_inputs_training.py`
  * Adversarial training with semi-supervision - `common/train/adversarial_semi_supervised_training.py`
  * Adversarial training with Entropy-SGD - `common/train/entropy_adversarial_training.py`
  * TRADES or MART - `common/train/[mart|trades]_adversarial_training.py`
* Adversarial attacks:
  * PGD and variants - `attacks/batch_gradient_descent.py`
  * AutoAttack - `attacks/batch_auto_attack.py`
* Computing Hessian eigenvalues and vectors - `common/hessian.py`

In the following, there are some examples of how to compute average- and worst-case flatness on clean and robust
cross-entropy loss.

**For the below examples, make sure to download CIFAR10 and setup the variables in `common/paths.py` as described above!**

### Clean Flatness

Examples for computing average- and worst-case flatness on the clean loss can be found in `examples/average_flatness.py`
and `examples/worst_flatness.py`. The following code snippet shows an excerpt for computing average-case flatness:

    # imports ...    
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

Note that the example runs only on one batch of 128 test examples. The essential steps are:
* Define a "weight attack", i.e., a method to perturb weights; in this case, random perturbations are used.
* Define a function for selecting the layers to perturb, see `no_normalization`.
* Evaluate the model on the clean examples without weight perturbations.
* Run the weight attack, i.e., compute random weight perturbations and evaluate each perturbed model.
* Compute flatness as the difference of cross-entropy loss of the unperturbed reference model
  and the mean cross-entropy loss over all perturbed model.

Worst-case flatness can be computed simply by replacing the method for computing weight perturbations:

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

Note that in both cases, we use a layer-wise and relative L2 initialization and normalization
(e.g., `attacks.weights.initializations.LayerWiseL2UniformNormInitialization`) which implements the `B_xi(w)`
ball described in the paper.

### Robust Flatness

Examples for computing average- and worst-case flatness on the **robust** loss can be found in
`examples/average_robust_flatness.py` and `examples/worst_robust_flatness.py`. An excerpt for computing average-case
robust flatness is shown below:

    # imports ...
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
    
    
    # the "weight attack", i.e., random weight perturbations
    error_rate = 0.5
    weight_attack = attacks.weights.RandomAttack()
    weight_attack.epochs = 1
    weight_attack.initialization = attacks.weights.initializations.LayerWiseL2UniformSphereInitialization(error_rate)
    weight_attack.norm = attacks.weights.norms.L2Norm()
    weight_attack.projection = None
    weight_attack.get_layers = no_normalization
    
    # objectives: maximize cross-entropy loss
    # SequentialAttack2 runs the weight attack first and then the input attack
    input_objective = attacks.objectives.UntargetedF0Objective()
    weight_objective = attacks.weights.objectives.UntargetedF0Objective()
    attack = attacks.weights_inputs.SequentialAttack2(weight_attack, input_attack)
    
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
    perturbed_loss, _ = weight_evaluation('robust_loss', 'mean')
    print('reference loss:', reference_loss)
    print('perturbed loss:', perturbed_loss)
    print('average-case robust flatness:', perturbed_loss - reference_loss)

The essential steps follow the computation of average-case flatness on the clean cross-entropy loss:
* Define an adversarial example attack, e.g., PGD.
* Define a weight attack, for example random weight perturbations.
* Combine both attacks in a sequential manner (apply weight attack first, then compute adversarial examples) - this
  is done using `SequentialAttack2`.
* Run the adversarial example attack to obtain the reference robust loss.
* Run the sequential weight and input attack to obtain the robust loss with weight perturbations; this is achieved
  by averaging the robust loss per weight perturbation.

For computing worst-case robust flatness, we cannot compute weight perturbations and adversarial examples sequentially.
Instead, adversarial examples and weight perturbations are optimized jointly. This means that the sequential attack
in the above example is replaced by:

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

## Reproduce Experiments

Experiments are defined in `experiments/iccv`. The experiments, i.e., attacks, flatness measures and training modesl,
are defined in `experiments/iccv/common.py`. This is done for three cases on CIFAR10: with AutoAugment using `cifar10.py`,
without AutoAugment in `cifar10_noaa.py` and with unlabeled data (without AutoAugment) in `cifar10_noaa_500k.py`.

The experiments are run using the command line tools provided in `experiments/`, e.g., `experiments/train.py` for
training a model and `experiments/attack.py` for injecting bit errors. Results are evaluated in Jupyter notebooks,
an examples can be found in `experiments/mlsys/eval/evaluation_cifar10.ipynb`.

All experiments are saved in `BASE_EXPERIMENTS`.

### Training

Training a model is easy using the following command line tool:

    python3 train.py iccv.cifar10_noaa resnet18 at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100 --whiten --n=rebn --channels=64

It also allows to use different activation functions using the `-a` option, different architectures or normalization
layers. As detailed above, `iccv.cifar10_noaa` corresponds to CIFAR10 without AutoAugment. The same models can be trained
with AutoAugment using `iccv.cifar10` or with additional unlabeled data using `iccv.cifar10_noo_500k`. 
The model identifier, e.g., `at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100` is defined in `experiments/iccv/common.py`
and examples can be found below.

### Evaluation

To evaluate trained models on clean test or training examples use:

    python3 test.py iccv.cifar10_noaa resnet18 at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100 --whiten --n=rebn --channels=64
    
with `--train` for training examples. Using `--epochs` this can be done for all snapshots, i.e., every 5th epoch.

Adversarial evaluation involves computing robust test error using AutoAttack, robust loss using PGD and average- as well as worst-case flatness:

    python3 attack.py iccv.cifar10_noaa resnet18 at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100 --whiten --n=rebn --channels=64 cifar10_benchmark

This can also be done for every 5th epoch as follows:

    python3 attack.py iccv.cifar10_noaa resnet18 at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100 --whiten --n=rebn --channels=64 cifar10_epochs_benchmark --epochs

(Note that the downloadable experiment data only includes snapshots for vanilla adversarial training in the interest of download size.)

### Visualization

**Pre-computed experiments can be downloaded [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/T2YNzeG825Nrz7a).** Note that this data does *not* correspond to the results from the paper, but were
generated using this repository to illustrate usage. These models also do not include snapshots in the interest of download size.
Log files for plotting training curves are also not included.

The plots from the paper can be produced using `experiments/iccv/eval/evaluation_iccv.ipynb`.
When ran correctly, the notebook should look as in `experiments/iccv/eval/evaluation_iccv.pdf`.
The evaluation does not include all models from the paper by default, but illustrates the usage on some key models.
To run the evaluation and create the below plots, the following models need to be trained and evaluated using `cifar10_benchmark` defined in `experiments/iccv.common.py`:

* `at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100`
* `at_linf_gd_normalized_lr0007_mom0_i14_e00314_f100`
* `at_linf_gd_normalized_lr0007_mom0_i7_e00352_f100`
* `at_ii_linf_gd_normalized_lr0007_mom0_i7_e00314_f100`
* `at_pll_linf_gd_normalized_lr0007_mom0_i7_e00314_f100`
* `0005p_at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100`
* `at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ls01`
* `at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ls02`
* `at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ls03`
* `at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ls04`
* `at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ls05`
* `at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ln01`
* `at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ln02`
* `at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ln03`
* `at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ln04`
* `at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_ln05`
* `at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_cyc`
* `at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_wd0001`
* `at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_wd001`
* `at_linf_gd_normalized_lr0007_mom0_i7_e00314_f100_wd005`
* `at_ssl05_linf_gd_normalized_lr0007_mom0_i7_e00314_f100`
* `at_ssl1_linf_gd_normalized_lr0007_mom0_i7_e00314_f100`
* `at_ssl2_linf_gd_normalized_lr0007_mom0_i7_e00314_f100`
* `at_ssl4_linf_gd_normalized_lr0007_mom0_i7_e00314_f100`
* `at_ssl8_linf_gd_normalized_lr0007_mom0_i7_e00314_f100`
* `trades1_linf_gd_normalized_lr0007_mom0_i7_e00314_f100`
* `trades3_linf_gd_normalized_lr0007_mom0_i7_e00314_f100`
* `trades6_linf_gd_normalized_lr0007_mom0_i7_e00314_f100`
* `trades9_linf_gd_normalized_lr0007_mom0_i7_e00314_f100`

Examples for training and evaluation can be found above.
The corresponding correlation plots from the paper should look as follows with the downloaded experiment data:

![Average-Case Robust Flatness and RLoss.](flatness_rloss.png?raw=true "Average-Case Robust Flatness and RLoss.")

![Average-Case Robust Flatness and Robust Generalization.](flatness_gen.png?raw=true "Average-Case Robust Flatness and Robust Generalization.")

### Visualizing Robust Flatness

For visualizing the robust loss landscape across, the following commands can be used:

    python3 visualize.py iccv.cifar10_noaa resnet18 at_linf_gd_normalized_lr0007_mom0_i14_e00314_f100 --channels=64 --whiten -n=rebn weight_l2_random_nonorm2_e01_at10 -l=input_linf_gd_normalized_lr0007_mom0_i10_e00314_at10 -d=layer_l2_05
    python3 visualize.py iccv.cifar10_noaa resnet18 at_linf_gd_normalized_lr0007_mom0_i14_e00314_f100 --channels=64 --whiten -n=rebn weight_l2_gd_nonorm2_lwrl2normalized_i7_lr001_mom0_e0005_at10_test -l=input_linf_gd_normalized_lr0007_mom0_i10_e00314_at10 -d=layer_l2_001

![Random Direction.](flatness_rnd.png?raw=true "Average-Case Robust Flatness and RLoss.")

![Adversarial Direction.](flatness_adv.png?raw=true "Average-Case Robust Flatness and Robust Generalization.")

### Hessian Eigenvalues

The following command allows to compute Hessian eigenvalues:

    python3 hessian.py iccv.cifar10_noaa resnet18 at_linf_gd_normalized_lr0007_mom0_i14_e00314_f100 --channels=64 --whiten -n=rebn -k=4

## License

This repository includes code from:

* [fra31/auto-attack](https://github.com/fra31/auto-attack)
* [tomgoldstein/loss-landscape](https://github.com/tomgoldstein/loss-landscape)
* [uoguelph-mlrg/Cutout](https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
* [DeepVoltaire/AutoAugment](https://github.com/DeepVoltaire/AutoAugment)

Copyright (c) 2021 David Stutz, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the corresponding papers (see above) in documents and papers that report on research using the Software.