import os
import common.experiments
import common.utils
import common.eval
import common.paths
import common.imgaug
import common.datasets
import common.plot
import common.summary
from common.log import log, LogLevel
import numpy


def gformat(value):
    return ('%f' % float(value)).rstrip('0').replace('.', '')


class CheapAttackConfig:
    """
    Simple surrogate for attack config just storing attempts and directory.
    """
    def __init__(self, config):
        self.directory = config.directory
        self.attempts = config.attempts


class CheapTrainingConfig:
    """
    Simple surrogate for training config just storing attempts and epochs.
    """
    def __init__(self, config):
        self.directory = config.directory
        self.epochs = config.epochs


def get_log_directory(config, training_config):
    """
    Get log directory for given training configuration.
    """
    return common.paths.log_dir(training_config.directory)


def get_model_file(config, training_config):
    """
    Get model file for a given training configuration.
    """
    model_file = common.paths.experiment_file(training_config.directory, 'classifier', common.paths.STATE_EXT)
    model_epochs = training_config.epochs
    if not os.path.exists(model_file):
        model_file, model_epochs = common.experiments.find_incomplete_file(model_file)

    return model_file, model_epochs


def get_model_files(config, training_config):
    """
    Get model files for all epochs for given training configuration.
    """
    model_file = common.paths.experiment_file(training_config.directory, 'classifier', common.paths.STATE_EXT)
    model_files, model_epochs = common.experiments.find_incomplete_files(model_file)
    assert model_files is not None and model_epochs is not None
    model_epochs.append(None)
    model_files.append(model_file)

    return model_files, model_epochs


def get_probabilities_file(config, training_config, train=False, epoch=None):
    """
    Get test or training probabilities file for given training configuration.
    """
    if train:
        probabilities_file = common.paths.experiment_file(training_config.directory, 'train_probabilities', common.paths.HDF5_EXT)
    else:
        probabilities_file = common.paths.experiment_file(training_config.directory, 'probabilities', common.paths.HDF5_EXT)

    if epoch is None:
        probabilities_epochs = training_config.epochs
        if not os.path.exists(probabilities_file):
            probabilities_file, probabilities_epochs = common.experiments.find_incomplete_file(probabilities_file)
    else:
        probabilities_epochs = epoch
        probabilities_file += '.%d' % epoch
        if not os.path.exists(probabilities_file):
            probabilities_file = None

    return probabilities_file, probabilities_epochs


def get_perturbed_model_file(config, training_config, attack_config, attempt):
    """
    Get model with perturbation given training and attack configuration.
    """
    perturbed_model_file = common.paths.experiment_file('%s/%s' % (training_config.directory, attack_config.directory), 'perturbation%d' % attempt, common.paths.STATE_EXT)
    perturbed_model_epochs = training_config.epochs

    if not os.path.exists(perturbed_model_file):
        perturbed_model_file, perturbed_model_epochs = common.experiments.find_incomplete_file(perturbed_model_file)

    if perturbed_model_file is not None:
        if common.utils.creation_date(perturbed_model_file) < 1600880416.4045188:
            log('[Warning] too old: %s' % perturbed_model_file)

    return perturbed_model_file, perturbed_model_epochs


def get_perturbed_probabilities_file(config, training_config, attack_config, attempt, basename='probabilities', epoch=None):
    """
    Get probabilities for weight perturbation for given training and attack configuration.
    """
    adversarial_probabilities_file = common.paths.experiment_file('%s/%s' % (training_config.directory, attack_config.directory), '%s%d' % (basename, attempt), common.paths.HDF5_EXT)

    if epoch is None:
        adversarial_probabilities_epochs = training_config.epochs
        if not os.path.exists(adversarial_probabilities_file):
            adversarial_probabilities_file, adversarial_probabilities_epochs = common.experiments.find_incomplete_file(adversarial_probabilities_file)
    else:
        adversarial_probabilities_epochs = epoch
        adversarial_probabilities_file += '.%d' % epoch
        if not os.path.exists(adversarial_probabilities_file):
            adversarial_probabilities_file = None

    return adversarial_probabilities_file, adversarial_probabilities_epochs


def get_perturbations_file(config, training_config, attack_config, epoch=None):
    """
    Get adversarial examples or probabilities on adversarial examples for training and attack configuration.
    """
    adversarial_probabilities_file = common.paths.experiment_file('%s/%s' % (training_config.directory, attack_config.directory), 'perturbations', common.paths.HDF5_EXT)

    if epoch is None:
        adversarial_probabilities_epochs = training_config.epochs
        if not os.path.exists(adversarial_probabilities_file):
            adversarial_probabilities_file, adversarial_probabilities_epochs = common.experiments.find_incomplete_file(adversarial_probabilities_file)
    else:
        adversarial_probabilities_file = common.paths.experiment_file('%s/%s' % (training_config.directory, attack_config.directory), 'perturbations%d' % epoch, common.paths.HDF5_EXT)
        adversarial_probabilities_epochs = epoch

        if not os.path.exists(adversarial_probabilities_file):
            adversarial_probabilities_file = None

    return adversarial_probabilities_file, adversarial_probabilities_epochs


def load(config, training_config_vars, training_suffixes, attack_config_vars):
    """
    Load training configurations.
    """
    attack_configs = []
    for attack in attack_config_vars:
        if isinstance(attack, list):
            attack_configs_ = []
            for attack_ in attack:
                attack_configs_.append(CheapAttackConfig(getattr(config, attack_)))
            attack_configs.append(attack_configs_)
        else:
            attack_configs.append(CheapAttackConfig(getattr(config, attack)))
    training_configs = []
    for t in range(len(training_config_vars)):
        training_config = CheapTrainingConfig(getattr(config, training_config_vars[t]))
        if isinstance(training_suffixes, list):
            training_config.directory += training_suffixes[t]
        else:
            training_config.directory += training_suffixes
        training_configs.append(training_config)

    return training_configs, attack_configs


def load_input(config, attack_config_vars):
    """
    Load adversarial example attack configurations.
    """
    attack_configs = []
    for attack in attack_config_vars:
        if isinstance(attack, list):
            attack_configs_ = []
            for attack_ in attack:
                attack_configs_.append(CheapAttackConfig(getattr(config, attack_)))
            attack_configs.append(attack_configs_)
        else:
            attack_configs.append(CheapAttackConfig(getattr(config, attack)))

    return attack_configs


def load_weight_input(config, attack_config_vars):
    """
    Load weight and input attack configurations.
    """
    attack_configs = []
    for attack in attack_config_vars:
        if isinstance(attack, list):
            attack_configs_ = []
            for attack_ in attack:
                attack_configs_.append(CheapAttackConfig(getattr(config, attack_)))
            attack_configs.append(attack_configs_)
        else:
            attack_configs.append(CheapAttackConfig(getattr(config, attack)))

    return attack_configs


def get_attack_evaluations(config, training_configs, attack_configs, evaluation_class=common.eval.AdversarialWeightsEvaluation, limit=9000, train=False, epoch=None):
    """
    Get weight attack evaluations.
    """
    evaluations = []
    epochs = []
    epoch_table = []

    testset = config.testloader.dataset
    if train:
        testset = config.testtrainloader.dataset

    for training_config in training_configs:
        attack_evaluations = []
        attack_epochs = []
        for attack_config in attack_configs:
            evaluation = None
            evaluation_epoch = None
            if isinstance(attack_config, list):
                if training_config is not None:
                    probabilities_file, probabilities_epochs = get_probabilities_file(config, training_config, train=train, epoch=epoch)

                    if probabilities_file is not None:
                        clean_probabilities = common.utils.read_hdf5(probabilities_file, 'probabilities')
                        epoch_table.append(['**' + training_config.directory + '**', probabilities_epochs, ''])

                        attempt_evaluations = []
                        min_adversarial_probabilities_epoch = 1e12
                        for attack_config_ in attack_config:
                            if attack_config_ is not None:
                                # find adversarial probabilities
                                a_count = 0

                                for a in range(attack_config_.attempts):
                                    adversarial_probabilities_file, adversarial_probabilities_epochs = get_perturbed_probabilities_file(config, training_config, attack_config_, a, epoch=epoch)

                                    if adversarial_probabilities_file is not None:
                                        adversarial_probabilities = common.utils.read_hdf5(adversarial_probabilities_file, 'probabilities')
                                        if adversarial_probabilities.shape[0] < limit:
                                            pass
                                        else:
                                            adversarial_probabilities = adversarial_probabilities[:limit]
                                        adversarial_evaluation = evaluation_class(clean_probabilities, adversarial_probabilities, testset.labels)
                                        attempt_evaluations.append(adversarial_evaluation)

                                        min_adversarial_probabilities_epoch = min(min_adversarial_probabilities_epoch, adversarial_probabilities_epochs)
                                        a_count += 1
                                epoch_table.append([attack_config_.directory, str(min_adversarial_probabilities_epoch), str(a_count)])

                        if len(attempt_evaluations) > 0:
                            evaluation = common.eval.EvaluationStatistics(attempt_evaluations)
                            evaluation_epoch = min_adversarial_probabilities_epoch
            else:
                if attack_config is not None and training_config is not None:
                    probabilities_file, probabilities_epochs = get_probabilities_file(config, training_config, train=train, epoch=epoch)
                    if probabilities_file is not None:
                        clean_probabilities = common.utils.read_hdf5(probabilities_file, 'probabilities')
                        epoch_table.append(['**' + training_config.directory + '**', probabilities_epochs, ''])

                        a_count = 0
                        min_adversarial_probabilities_epoch = 1e12

                        attempt_evaluations = []
                        for a in range(attack_config.attempts):
                            adversarial_probabilities_file, adversarial_probabilities_epochs = get_perturbed_probabilities_file(config, training_config, attack_config, a, epoch=epoch)

                            if adversarial_probabilities_file is not None:
                                adversarial_probabilities = common.utils.read_hdf5(adversarial_probabilities_file, 'probabilities')
                                if adversarial_probabilities.shape[0] < limit:
                                    pass
                                else:
                                    adversarial_probabilities = adversarial_probabilities[:limit]
                                adversarial_evaluation = evaluation_class(clean_probabilities, adversarial_probabilities, testset.labels)
                                attempt_evaluations.append(adversarial_evaluation)

                                min_adversarial_probabilities_epoch = min(min_adversarial_probabilities_epoch, adversarial_probabilities_epochs)
                                a_count += 1

                        if len(attempt_evaluations) > 0:
                            evaluation = common.eval.EvaluationStatistics(attempt_evaluations)
                            evaluation_epoch = adversarial_probabilities_epochs

                        epoch_table.append([attack_config.directory, str(min_adversarial_probabilities_epoch), str(a_count)])

            attack_evaluations.append(evaluation)
            attack_epochs.append(evaluation_epoch)
        evaluations.append(attack_evaluations)
        epochs.append(attack_epochs)

    return evaluations, epochs


def get_input_attack_evaluations(config, training_configs, attack_configs,
                                 evaluation_class=common.eval.AdversarialEvaluation, limit=1000, validation=0, train=False, epoch=None):
    """
    Get input attack / adversarial example evaluations.
    """
    evaluations = []
    epochs = []

    testset = config.testloader.dataset
    if train:
        testset = config.testtrainloader.dataset

    for training_config in training_configs:
        attack_evaluations = []
        attack_epochs = []
        for attack_config in attack_configs:
            if isinstance(attack_config, list):
                evaluation = None
                evaluation_epoch = None

                if training_config is not None:
                    adversarial_probabilities = None
                    adversarial_errors = None

                    for attack_config_ in attack_config:
                        if attack_config_ is not None:
                            adversarial_probabilities_file, adversarial_probabilities_epochs = get_perturbations_file(config, training_config, attack_config_, epoch=epoch)

                            if adversarial_probabilities_file is not None:
                                if evaluation_epoch is None:
                                    evaluation_epoch = adversarial_probabilities_epochs
                                else:
                                    evaluation_epoch = min(evaluation_epoch, adversarial_probabilities_epochs)

                                adversarial_probabilities_ = common.utils.read_hdf5(adversarial_probabilities_file, 'probabilities')
                                if adversarial_probabilities_.shape[1] > limit:
                                    adversarial_probabilities_ = adversarial_probabilities_[:, :limit]
                                # adversarial_errors_ = common.utils.read_hdf5(adversarial_probabilities_file, 'errors')
                                adversarial_errors_ = numpy.copy(adversarial_probabilities_)

                                adversarial_errors_[
                                :,
                                numpy.arange(adversarial_errors_.shape[1]),
                                testset.labels[:adversarial_errors_.shape[1]],
                                ] = 0
                                assert len(adversarial_errors_.shape) == 3
                                adversarial_errors_ = -numpy.max(adversarial_errors_, axis=2)

                                if adversarial_probabilities is not None:
                                    assert adversarial_probabilities.shape[1] == adversarial_probabilities_.shape[1], (adversarial_probabilities.shape[1] == adversarial_probabilities_.shape[1])
                                adversarial_probabilities = common.numpy.concatenate(adversarial_probabilities, adversarial_probabilities_, axis=0)
                                adversarial_errors = common.numpy.concatenate(adversarial_errors, adversarial_errors_, axis=0)

                        probabilities_file, probabilities_epochs = get_probabilities_file(config, training_config, train=train, epoch=epoch)
                        if probabilities_file is not None and adversarial_probabilities is not None:
                            clean_probabilities = common.utils.read_hdf5(probabilities_file, 'probabilities')
                            if clean_probabilities.shape[0] != testset.labels.shape[0]:
                                print(training_config.directory)
                            assert clean_probabilities.shape[0] == testset.labels.shape[0]
                            evaluation = evaluation_class(clean_probabilities, adversarial_probabilities,
                                                          testset.labels, errors=adversarial_errors,
                                                          validation=validation)

                attack_evaluations.append(evaluation)
                attack_epochs.append(evaluation_epoch)

            else:
                evaluation = None
                evaluation_epoch = None

                if attack_config is not None and training_config is not None:
                    # find clean probabilities
                    probabilities_file, probabilities_epochs = get_probabilities_file(config, training_config, train=train, epoch=epoch)
                    if probabilities_file is not None:
                        clean_probabilities = common.utils.read_hdf5(probabilities_file, 'probabilities')

                        adversarial_probabilities_file, adversarial_probabilities_epochs = get_perturbations_file(config, training_config, attack_config, epoch=epoch)

                        if adversarial_probabilities_file is not None:
                            adversarial_probabilities = common.utils.read_hdf5(adversarial_probabilities_file, 'probabilities')
                            adversarial_errors = common.utils.read_hdf5(adversarial_probabilities_file, 'errors')
                            if adversarial_probabilities.shape[1] > limit:
                                adversarial_probabilities = adversarial_probabilities[:, :limit]
                                adversarial_errors = adversarial_errors[:, :limit]
                            if clean_probabilities.shape[0] != testset.labels.shape[0]:
                                print(training_config.directory)
                            assert clean_probabilities.shape[0] == testset.labels.shape[0]
                            evaluation = evaluation_class(clean_probabilities, adversarial_probabilities,
                                                          testset.labels, errors=adversarial_errors,
                                                          validation=validation)

                attack_evaluations.append(evaluation)
                attack_epochs.append(evaluation_epoch)
        evaluations.append(attack_evaluations)
        epochs.append(attack_epochs)

    return evaluations, epochs


def get_weight_input_attack_evaluations(config, training_configs, attack_configs, evaluation_class=common.eval.AdversarialWeightsEvaluation, limit=9000, train=False, epoch=None):
    """
    Get weight and input attack evaluations.
    """
    evaluations = []
    epochs = []
    epoch_table = []

    testset = config.testloader.dataset
    if train:
        testset = config.testtrainloader.dataset

    for training_config in training_configs:
        attack_evaluations = []
        attack_epochs = []
        for attack_config in attack_configs:
            evaluation = None
            evaluation_epoch = None
            if isinstance(attack_config, list):
                if training_config is not None:
                    probabilities_file, probabilities_epochs = get_probabilities_file(config, training_config, train=train, epoch=epoch)

                    if probabilities_file is not None:
                        clean_probabilities = common.utils.read_hdf5(probabilities_file, 'probabilities')
                        epoch_table.append(['**' + training_config.directory + '**', probabilities_epochs, ''])

                        attempt_evaluations = []
                        min_adversarial_probabilities_epoch = 1e12
                        for attack_config_ in attack_config:
                            if attack_config_ is not None:
                                # find adversarial probabilities
                                a_count = 0

                                for a in range(attack_config_.attempts):
                                    adversarial_probabilities_file, adversarial_probabilities_epochs = get_perturbed_probabilities_file(config, training_config, attack_config_, a, basename='perturbations', epoch=epoch)

                                    if adversarial_probabilities_file is not None:
                                        adversarial_probabilities = common.utils.read_hdf5(adversarial_probabilities_file, 'probabilities')

                                        if adversarial_probabilities.shape[0] < limit:
                                            pass
                                        else:
                                            adversarial_probabilities = adversarial_probabilities[:limit]
                                            
                                        if clean_probabilities.shape[0] != testset.labels.shape[0]:
                                            print('probabilities', training_config.directory)
                                        assert clean_probabilities.shape[0] == testset.labels.shape[0], (clean_probabilities.shape[0], testset.labels.shape[0])
                                        adversarial_evaluation = evaluation_class(clean_probabilities, adversarial_probabilities, testset.labels)
                                        attempt_evaluations.append(adversarial_evaluation)

                                        min_adversarial_probabilities_epoch = min(min_adversarial_probabilities_epoch, adversarial_probabilities_epochs)
                                        a_count += 1
                                epoch_table.append([attack_config_.directory, str(min_adversarial_probabilities_epoch), str(a_count)])

                        if len(attempt_evaluations) > 0:
                            evaluation = common.eval.EvaluationStatistics(attempt_evaluations)
                            evaluation_epoch = min_adversarial_probabilities_epoch
            else:
                if attack_config is not None and training_config is not None:
                    probabilities_file, probabilities_epochs = get_probabilities_file(config, training_config, train=train, epoch=epoch)
                    if probabilities_file is not None:
                        clean_probabilities = common.utils.read_hdf5(probabilities_file, 'probabilities')
                        epoch_table.append(['**' + training_config.directory + '**', probabilities_epochs, ''])

                        a_count = 0
                        min_adversarial_probabilities_epoch = 1e12

                        attempt_evaluations = []
                        for a in range(attack_config.attempts):
                            adversarial_probabilities_file, adversarial_probabilities_epochs = get_perturbed_probabilities_file(config, training_config, attack_config, a, epoch=epoch)

                            if adversarial_probabilities_file is not None:
                                adversarial_probabilities = common.utils.read_hdf5(adversarial_probabilities_file, 'probabilities')
                                if adversarial_probabilities.shape[0] < limit:
                                    pass
                                else:
                                    adversarial_probabilities = adversarial_probabilities[:limit]

                                if clean_probabilities.shape[0] != testset.labels.shape[0]:
                                    print('probabilities', training_config.directory)
                                assert clean_probabilities.shape[0] == testset.labels.shape[0], (clean_probabilities.shape[0], testset.labels.shape[0])
                                adversarial_evaluation = evaluation_class(clean_probabilities, adversarial_probabilities, testset.labels)
                                attempt_evaluations.append(adversarial_evaluation)

                                min_adversarial_probabilities_epoch = min(min_adversarial_probabilities_epoch, adversarial_probabilities_epochs)
                                a_count += 1

                        if len(attempt_evaluations) > 0:
                            evaluation = common.eval.EvaluationStatistics(attempt_evaluations)
                            evaluation_epoch = adversarial_probabilities_epochs

                        epoch_table.append([attack_config.directory, str(min_adversarial_probabilities_epoch), str(a_count)])

            attack_evaluations.append(evaluation)
            attack_epochs.append(evaluation_epoch)
        evaluations.append(attack_evaluations)
        epochs.append(attack_epochs)

    return evaluations, epochs
