import random
import torch
import torchvision
import torch.utils.data
from . import utils
import numpy
from . import paths
import skimage.transform
from .log import log


class CleanDataset(torch.utils.data.Dataset):
    """
    General, clean dataset used for training, testing and attacking.
    """

    def __init__(self, images, labels, indices=None, resize=None):
        """
        Constructor.

        :param images: images/inputs
        :type images: str or numpy.ndarray
        :param labels: labels
        :type labels: str or numpy.ndarray
        :param indices: indices
        :type indices: numpy.ndarray
        :param resize: resize in [channels, height, width
        :type resize: resize
        """

        self.images_file = None
        """ (str) File images were loaded from. """

        self.labels_file = None
        """ (str) File labels were loaded from. """

        if isinstance(images, str):
            self.images_file = images
            images = utils.read_hdf5(self.images_file)
            log('read %s' % self.images_file)
        if not images.dtype == numpy.float32:
            images = images.astype(numpy.float32)

        if isinstance(labels, str):
            self.labels_file = labels
            labels = utils.read_hdf5(self.labels_file)
            log('read %s' % self.labels_file)
        labels = numpy.squeeze(labels)
        if not labels.dtype == int:
            labels = labels.astype(int)

        assert isinstance(images, numpy.ndarray)
        assert isinstance(labels, numpy.ndarray)
        assert images.shape[0] == labels.shape[0]

        if indices is None:
            indices = range(images.shape[0])
        assert numpy.min(indices) >= 0
        assert numpy.max(indices) < images.shape[0]

        images = images[indices]
        labels = labels[indices]

        if resize is not None:
            assert isinstance(resize, list)
            assert len(resize) == 3

            size = images.shape
            assert len(size) == 4

            # resize
            if resize[1] != size[1] or resize[2] != size[2]:
                out_images = numpy.zeros((size[0], resize[1], resize[2], size[3]), dtype=numpy.float32)
                for n in range(out_images.shape[0]):
                    out_images[n] = skimage.transform.resize(images[n], (resize[1], resize[2]))
                images = out_images

            # update!
            size = images.shape

            # color to grayscale
            if resize[0] == 1 and size[3] == 3:
                out_images = numpy.zeros((size[0], size[1], size[2], 1), dtype=numpy.float32)
                for n in range(out_images.shape[0]):
                    out_images[n, :, :, 0] = 0.2125 * images[n, :, :, 0] + 0.7154 * images[n, :, :, 1] + 0.0721 * images[n, :, :, 2]
                images = out_images

            # grayscale to color
            if resize[0] == 3 and size[3] == 1:
                out_images = numpy.zeros((size[0], size[1], size[2], 3), dtype=numpy.float32)
                for n in range(out_images.shape[0]):
                    out_images[n, :, :, 0] = images[n, :, :, 0]
                    out_images[n, :, :, 1] = images[n, :, :, 0]
                    out_images[n, :, :, 2] = images[n, :, :, 0]
                images = out_images

        self.images = images
        """ (numpy.ndarray) Inputs. """

        self.labels = labels
        """ (numpy.ndarray) Labels. """

        self.indices = indices
        """ (numpy.ndarray) Indices. """

        self.N_class = numpy.max(labels) + 1
        """ (int) Number of classes. """

        self.transform = None
        """ (torchvision.transforms.Transform) Transforms. """

    def __getitem__(self, index):
        assert index < len(self)
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        assert self.images.shape[0] == self.labels.shape[0]
        return self.images.shape[0]

    def __add__(self, other):
        return torch.utils.data.ConcatDataset([self, other])


class AdversarialDataset(torch.utils.data.Dataset):
    """
    Dataset consisting of adversarial examples.
    """

    def __init__(self, images, perturbations, labels, indices=None):
        """
        Constructor.

        :param images: images
        :type images: str or numpy.ndarray
        :param perturbations: additive perturbations
        :type perturbations: str or numpy.ndarray
        :param labels: true labels
        :type labels: str or numpy.ndarray
        :param indices: indices
        :type indices: numpy.ndarray
        """

        self.images_file = None
        """ (str) File images were loaded from. """

        self.perturbations_file = None
        """ (str) File perturbations were loaded from. """

        self.labels_file = None
        """ (str) File labels were loaded from. """

        if isinstance(images, str):
            self.images_file = images
            images = utils.read_hdf5(self.images_file)
        if not images.dtype == numpy.float32:
            images = images.astype(numpy.float32)

        if isinstance(images, str):
            self.perturbations_file = perturbations
            perturbations = utils.read_hdf5(self.perturbations_file)
        if not perturbations.dtype == numpy.float32:
            perturbations = perturbations.astype(numpy.float32)

        if isinstance(labels, str):
            self.labels_file = labels
            labels = utils.read_hdf5(self.labels_file)
        labels = numpy.squeeze(labels)
        if not labels.dtype == int:
            labels = labels.astype(int)

        assert isinstance(images, numpy.ndarray)
        assert isinstance(perturbations, numpy.ndarray)
        assert isinstance(labels, numpy.ndarray)
        assert images.shape[0] == labels.shape[0]

        if len(perturbations.shape) == len(images.shape):
            perturbations = numpy.expand_dims(perturbations, axis=0)

        assert len(perturbations.shape) == len(images.shape) + 1
        for d in range(len(images.shape)):
            assert perturbations.shape[d + 1] == images.shape[d]

        if indices is None:
            indices = range(images.shape[0])
        assert numpy.min(indices) >= 0
        assert numpy.max(indices) < images.shape[0]

        self.images = images[indices]
        """ (numpy.ndarray) Inputs. """

        self.perturbations = perturbations[:, indices]
        """ (numpy.ndarray) Perturbations. """

        self.labels = labels[indices]
        """ (numpy.ndarray) Labels. """

        self.indices = indices
        """ (numpy.ndarray) Indices. """

    def __getitem__(self, index):
        assert index < len(self)
        attempt_index = index // self.images.shape[0]
        sample_index = index % self.images.shape[0]
        assert attempt_index < self.perturbations.shape[0]
        assert sample_index < self.images.shape[0]

        return self.perturbations[attempt_index, sample_index] + self.images[sample_index], self.labels[sample_index]

    def __len__(self):
        assert self.images.shape[0] == self.labels.shape[0]
        assert self.images.shape[0] == self.perturbations.shape[1]
        return self.perturbations.shape[0]*self.perturbations.shape[1]

    def __add__(self, other):
        return torch.utils.data.ConcatDataset([self, other])


class MNISTTrainSet(CleanDataset):
    def __init__(self, indices=None, resize=None):
        super(MNISTTrainSet, self).__init__(paths.mnist_train_images_file(), paths.mnist_train_labels_file(), indices=indices, resize=resize)


class MNISTTestSet(CleanDataset):
    def __init__(self, indices=None, resize=None):
        super(MNISTTestSet, self).__init__(paths.mnist_test_images_file(), paths.mnist_test_labels_file(), indices=indices, resize=resize)


class Cifar10TrainSet(CleanDataset):
    def __init__(self, indices=None, resize=None):
        super(Cifar10TrainSet, self).__init__(paths.cifar10_train_images_file(), paths.cifar10_train_labels_file(), indices=indices, resize=resize)


class Cifar10TestSet(CleanDataset):
    def __init__(self, indices=None, resize=None):
        super(Cifar10TestSet, self).__init__(paths.cifar10_test_images_file(), paths.cifar10_test_labels_file(), indices=indices, resize=resize)


class TinyImages500kTrainSet(CleanDataset):
    def __init__(self, indices=None, resize=None):
        super(TinyImages500kTrainSet, self).__init__(paths.tinyimages500k_train_images_file(), paths.tinyimages500k_train_labels_file(), indices=indices, resize=resize)


class SemiSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, clean_dataset, aux_dataset):
        self.images = clean_dataset.images
        self.labels = clean_dataset.labels
        assert self.images.shape[0] == self.labels.shape[0]
        assert len(self.labels.shape) == 1

        self.sup_end = self.images.shape[0]
        self.unsup_length = aux_dataset.images.shape[0]
        self.sup_indices = list(range(self.sup_end))
        self.unsup_indices = list(range(self.sup_end, self.sup_end + self.unsup_length))

        self.images = numpy.concatenate((self.images, aux_dataset.images), axis=0)
        self.labels = numpy.concatenate((self.labels, numpy.ones(aux_dataset.labels.shape, dtype=int)*-1), axis=0)
        self.transform = None

    def __getitem__(self, index):
        if isinstance(index, list) and self.transform is not None:
            images = self.images[index]
            labels = self.labels[index]
            supervised = numpy.zeros(labels.shape[0])

            for i in range(len(index)):
                if self.transform is not None:
                    images[i] = self.transform(images[i])
                if index[i] >= self.sup_end:
                    supervised[i] = 0
                else:
                    supervised[i] = 1

            return images, labels
        else:
            assert index < len(self)
            image = self.images[index]
            if self.transform is not None:
                image = self.transform(image)
            supervised = 1
            if index >= self.sup_end:
                supervised = 0
            return image, self.labels[index], supervised

    def __len__(self):
        assert self.images.shape[0] == self.labels.shape[0]
        return self.images.shape[0]


class PseudoLabeledSemiSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, clean_dataset, aux_dataset):
        self.images = clean_dataset.images
        self.labels = clean_dataset.labels
        assert self.images.shape[0] == self.labels.shape[0]
        assert len(self.labels.shape) == 1

        self.N_class = clean_dataset.N_class
        self.sup_end = self.images.shape[0]
        self.unsup_length = aux_dataset.images.shape[0]
        self.sup_indices = list(range(self.sup_end))
        self.unsup_indices = list(range(self.sup_end, self.sup_end + self.unsup_length))

        self.images = numpy.concatenate((self.images, aux_dataset.images), axis=0)
        self.labels = numpy.concatenate((self.labels, aux_dataset.labels), axis=0)

        self.transform = None

    def __getitem__(self, index):
        if isinstance(index, list) and self.transform is not None:
            images = self.images[index]
            labels = self.labels[index]
            supervised = numpy.zeros(labels.shape[0])

            for i in range(len(index)):
                images[i] = self.transform(images[i])
                if index[i] >= self.sup_end:
                    supervised[i] = 0
                else:
                    supervised[i] = 1

            return images, labels
        else:
            assert index < len(self)
            image = self.images[index]
            if self.transform is not None:
                image = self.transform(image)
            supervised = 1
            if index >= self.sup_end:
                supervised = 0
            return image, self.labels[index]

    def __len__(self):
        assert self.images.shape[0] == self.labels.shape[0]
        return self.images.shape[0]


class RotatedSemiSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, clean_dataset, aux_dataset):
        assert getattr(clean_dataset, 'N_class', None) is not None, 'clean dataset does not have the N_Class property'
        self.images = clean_dataset.images
        self.labels = clean_dataset.labels
        assert self.images.shape[0] == self.labels.shape[0]
        assert len(self.labels.shape) == 1

        self.sup_end = self.images.shape[0]
        self.unsup_length = aux_dataset.images.shape[0]
        self.sup_classes = clean_dataset.N_class
        self.unsup_classes = 4
        self.sup_indices = list(range(self.sup_end))
        self.unsup_indices = list(range(self.sup_end, self.sup_end + self.unsup_length))

        self.images = numpy.concatenate((self.images, aux_dataset.images), axis=0)
        self.labels = numpy.concatenate((self.labels, numpy.zeros(aux_dataset.labels.shape, dtype=int)), axis=0)

        self.topil = torchvision.transforms.ToPILImage()
        self.totensor = torchvision.transforms.ToTensor()
        self.transform = None

    def __getitem__(self, index):
        if isinstance(index, list) and self.transform is not None:
            images = self.images[index]
            labels = self.labels[index]
            supervised = numpy.zeros(labels.shape[0])

            for i in range(len(index)):
                if index[i] >= self.sup_end:
                    degrees = [0, 90, 180, 270]
                    rand_choice = random.randint(0, len(degrees) - 1)

                    images[i] = skimage.transform.rotate(images[i], degrees[rand_choice], resize=False)
                    labels[i] = rand_choice
                    supervised[i] = 0
                else:
                    supervised[i] = 1

                if self.transform is not None:
                    images[i] = self.transform(images[i])

            return images, labels, supervised
        else:
            assert index < len(self)
            image = self.images[index]
            label = self.labels[index]
            supervised = 1

            if index >= self.sup_end:
                degrees = [0, 90, 180, 270]
                rand_choice = random.randint(0, len(degrees) - 1)

                image = skimage.transform.rotate(image, degrees[rand_choice], resize=False)
                label = rand_choice
                supervised = 0

            if self.transform is not None:
                image = self.transform(image)

            return image, label, supervised

    def __len__(self):
        assert self.images.shape[0] == self.labels.shape[0]
        return self.images.shape[0]


class SemiSupervisedSampler(torch.utils.data.Sampler):
    """Balanced sampling from the labeled and unlabeled data"""
    def __init__(self, sup_inds, unsup_inds, batch_size, fraction=0.5, num_batches=None):
        unsup_fraction = fraction
        if unsup_fraction is None or unsup_fraction < 0:
            self.sup_inds = sup_inds + unsup_inds
            unsup_fraction = 0.0
        else:
            self.sup_inds = sup_inds
            self.unsup_inds = unsup_inds

        self.batch_size = batch_size
        unsup_batch_size = int(batch_size * unsup_fraction)
        self.sup_batch_size = batch_size - unsup_batch_size

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            self.num_batches = int(numpy.ceil(len(self.sup_inds) / self.sup_batch_size))

        super().__init__(None)

    def __iter__(self):
        batch_counter = 0
        while batch_counter < self.num_batches:
            sup_inds_shuffled = [self.sup_inds[i] for i in torch.randperm(len(self.sup_inds))]
            for sup_k in range(0, len(self.sup_inds), self.sup_batch_size):
                if batch_counter == self.num_batches:
                    break
                batch = sup_inds_shuffled[sup_k:(sup_k + self.sup_batch_size)]
                if self.sup_batch_size < self.batch_size:
                    batch.extend([self.unsup_inds[i] for i in torch.randint(high=len(self.unsup_inds), size=(self.batch_size - len(batch),), dtype=torch.int64)])

                # this shuffle operation is very important, without it
                # batch-norm / DataParallel hell ensues
                numpy.random.shuffle(batch)
                yield batch
                batch_counter += 1

    def __len__(self):
        return self.num_batches
