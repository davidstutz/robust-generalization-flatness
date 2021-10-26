import torch
import numpy
import PIL


def Debug():
    def _debug(image):
        print(type(image))
        if isinstance(image, PIL.Image.Image):
            print(image.size)
        elif isinstance(image, numpy.ndarray):
            print(image.dtype, image.shape, numpy.max(image), numpy.min(image), numpy.mean(image))
        elif isinstance(image, torch.Tensor):
            print(image.dtype, image.shape, torch.max(image).item(), torch.min(image).item(), torch.mean(image).item())
        else:
            assert False
        return image
    return _debug


class CutoutAfterToTensor(object):
    """
    https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    """
    def __init__(self, n_holes, length, fill_color=torch.tensor([0,0,0])):
        self.n_holes = n_holes
        self.length = length
        self.fill_color = fill_color

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.shape[1]
        w = img.shape[2]
        mask = numpy.ones((h, w), numpy.float32)
        for n in range(self.n_holes):
            y = numpy.random.randint(h)
            x = numpy.random.randint(w)
            y1 = numpy.clip(y - self.length // 2, 0, h)
            y2 = numpy.clip(y + self.length // 2, 0, h)
            x1 = numpy.clip(x - self.length // 2, 0, w)
            x2 = numpy.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask + (1 - mask) * self.fill_color[:, None, None]
        return img
