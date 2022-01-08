from skimage.transform import resize, rotate
import numpy as np
import torch


class Rescale(object):
    """
    Rescale a given image and its mask to the specified dimensions.
     Args:
        output_size (tuple or int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # using skimage transform.resize
        sample['image'] = resize(image, (new_h, new_w))
        sample['mask'] = resize(mask, (new_h, new_w))

        return sample


class RandomCrop(object):
    """
    Randomly crop a given image and its mask, with the given probability.

    Args:
        output_size (tuple or int): Desired output size.
        prob (float): chance of image and mask being cropped.
    """

    def __init__(self, output_size, prob):
        self.prob = prob
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if self.prob > np.random.random():
            h, w = image.shape[:2]
            new_h, new_w = self.output_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            sample['image'] = image[top: top + new_h,
                                    left: left + new_w]

            sample['mask'] = mask[top: top + new_h,
                                  left: left + new_w]

        return sample


class RandomFlip(object):
    """
    Randomly apply horizontal or vertical flip to an image and its mask, with given probability.

    Args:
        direc (str): "horizontal" or "vertical" flip axis.
        prob (float): probability of applying a flip.
    """

    def __init__(self, prob, direc):
        assert isinstance(prob, float)
        assert direc in ["horizontal", "vertical"]
        self.prob = prob
        self.flip_axs = 1 if direc == "horizontal" else 0

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if np.random.random() < self.prob:
            sample['image'], sample['mask'] = np.flip(
                image, axis=self.flip_axs).copy(), np.flip(mask, axis=self.flip_axs).copy()
        return sample


class Normalize(object):
    """Normalize a given image by pixel intensity, mean and standard deviation"""

    def __call__(self, sample):
        image = sample['image']
        image = image/255
        mean = np.mean(image, axis=(0, 1, 2))
        std = np.std(image, axis=(0, 1, 2))
        image = (image - mean) / std
        sample['image'] = image
        sample['omean'], sample['ostd'] = mean, std
        return sample


class RandomRotate(object):
    """
    Apply random rotation to the given image and its mask.

    Args:
        min_angle (float): minimum rotation angle (degrees).
        max_angle (float): maximum rotation angle (degrees).
        prob (float): 0 to 1 probability of applying a random rotation.
    """

    def __init__(self, min_angle, max_angle, prob):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if np.random.random() < self.prob:
            angle = (self.max_angle - self.min_angle) * \
                np.random.random() + self.min_angle
            sample['image'], sample['mask'] = rotate(
                image, angle), rotate(mask, angle)
        return sample


class ToTensor():
    """Convert numpy ndarray image and mask to torch Tensor by transposition"""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        # switch channel ordering from HxWxC to CxHxW
        image = image.transpose((2, 0, 1))
        mask = np.expand_dims(mask, axis=2)
        mask = mask.transpose((2, 0, 1))
        sample['image'] = torch.from_numpy(image)
        sample['mask'] = torch.from_numpy(mask)
        return sample
