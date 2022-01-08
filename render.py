import numpy as np
import matplotlib.pyplot as plt
import torch


def sample_to_np(sample):
    """"
    Denormalize and revert transposition on image and mask, converting to numpy ndarrays for plt imshow format.
    """
    image = sample['image'] * sample['ostd'] + sample['omean']  # denormalize
    image = image.permute(1, 2, 0).detach(
    ).numpy().copy()     # revert channel order
    mask = sample['mask'].permute(
        1, 2, 0).numpy().copy()      # revert channel order
    return image, mask


def view_samples(sample_batch, n):
    """
    Visualize n random samples from data loader batch.

    Args:
        sample_batch (torch.Tensor): batch containing multiple slice and mask samples.
        n (int): number of samples to display
    """
    assert (n <= sample_batch.shape[0]), "n is greater than batch size"

    perm = torch.randperm(n)
    samples = sample_batch[perm]

    fig, axs = plt.subplots(n, 2)
    for i, sample in enumerate(samples):
        axs[i][0].imshow(sample['image'])
        axs[i][1].imshow(sample['mask'])
    plt.show()

