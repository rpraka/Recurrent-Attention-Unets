import pandas as pd
from data_utils import mask_distribution
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import BrainSegDataset
from sample_transforms import Normalize, Rescale, RandomFlip, RandomCrop, ToTensor


def stratified_data_gen(meta_path, img_root, train_size, batch_size, num_workers, random_state):
    """
    Generate train and validation DataLoaders. Stratified by percentage of blank slices / volume.
    Ensure NOT to split the same patient volume across train/valid sets, I frequently see people make this mistake.

    Args:
        meta_path (string): Path of patient metadata csv.
        train_size (float): Percentage of data to use for training (0.0 to 1.0).
        random_state (int): Random seed used for splitting.
    """
    df = pd.read_csv(meta_path)
    df['n_masks'], df['n_blanks'] = zip(
        *df.Patient.apply(lambda x: mask_distribution(x, img_root=img_root)))  # get mask distribution metrics for each patient

    # cluster percentages based on their distribution
    df['bracket'] = ((df.n_blanks/df.n_masks) * 10 + 0.5).astype(int)

    # lower brackets must have 2 or more samples
    df.loc[df.bracket == 3, 'bracket'] = 4

    train_df, valid_df = train_test_split(df, train_size=train_size,
                                          shuffle=True, random_state=random_state, stratify=df['bracket'])

    train_transforms = transforms.Compose([Normalize(), Rescale(256), RandomFlip(
        0.6, 'horizontal'), RandomFlip(0.6, 'vertical'), RandomCrop(200, 0.5), Rescale(256), ToTensor()])
    valid_transforms = transforms.Compose(
        [Normalize(), Rescale(256), ToTensor()])
    train_dset = BrainSegDataset(
        train_df, img_root, transform=train_transforms)
    val_dset = BrainSegDataset(valid_df, img_root, transform=valid_transforms)

    train_loader = DataLoader(dataset=train_dset, shuffle=True,
                              batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dset, shuffle=False,
                            batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader
