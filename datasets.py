import torch
from torch.utils.data import Dataset
from os.path import join
import glob
from skimage.io import imread


class BrainSegDataset(Dataset):
    """Brain Segmentation dataset containing images and segmentation masks"""

    def __init__(self, df, image_dir, transform=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing patient ids and metagenomic data.
            image_dir (string): Directory containing mri images and masks.
            transform (object): Sample transformations to be applied to retrieved samples.
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

        self.images = []  # image paths
        self.masks = []  # mask paths

        for patient in self.df.Patient:
            patient = join(image_dir, patient)
            patient += "_*"
            mask_dirs = join(patient, '*_mask.tif')
            pmasks = glob.glob(mask_dirs)
            self.masks += pmasks
            self.images += [m.replace('_mask', '') for m in pmasks]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        impath = self.images[idx]
        mpath = self.masks[idx]

        image = imread(impath)
        mask = imread(mpath)
        sample = {'image': image, 'mask': mask,
                  'impath': impath, 'mpath': mpath}  # package original image and mask paths in sampled object
        if self.transform:
            sample = self.transform(sample)
        return sample
