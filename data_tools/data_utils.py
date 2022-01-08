import pandas as pd
from os.path import join
from skimage.io import imread
import glob
import numpy as np



def mask_distribution(patient_dir, img_root):
    """
    Returns mask counts and number of blanks for each provided patient directory.

    Args:
        patient_dir (string): Directory path for a volume of scans from a single patient.
    """

    patient_dir = join(img_root, patient_dir)
    glob_dir = join(f"{patient_dir}_*", '*_mask.tif')
    mask_files = glob.glob(glob_dir)
    n_masks = len(mask_files)
    blanks = 0
    for mask in mask_files:
        img = imread(mask, as_gray=True)
        assert img.shape == (256, 256), "Incorrect mask dims found."
        mri = imread(mask.replace('_mask', ''))
        assert mri.shape == (256, 256, 3), "Incorrect image dims found."

        if np.max(img) == 0:
            blanks += 1
    return (n_masks, blanks)


