"""A simple example showing how to get X from 3D structural nifti images
"""
from os.path import join
import glob
import numpy as np
from nilearn.input_data import MultiNiftiMasker

# pre-computed mask. see script compute_mask.py for details
mask = 'mask.nii.gz'
# structural images
data_dir = '/storage/store/data/pac2018_data/pac2018.zip.001_FILES'
data_paths = sorted(glob.glob(join(data_dir, '*.nii')))

# Initialize MultiNiftiMasker for list of multiple subjects
multi_masker = MultiNiftiMasker(mask_img=mask, mask_strategy='background')

# Either use fit first and then transform to get X. fit method fits given mask
# image to input images.

X = multi_masker.fit_transform(data_paths)  # outputs list of masked subjects
X = np.concatenate(X)  # (number of samples, number of features)
