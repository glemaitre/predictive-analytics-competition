"""Compute mask using nilearn masking
"""
import glob
from nilearn import masking

data_path = '/storage/store/data/pac2018_data/pac2018.zip.001_FILES/*.nii'
data_imgs = glob.glob(data_path)

mask = masking.compute_multi_background_mask(data_imgs, n_jobs=5, verbose=2)
mask.to_filename('mask.nii.gz')
