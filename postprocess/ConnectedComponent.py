import numpy as np
from skimage.measure import label
import torch.utils.data as data
from PIL import Image
import os
import os.path
import glob
import warnings
import shutil
import random
from scipy import ndimage
import SimpleITK as sitk
import nibabel as nib
from scipy.ndimage import map_coordinates, gaussian_filter

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def read_nifti(filepath_image, filepath_label):

    img = nib.load(filepath_image)
    image_data = img.get_fdata()

    lbl = nib.load(filepath_label)
    label_data = lbl.get_fdata()

    return image_data, label_data

def save_nifti(image, filepath_name):
    img = nib.Nifti1Image(image, np.eye(4))
    nib.save(img, filepath_name)


filepath_image = '/path/to/segmentation.nii.gz'
img = nib.load(filepath_image)
image_data = img.get_fdata()

largestCC = getLargestCC(image_data)
img = nib.Nifti1Image(np.float64(largestCC), np.eye(4))
nib.save(img, '/path/to/segmentation_largest_connected_component.nii.gz')
