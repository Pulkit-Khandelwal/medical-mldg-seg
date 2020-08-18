from medpy.io import load
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from medpy.metric import binary
import glob

import torch.utils.data as data
from PIL import Image
import numpy as np
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
from skimage.measure import label


def read_nifti(filepath_image, filepath_label=False):

    img = nib.load(filepath_image)
    image_data = img.get_fdata()

    try:
        lbl = nib.load(filepath_label)
        label_data = lbl.get_fdata()
    except:
        label_data = 0

    return image_data

def save_nifti(image, filepath_name, img_obj=False):
    if img_obj:
        img = nib.Nifti1Image(np.float64(image), img_obj.affine, header=img_obj.header) #np.eye(4)
    else:
        img = nib.Nifti1Image(np.float64(image), np.eye(4)) #np.eye(4)

    nib.save(img, filepath_name)

# get the largest conected component of the segmentation which helps in removing spurious mis-segmented voxels
# used a post-processing step
def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 )
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

voxelspacing =[1, 1, 1]

# list of groundtruth segmentations
list_of_gt = ['', '', '', ......]

# list of output segmentations
list_of_output = ['', '', '', ......]

all_assd = []
all_hd95 = []
for x in range(len(list_of_gt)):

    gt_label = '/path/to/current/groundtruth/segmentation.nii.gz'
    gt_path = read_nifti(gt_label)

    read_segm_output = read_nifti(segm_output)
    segm_output_cc = getLargestCC(read_segm_output)
    segm_output_cc_path = '/path/to/current/segmentation/output.nii.gz'
    save_nifti(segm_output_cc, segm_output_cc_path)

    result, h1 = load(segm_output_cc_path)
    reference, h2 = load(gt_path)

    all_hd95.append(binary.hd95(reference, result, voxelspacing =voxelspacing))
    all_assd.append(binary.assd(reference, result, voxelspacing =voxelspacing))


print(">>>>>>>")
print("hd95 for all subjects", np.mean(all_hd95), np.std(all_hd95))

print(">>>>>>>")
print("assd for all subjects", np.mean(all_assd), np.std(all_assd))

print(">>>>>>>")
print("asd for all subjects", np.mean(all_asd), np.std(all_asd))
