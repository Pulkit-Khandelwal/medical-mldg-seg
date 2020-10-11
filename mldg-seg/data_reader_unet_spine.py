"""
This file contains a lot of functions: read, write images, dataloaders,
image transformations, etc.
"""
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
import importlib
import torch
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from scipy.ndimage.filters import convolve
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
from torchvision.transforms import Compose
from nipype.interfaces.ants import N4BiasFieldCorrection
import scipy.ndimage as ndi
from scipy.ndimage import map_coordinates, gaussian_filter

## supporting functions for generating probability based patches
def perturb_patch_locations(patch_locations, radius):
    x, y, z = patch_locations
    x = np.rint(x + np.random.uniform(-radius, radius, len(x)))
    y = np.rint(y + np.random.uniform(-radius, radius, len(y)))
    z = np.rint(z + np.random.uniform(-radius, radius, len(z)))
    return x, y, z

def generate_patch_locations(patches, patch_size, im_size):
    nx = round((patches * 8 * im_size[0] * im_size[0] / im_size[1] / im_size[2]) ** (1.0 / 3))
    ny = round(nx * im_size[1] / im_size[0])
    nz = round(nx * im_size[2] / im_size[0])
    x = np.rint(np.linspace(patch_size, im_size[0] - patch_size, num=nx))
    y = np.rint(np.linspace(patch_size, im_size[1] - patch_size, num=ny))
    z = np.rint(np.linspace(patch_size, im_size[2] - patch_size, num=nz))
    return x, y, z

def generate_patch_probs(patch_locations, patch_size, im_size, label_cropped):
    x, y, z = patch_locations
    seg = label_cropped.astype(np.float32)
    p = []
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                patch = seg[int(x[i] - patch_size / 2) : int(x[i] + patch_size / 2),
                            int(y[j] - patch_size / 2) : int(y[j] + patch_size / 2),
                            int(z[k] - patch_size / 2) : int(z[k] + patch_size / 2)]
                patch = (patch > 0).astype(np.float32)
                percent = np.sum(patch) / (patch_size * patch_size * patch_size)
                p.append((1 - np.abs(percent - 0.5)) * percent)
    p = np.asarray(p, dtype=np.float32)
    p[p == 0] = np.amin(p[np.nonzero(p)])
    p = p / np.sum(p)
    return p

def get_patches_based_prob(image_cropped, label_cropped, patches_per_image, patch_size, image_size):
    base_locs = generate_patch_locations(patches_per_image, patch_size, image_size)
    x, y, z = perturb_patch_locations(base_locs, patch_size / 16)
    probs = generate_patch_probs((x, y, z), patch_size, image_size, label_cropped=label_cropped)
    selections = np.random.choice(range(len(probs)), size=patches_per_image, replace=False, p=probs)

    patch_im_list = []
    patch_lb_list = []

    for num, sel in enumerate(selections):
        i, j, k = np.unravel_index(sel, (len(x), len(y), len(z)))
        patch_im = image_cropped[int(x[i] - patch_size / 2) : int(x[i] + patch_size / 2),
                      int(y[j] - patch_size / 2) : int(y[j] + patch_size / 2),
                      int(z[k] - patch_size / 2) : int(z[k] + patch_size / 2)]

        patch_lb = label_cropped[int(x[i] - patch_size / 2) : int(x[i] + patch_size / 2),
                      int(y[j] - patch_size / 2) : int(y[j] + patch_size / 2),
                      int(z[k] - patch_size / 2) : int(z[k] + patch_size / 2)]

        patch_im_list.append(patch_im)
        patch_lb_list.append(patch_lb)
    return np.asarray(patch_im_list), np.asarray(patch_lb_list)


# read nifti
def read_nifti(filepath):
    img = nib.load(filepath)
    image_data = img.get_fdata()

    # you can define your own "label_path" as you wish.

    lbl = nib.load(label_path)
    label_data = lbl.get_fdata()
    label_data[label_data>0] = 1

    return image_data, label_data

def save_nifti(image, filepath_name):
    """
    Save image as nifti images
    """
    img = nib.Nifti1Image(image, np.eye(4))
    nib.save(img, filepath_name)


def get_patches(image_cropped, label_cropped, all_patches=False, many_patches= False, patch_size=128, patch_shape= (128,128,128), overlap=0, num_patches=20):
    """
    get image, and label patches
    Default: returns one randomly selected patch of the given size, else
    returns a list of all the patches.
    """
    image_shape = np.shape(image_cropped)
    sg_size = image_shape[0]
    cr_size = image_shape[1]
    ax_size = image_shape[2]

    rand_sg = random.randint(0, sg_size-patch_size)
    rand_cr = random.randint(0, cr_size-patch_size)
    rand_ax = random.randint(0, ax_size-patch_size)

    idx1_sg = rand_sg
    idx1_cr = rand_cr
    idx1_ax = rand_ax

    idx2_sg = idx1_sg + patch_size
    idx2_cr = idx1_cr + patch_size
    idx2_ax = idx1_ax + patch_size

    image_patch = image_cropped[idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]
    label_patch = label_cropped[idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]

    if many_patches:
        image_patch_list = []
        label_patch_list = []
        count = 0
        while(count < num_patches):

            rand_sg = random.randint(0, sg_size-patch_size)
            rand_cr = random.randint(0, cr_size-patch_size)
            rand_ax = random.randint(0, ax_size-patch_size)

            idx1_sg = rand_sg
            idx1_cr = rand_cr
            idx1_ax = rand_ax

            idx2_sg = idx1_sg + patch_size
            idx2_cr = idx1_cr + patch_size
            idx2_ax = idx1_ax + patch_size

            image_patch = image_cropped[idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]
            label_patch = label_cropped[idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]

            if np.sum(label_patch) != 0:
                image_patch_list.append(image_patch)
                label_patch_list.append(label_patch)
                count+=1

        # returns a list of patches for all the four modalities
        return image_patch_list, label_patch_list

    if all_patches:
        # get list of patch indices
        array_patch_indices = compute_patch_indices(image_shape, patch_size, overlap, start=None)
        image_patch_list = []
        label_patch_list = []
        for x in range(0, len(array_patch_indices)):
            # get patch from each patch index
            image_patch = get_patch_from_3d_data(image_cropped, patch_shape, patch_index=array_patch_indices[x])
            label_patch = get_patch_from_3d_data(label_cropped, patch_shape, patch_index=array_patch_indices[x])

            image_patch_list.append(image_patch)
            label_patch_list.append(label_patch)

        return image_patch_list, label_patch_list

    else:
        return image_patch, label_patch


def cutup(data, blck, strd):
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    dims = np.r_[nbl, blck]
    data6 = stride_tricks.as_strided(data, strides=strides, shape=dims)
    return data6.reshape(-1, *blck)


##### Transformations

def RandomFlip(image, image_label, patch_size=64):
    """
    Randomly flips the image across the given axes.
    Note from original repo: When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """
    image = np.reshape(image, (patch_size, patch_size, patch_size))
    image_label = np.reshape(image_label, (patch_size, patch_size, patch_size))
    axes = (0, 1, 2)
    image_rot = np.flip(image, axes[1])
    label_rot = np.flip(image_label, axes[1])
    return image_rot, label_rot

def RandomRotate90(image, image_label, patch_size=64):
    """
    Randomly rotate an image
    """
    image = np.reshape(image, (patch_size, patch_size, patch_size))
    image_label = np.reshape(image_label, (patch_size, patch_size, patch_size))
    k = random.randint(0, 4)
    image_rot = np.rot90(image, k, (1, 2))
    label_rot = np.rot90(image_label, k, (1, 2))
    return image_rot, label_rot

def Standardize(image):
    """
    zero-mean, unit standard deviation
    """
    eps=1e-6
    standardized_image = (image - np.mean(image)) / np.clip(np.std(image), a_min=eps, a_max=None)
    return standardized_image

def Normalize(image, min_value=0, max_value=1):
    """
    chnage the intensity range
    """
    value_range = max_value - min_value
    normalized_image = (image - np.min(image)) * (value_range) / (np.max(image) - np.min(image))
    normalized_image = normalized_image + min_value
    return normalized_image

def bias_correct(input_image, label_image):
    """
    N4BiasFieldCorrection: PICSL UPenn
    """
    print("shapes of the input images", np.shape(input_image), np.shape(label_image))
    label_image[label_image> 0] = 1
    bias_free_image = sitk.N4BiasFieldCorrection(input_image, label_image)
    return bias_free_image

def elastic_deformation(X, Y = None, alpha=15, sigma=3, patch_size=64):
    """
    Code adapted from elsewhere. See references in README.
    Elastic deformation of 2D or 3D images on a pixelwise basis
    X: image
    Y: segmentation of the image
    alpha = scaling factor the deformation
    sigma = smooting factor
    inspired by: https://gist.github.com/fmder/e28813c1e8721830ff9c which inspired imgaug through https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    based on [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    First a random displacement field (sampled from a gaussian distribution) is created,
    it's then convolved with a gaussian standard deviation, σ determines the field : very small if σ is large,
        like a completely random field if σ is small,
        looks like elastic deformation with σ the elastic coefficent for values in between.
    Then the field is added to an array of coordinates, which is then mapped to the original image.
    """
    X = np.reshape(X, (patch_size, patch_size, patch_size))
    Y = np.reshape(Y, (patch_size, patch_size, patch_size))

    shape = X.shape
    dx = gaussian_filter(np.random.randn(*shape), sigma, mode="constant", cval=0) * alpha #originally with random_state.rand * 2 - 1
    dy = gaussian_filter(np.random.randn(*shape), sigma, mode="constant", cval=0) * alpha
    if len(shape)==2:
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = x+dx, y+dy

    elif len(shape)==3:
        dz = gaussian_filter(np.random.randn(*shape), sigma, mode="constant", cval=0) * alpha
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = x+dx, y+dy, z+dz

    else:
        raise ValueError("can't deform because the image is not either 2D or 3D")

    return map_coordinates(X, indices, order=3).reshape(shape), map_coordinates(Y, indices, order=0).reshape(shape)

def random_transform(x, y=None,
                     rotation_range_alpha = 0,
                     rotation_range_beta = 0,
                     rotation_range_gamma = 0,
                     height_shift_range = 0,
                     width_shift_range = 0,
                     depth_shift_range = 0,
                     zoom_range = [1, 1],
                     horizontal_flip = False,
                     vertical_flip = False,
                     z_flip = False
                     ):
    '''Random image tranformation of 2D or 3D images
    x: image
    y: segmentation of the image
    # Arguments
        rotation_range_alpha: angle in degrees (0 to 180), produces a range in which to uniformly pick the rotation.
        rotation_range_beta = ...
        rotation_range_gamma = ...
        width_shift_range: fraction of total width, produces a range in which to uniformly pick the shift.
        height_shift_range: fraction of total height, produces a range in which to uniformly pick the shift.
        depth_shift_range: fraction of total depth, produces a range in which to uniformly pick the shift.
        #shear_range: shear intensity (shear angle in radians).
        zoom_range: factor of zoom. A zoom factor per axis will be randomly picked
            in the range [a, b].
        #channel_shift_range: shift range for each channels.
        horizontal_flip: boolean, whether to randomly flip images horizontally.
        vertical_flip: boolean, whether to randomly flip images vertically.
        z_flip: boolean, whether to randomly flip images along the z axis.
    '''
    # use composition of homographies to generate final transform that needs to be applied
    if rotation_range_alpha:
        alpha = np.pi / 180 * np.random.uniform(-rotation_range_alpha, rotation_range_alpha)
    else:
        alpha = 0

    if rotation_range_beta:
        beta = np.pi / 180 * np.random.uniform(-rotation_range_beta, rotation_range_beta)
    else:
        beta = 0

    if rotation_range_gamma:
        gamma = np.pi / 180 * np.random.uniform(-rotation_range_gamma, rotation_range_gamma)
    else:
        gamma = 0

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    cb = np.cos(beta)
    sb = np.sin(beta)

    cg = np.cos(gamma)
    sg = np.sin(gamma)

    img_row_index = 0
    img_col_index = 1
    if len(x.shape) == 2:
        img_z_index = None
        img_channel_index = 2
    elif len(x.shape) == 3:
        img_z_index = 2
        img_channel_index = 3
    else:
        raise ValueError("can't augment because the image is not either 2D or 3D")

    if height_shift_range:
        tx = np.random.uniform(-height_shift_range, height_shift_range) * x.shape[img_row_index]
    else:
        tx = 0

    if width_shift_range:
        ty = np.random.uniform(-width_shift_range, width_shift_range) * x.shape[img_col_index]
    else:
        ty = 0

    if depth_shift_range:
        tz = np.random.uniform(-depth_shift_range, depth_shift_range) * x.shape[img_z_index]
    else:
        tz = 0

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy, zz = 1, 1, 1
    else:
        zx, zy, zz = np.random.uniform(zoom_range[0], zoom_range[1], 3)

    if len(x.shape) == 2:
        rotation_matrix = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                                    [np.sin(alpha), np.cos(alpha), 0],
                                    [0, 0, 1]])

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])

        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        transform_matrix = np.dot(np.dot(rotation_matrix, translation_matrix), zoom_matrix)
        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center_2d(transform_matrix, h, w)

        apply_transform_gd = apply_transform_2d

    else:
        rotation_matrix = np.array([[cb*cg, -cb*sg, sb, 0],
                                [ca*sg+sa*sb*cg, ca*cg-sa*sb*sg, -sa*cb, 0],
                                [sa*sg-ca*sb*cg, sa*cg+ca*sb*sg, ca*cb, 0],
                                [0, 0, 0, 1]])

        translation_matrix = np.array([[1, 0, 0, tx],
                                       [0, 1, 0, ty],
                                       [0, 0, 1, tz],
                                       [0, 0, 0, 1]])

        zoom_matrix = np.array([[zx, 0, 0, 0],
                                [0, zy, 0, 0],
                                [0, 0, zz, 0],
                                [0, 0, 0, 1]])

        transform_matrix = np.dot(np.dot(rotation_matrix, translation_matrix), zoom_matrix)
        h, w, d = x.shape[img_row_index], x.shape[img_col_index], x.shape[img_z_index]
        transform_matrix = transform_matrix_offset_center_3d(transform_matrix, h, w, d)

        apply_transform_gd = apply_transform_3d

    if y is None:
        x = np.expand_dims(x, len(x.shape))
        x = apply_transform_gd(x, transform_matrix, img_channel_index)

        if horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)

        if vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)

        if z_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_z_index)

        x = np.squeeze(x)
        return x, None

    else:
        x = np.expand_dims(x, len(x.shape))
        y = np.expand_dims(y, len(y.shape))
        x = apply_transform_gd(x, transform_matrix, img_channel_index)
        y = apply_transform_gd(y, transform_matrix, img_channel_index)

        if horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                y = flip_axis(y, img_col_index)

        if vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                y = flip_axis(y, img_row_index)

        if z_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_z_index)
                y = flip_axis(y, img_z_index)

        x = np.squeeze(x)
        y = np.squeeze(y)
        return x, y


class BatchImageGenerator(data.Dataset):
    def __init__(self, image_list, modality, transform=False, patch_size=32, n_patches_transform=30, all_patches=False, is_test=False, val=False):

        imlist = []
        #self.base_training_folder = base_training_folder
        self.image_list = image_list
        self.modality = modality
        self.transform = transform
        self.patch_size = patch_size
        self.val = val
        self.patches_per_image = 50
        if self.val == True:
            self.patches_per_image = 50
            
        self.image_size = (129, 166, 128)
        self.n_patches_transform = n_patches_transform
        self.is_test = is_test

        with open(self.image_list, 'r') as rf:
            count=0
            for line in rf.readlines():
                impath=line
                impath=impath.replace('\n', '')
                imlist.append(impath)
                count+=1

        self.imlist=imlist

    def __getitem__(self, index):
        
        single_data_name = self.imlist[index]
        print(">>>>>>>> current image is >>>>>>>>>>> ", single_data_name)
        image_data, label_data = read_nifti(single_data_name)
        image_standardized = Standardize(image_data)
        image_normalized = Normalize(image_standardized, min_value=0, max_value=1)
        image_cropped, label_cropped = image_normalized, label_data
        
        image_shape = np.shape(image_data)
        sg_size = image_shape[0]
        cr_size = image_shape[1]
        ax_size = image_shape[2]
        step_size = 40
        img_size = [sg_size, cr_size, ax_size]
        
        if self.is_test is True:
            c=1
            # zero-pad the image
            #print(np.shape(image_cropped))
            image_cropped = np.pad(image_cropped, ((step_size//2, step_size//2) , (step_size//2, step_size//2), (step_size//2, step_size//2)), 'symmetric')
            label_cropped = np.pad(label_cropped, ((step_size//2, step_size//2) , (step_size//2, step_size//2), (step_size//2, step_size//2)), 'symmetric')
            image_shape = np.shape(image_cropped)
            sg_size = image_shape[0]
            cr_size = image_shape[1]
            ax_size = image_shape[2]
            img_size = [sg_size, cr_size, ax_size]
            #print(np.shape(image_cropped))
            
            image_patch_list = []
            label_patch_list = []
            patch_indices = []
            for x in range(0, ax_size - self.patch_size, step_size):
                idx1_ax = x
                idx2_ax = idx1_ax + self.patch_size

                for y in range(0, cr_size - self.patch_size, step_size):
                    idx1_cr = y
                    idx2_cr = idx1_cr + self.patch_size

                    for z in range(0, sg_size - self.patch_size, step_size):
                        #print(c)
                        idx1_sg = z
                        idx2_sg = idx1_sg + self.patch_size

                        image_return = image_cropped[idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]
                        label_return = label_cropped[idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]
                        
                        if np.sum(label_return) != 0:
                            image_patch_list.append(image_return)
                            label_patch_list.append(label_return)
                            c=c+1
                            patch_indices.append([(idx1_sg,idx2_sg), (idx1_cr,idx2_cr), (idx1_ax,idx2_ax)])
                            
            image_patch_list = np.reshape(image_patch_list, (c-1, 1, self.patch_size, self.patch_size, self.patch_size))
            label_patch_list = np.reshape(label_patch_list, (c-1, self.patch_size, self.patch_size, self.patch_size))
            print("size of the images and labels are", np.shape(image_patch_list), np.shape(label_patch_list))
            print("type is", type(image_patch_list), type(label_patch_list))
            return image_patch_list, label_patch_list, patch_indices, img_size

        else:

            image_patch_list, label_patch_list = get_patches(image_cropped, label_cropped, many_patches=True, all_patches=False, patch_size=self.patch_size, overlap=0, num_patches=self.patches_per_image)

            # the above was returned as a numpy array
            image_patch_list = np.reshape(image_patch_list, (self.patches_per_image, 1, self.patch_size, self.patch_size, self.patch_size))
            label_patch_list = np.reshape(label_patch_list, (self.patches_per_image, self.patch_size, self.patch_size, self.patch_size))
        
            print(">>>>>> see should be like this >>>>>>", np.shape(image_patch_list), np.shape(label_patch_list))

            if self.transform is True:

                # randomly select some patches to apply
                indices_random_patches = random.sample(range(0, len(image_patch_list)), k=self.n_patches_transform)
                transformed_list_images = []
                transformed_list_label = []

                for x in indices_random_patches:
                    image_flip, label_flip = RandomFlip(image_patch_list[x], label_patch_list[x], patch_size=self.patch_size)
                    image_rot, label_rot = RandomRotate90(image_patch_list[x], label_patch_list[x], patch_size=self.patch_size)
                    #image_elastic, label_elastic = elastic_deformation(image_patch_list[x], label_patch_list[x], patch_size=self.patch_size)

                    """
                    image_random, label_random = random_transform(image_patch_list[x], label_patch_list[x],
                                         rotation_range_alpha = 0,
                                         rotation_range_beta = 0,
                                         rotation_range_gamma = 0,
                                         height_shift_range = 0,
                                         width_shift_range = 0,
                                         depth_shift_range = 0,
                                         zoom_range = [1, 1],
                                         horizontal_flip = False,
                                         vertical_flip = True,
                                         z_flip = True)
                    """
                    transformed_list_images.extend([image_flip, image_rot])
                    transformed_list_label.extend([label_flip, label_rot])

                transformed_list_images = np.asarray(transformed_list_images)
                transformed_list_label = np.asarray(transformed_list_label)

                transformed_list_images = np.reshape(transformed_list_images, (self.n_patches_transform*2, 1, self.patch_size, self.patch_size, self.patch_size))
                transformed_list_label = np.reshape(transformed_list_label, (self.n_patches_transform*2, self.patch_size, self.patch_size, self.patch_size))

                images_all = np.vstack((transformed_list_images, image_patch_list))
                labels_all = np.vstack((transformed_list_label, label_patch_list))

                #print(">>>>> actual number of images for an image <<<<<", np.shape(images_all), np.shape(labels_all))

                return images_all, labels_all

            else:
                # returns numpy array of patches of images, and labels
                return image_patch_list, label_patch_list
        
    def __len__(self):
        return len(self.imlist)
