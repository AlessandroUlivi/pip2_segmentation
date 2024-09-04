# the present functions are adapted from https://github.com/dl4mia/DL4MIA_Pre-course_Webinar/blob/main/notebooks/utils.py

import os
# from functools import partial
# from itertools import product

import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.tensorboard import SummaryWriter

# import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

# from imageio import imread
import tifffile
from utils import listdirNHF


def load_dataset(input_data_dir, labels_data_dir, stack_axis=0):
    """
    returns the input images and corresponding labels as numpy arrays.
    Inputs:
    - input_data_dir; the directory of the input data. It is expected that only the files to be used as input data are found in the directory. Input data must
    have the same shape.
    - labels_data_dir; the directory of labels data. It is expected that only the files to be used as input data are found in the directory. The names of the labels
    data is expected to match the name of the corresponding input data.
    - stack_axis. Optional. Defauls 0. The axis along which data are stacked in the output arrays.

    Outputs: tuple.
    - position 0. np.array. Inputs data are stacked along stack_axis.
    - position 1. np.array. Labels data are stacked along stack_axis. The position along stack_axis matches the position of the corresponding input data.
    NOTE: data are not shuffled, their order relies on the os.listdir function.
    """
    #initialize a collection list for the input data and a collection list for the corresponding labels
    input_data_l = []
    labels_l = []

    #list all file in the input_data_dir and labels_data_dir
    input_data_list = listdirNHF(input_data_dir)

    #Itearate through the list of input data (names) 
    for position_c, inp_data_n in enumerate(input_data_list):
        #form the full directory of the input data and correspond label.
        inp_data_d = os.path.join(input_data_dir, inp_data_n)
        lab_data_d = os.path.join(labels_data_dir, inp_data_n)
        #Open inputdata and corresponding label
        inp_data = tifffile.imread(inp_data_d)
        lab_data = tifffile.imread(lab_data_d)
        #append the diles in the respective collection lists
        input_data_l.append(inp_data)
        labels_l.append(lab_data)

    # from list of arrays to a single numpy array by stacking along new "batch" axis
    images = np.concatenate([im[None] for im in input_data_l], axis=stack_axis)
    labels = np.concatenate([lb[None] for lb in labels_l], axis=stack_axis)

    return images, labels


def make_dataset_train_val_split(images, labels, validation_fraction=0.20, shuffle_data=True):
    """
    splits inputs data and matching labels into train and validation sub-sets.

    Inputs:
    - images. np.array.
    - labels. np.array. The size of labels axis 0 (shape[0]) must match the size of images axis 0 (shape[0]).
    - validation_fraction. float. Optional. Defauls 0.2. The fraction of data to use as validation dataset.
    - shuffle_data. Bool. Optional. Defaul True. Whether or not to shuffle data before splitting.

    Outputs: tuple.
    - position 0. np.array. Sub-array of images along axis 0. The size of axis 0 corresponds to 1-validation_fraction of the original axis 0 size.
    - position 1. np.array. Sub-array of images along axis 0. The size of axis 0 corresponds to validation_fraction of the original axis 0 size.
    - position 2. np.array. Sub-array of labels along axis 0. The size of axis 0 corresponds to 1-validation_fraction of the original axis 0 size.
    - position 3. np.array. Sub-array of labels along axis 0. The size of axis 0 corresponds to validation_fraction of the original axis 0 size.

    """
    #splits images and labels in train and validation datasets
    (train_images, val_images,
     train_labels, val_labels) = train_test_split(images, labels, shuffle=shuffle_data,
                                                  test_size=validation_fraction)
    #check that the splitting was correctly done
    assert len(train_images) == len(train_labels)
    assert len(val_images) == len(val_labels)
    assert len(train_images) + len(val_images) == len(images)

    return train_images, train_labels, val_images, val_labels


def add_channel(image, target, axis_to_use=0):
    """
    Add an extra dimension to images at position axis_to_use.
    NOTE: Although the position can be changed, this is meant to be done so that it could be interpreted as the color channel dimension by PyTorch. Final output (CWH)
    
    Inputs:
    - image. np.array.
    - target. anything.

    Outputs: tuple.
    - position 0. np.array. Image with extra dimension of size 1 in position axis_to_use.
    - position 1. target.

    """
    # put channel first
    image_w_c = np.expand_dims(image, axis=axis_to_use)
    return image_w_c, target


def normalize(image, target, channel_wise=True):
    """
    returns the min-max normalization of a image.
    Inputs:
    - image. 3D np.array. channel dimension in position 0.
    - target. anything.

    Outputs: tuple.
    - position 0. np.array. min-max normalized image on axes 1 and 2.
    - position 1. target.
    - 
    """
    #initialize a small variable, to prevent a 0 division
    eps = 1.e-6
    #transform image to 'float32' dtype, for allowing a correct division calculation
    image = image.astype('float32')
    #calculate the minumum of the image using the axes 1 and 2
    chan_min = image.min(axis=(1, 2), keepdims=True)
    #subtract the minimum from the image
    image -= chan_min
    #calculate the maximum of the image using the axes 1 and 2
    chan_max = image.max(axis=(1, 2), keepdims=True)
    #divide the image for the maximum value NOTE: add eps to maximum value to prevent a 0 division
    image /= (chan_max + eps)
    return image, target
