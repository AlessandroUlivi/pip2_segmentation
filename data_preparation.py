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



