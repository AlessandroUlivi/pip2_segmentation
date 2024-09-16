import os
import numpy as np
import random


def listdirNHF(input_directory):
    """
    creates a list of files in the input directory, avoiding hidden files. Hidden files are identified by the fact that they start with a .
    input: directory of the folder whose elements have to be listed.
    output: list of elements in the input-folder, except hidden files. 
    """
    return [f for f in os.listdir(input_directory) if not f.startswith(".")]


def check_files_else_make_folders(folder_dir_to_create):
    """
    Check if a folder is present and if it contains any object. If the folder is not present, it creates it.
    inputs:
    - the directory of a folder to check
    outputs:
    - False if the folder is not present.
    - False if the folder is present and it is empty
    - True is the folder is present and it is not empty
    """
    #if the folder does not exist
    if not os.path.exists(folder_dir_to_create):
        os.makedirs(folder_dir_to_create)
        #return False as no file is present in the folder
        return False
    #if the folder exists
    else:
        #get a list of files in the folder
        folders_files_list = listdirNHF(folder_dir_to_create)
        #if files are present in the folder
        if len(folders_files_list)>0:
            #return True as files are present in the folder
            return True
        #if no files are present in the folder
        else:
            #return False as no file is present in the folder
            return False

def get_random_image_label_pair(images, labels):
    """
    returns a randomly chosen image and the correponding label

    Input:
    - images. np.array.
    - labels. np.array. The size of the axis 0 (shape[0]) must match the size of images.

    Output: tuple.
    - position 0. np.array. Sub-array of images randomly picked from the axis 0.
    - position 1. np.array. Sub-array of labels picked from the axis 0 in the position matching position 0 output.

    """

    #get a random index along the axis 0 of images
    random_indx = random.choice(range(images.shape[0]))

    return images[random_indx, ...], labels[random_indx, ...]

def crop_2d(x, y):
        """
        Center-crop x to match spatial dimensions given by y. Note: for spatial dimensions are meant the xy image dimension, which expected in positions
        -1 and -2. It is assumed that no dimension of y has size > than the corresponding dimension in x.
        """

        #per each dimension, get the how many pixels should be added/removed (offset) from x in order to match y.
        #Note: the offset is divided by 2 because it will be added/removed from x to both sides of each dimension.
        offset = tuple((a - b) // 2 for a, b in zip(x.shape[-2:], y.shape[-2:]))

        #create a tuple of "slice" objects, with one object per dimension of the inputs.
        #each slice indicates the beginning and end (initial and final indeces) of the part of x which should be kept.
        #NOTE Because each slice starts at the index 'offset' and ends at the index 'offset + size of y' the process effectively guarantees
        #that the central part of x is maintened and matches y.
        slices = tuple(slice(o, o + s) for o, s in zip(offset, y.shape[-2:]))

        return x[slices]
