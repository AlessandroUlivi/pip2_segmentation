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

