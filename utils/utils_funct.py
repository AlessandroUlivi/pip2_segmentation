import os
import numpy as np
import random
import torch
import pandas as pd
from scipy.interpolate import make_interp_spline
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



def listdirNHF(input_directory):
    """
    creates a list of files in the input directory, avoiding hidden files. Hidden files are identified by the fact that they start with a .
    input: directory of the folder whose elements have to be listed.
    output: list of elements in the input-folder, except hidden files. 
    """
    return [f for f in os.listdir(input_directory) if not f.startswith(".")]


def get_var_value(filename="varstore.dat"):
    """
    Keep track of the number of tmes a script or notebook-cell has been run. NOTE: the function relies on the presence of an exteral file.
    This function taken from https://stackoverflow.com/questions/44012748/how-to-increment-variable-every-time-script-is-run-in-python

    For example, to use the function, call it as "runs_counter = get_var_value(filename="varstore.dat")" and add a "varstore.dat" in the script folder.

    inputs:
    - filename. The full directory of the external file where to store the counting of runs. By default a file named varstore.dat is expected in the
    same folder of the script.

    outputs: Int. The number of times a script or notebook cell has been run.
    """
    with open(filename, "a+") as f:
        f.seek(0)
        val = int(f.read() or 0) + 1
        f.seek(0)
        f.truncate()
        f.write(str(val))
        return val


def check_folder_files_else_make_folder(folder_dir_to_create):
    """
    Check if a folder is present and if it contains any object. If the folder is not present, it creates it.
    inputs:
    - folder_dir_to_create. The directory of a folder to check and, if not present, to create.
    outputs:
    - False if the folder_dir_to_create directory is not present. In addition, it creates it.
    - False if the folder_dir_to_create directory is present and it is empty.
    - True is the folder_dir_to_create directory is present and it is not empty.
    """
    #if the folder does not exist
    if not os.path.exists(folder_dir_to_create):
        os.makedirs(folder_dir_to_create)
        #return False as the folder is not present
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

def minmax_normalization(image, axis=(0,1)):
    """
    returns the min-max normalization of a image. By defalt the image is expected to be bidimensional.
    Inputs:
    - image. n-dimensional np.array. channel dimension in position 0.
    - axis. Int or tuple of ints. Optional. Default (0,1). The axis/axes to use to calculate the min and max values of image.

    Outputs:np.array. min-max normalized image on axis/axes specified by axis. dtype float32.
    """
    #initialize a small variable, to prevent a 0 division
    eps = 1.e-6
    #transform image to 'float32' dtype, for allowing a correct division calculation
    image = image.astype('float32')
    #calculate the minumum of the image using the axis
    image_min = image.min(axis=axis, keepdims=True)
    #subtract the minimum from the image
    image -= image_min
    #calculate the maximum of the image using the axes 1 and 2
    image_max = image.max(axis=axis, keepdims=True)
    #divide the image for the maximum value NOTE: add eps to maximum value to prevent a 0 division
    image /= (image_max + eps)
    return image

def chunk_center(image, chunk_y=256, chunk_x=256):
    """
    divides a 2D image in non-overlapping chunks of size chunk_y * chunk_x. When the image can't be perfectly divided, chunks are centered so that a padding remains
    at the image boarder.

    Inputs:
    - image. 2D numpy array.
    - chunk_y. Int. Default 256. The size, in pixels, of the y axis of each single chunk. It must be <= of the size of the y axis of image.
    - chunk_x. Int. Default 256. The size, in pixels, of the x axis of each single chunk. It must be <= of the size of the x axis of image.

    Outputs: tuple.
    - position 0. 3D numpy array. image chunks are stacked on axis 0.
    - position 1. tuple. Per each chunk, the y,x coordinates of the upper-left pixels are reported.

    NOTE: 1) when image is not perfectly dividable by the chunk dimension along an axis, one extra pixels is left in the padding on the bottom and/or right
    edges.
    2) The chunking process is iterative, it is thus possible to know the order of the chunks in the output. They will proceed by columns, from left to right
    and then by row from top to bottom. Thus the first chunk of the output will be the one on the top-left corner and the last the one on the bottom-right corner.
    """
    #calculate the number of chunks fitting the y axis
    y_chunks_n = image.shape[0]//chunk_y
    #calculate the padding on the y axis
    y_leftover_pixels = image.shape[0]%chunk_y
    if y_leftover_pixels%2==0:
        top_y_padding = int(y_leftover_pixels/2)
        bot_y_padding = int(y_leftover_pixels/2) #this is ultimately not required
    else:
        top_y_padding = int((y_leftover_pixels-1)/2+1)
        bot_y_padding = int((y_leftover_pixels-1)/2) #this is utlimately not required
    
    #calculate the number of chunks fitting the x axis
    x_chunks_n = image.shape[1]//chunk_x
    #calculate the padding on the x axis
    x_leftover_pixels = image.shape[1]%chunk_x
    if x_leftover_pixels%2==0:
        left_x_padding = int(x_leftover_pixels/2)
        right_x_padding = int(x_leftover_pixels/2) #this is ulitmately not required
    else:
        left_x_padding = int((x_leftover_pixels-1)/2+1)
        right_x_padding = int((x_leftover_pixels-1)/2) #this is ultimately not required
    
    assert (y_chunks_n*chunk_y)+top_y_padding+bot_y_padding==image.shape[0], "the pixels splitting did not work on the y axis"
    assert (x_chunks_n*chunk_x)+left_x_padding+right_x_padding==image.shape[1], "the pixels splitting did not work on the x axis"

    #initialize a collection list for the chunks, to be used to form the output array
    chunk_collection_list = []

    #intialize a collection list for the coordinated of the top-left pixels of each chunk, to be used for the output tuple
    coords_collection_list = []

    #initialize the starting position of the highest chunk on the y axis
    y_start = top_y_padding

    #iterate through the number of chunks which could be fit on the y axis
    for y in range(y_chunks_n):

        #initialize the starting position of the chunk on the far left on the x axis
        x_start = left_x_padding

        #iterate through the number of chunks which could be fit on the x axis
        for x in range(x_chunks_n):

            #slice image to obtain the chunck
            chunk = image[y_start:y_start+chunk_y, x_start:x_start+chunk_x]
            
            #add the chunk to the collection list
            chunk_collection_list.append(chunk)

            #add the coordinates to the collection list
            coords_collection_list.append((y_start,x_start))

            #update x_start, so that the following chunk will start from the end of the previous
            x_start = x_start+chunk_x
        
        #update y_start, so that the following chunk will start from the end of the previous
        y_start = y_start+chunk_y
    
    #transform chunk collection list in an array
    chunk_collection_array = np.stack(chunk_collection_list, axis=0)

    #transform coordinates collection list in a tuple
    coords_collection_tuple = tuple(coords_collection_list)

    return chunk_collection_array, coords_collection_tuple


def crop_dimension(image1, dimension2use, min_size):
    # randomize the choice between picking the smaller coordinate first and larger coordinate second or the opposite
    small_first = random.choice([True, False])

    # if small_first coordinate is picked first
    if small_first:

        # randomly pick the small coordinate, making sure that it will not result in a cropped image of size smaller than min_size
        small_coord = random.choice(range(0,image1.shape[dimension2use]-min_size+1))

        # check that small coordinate is not closer to the image size than min_size
        assert small_coord + min_size <= image1.shape[dimension2use]

        # randomly pick the larger coordinate, making sure that it will not result in a cropped image of size smaller than min_size
        if small_coord + min_size == image1.shape[dimension2use]:
            large_coord = image1.shape[dimension2use]
        else:
            large_coord = random.choice(range(small_coord + min_size, image1.shape[dimension2use]+1))
        
    else:
        # randomly pick the large coordinate, making sure that it will not result in a cropped image of size smaller than min_size
        large_coord = random.choice(range(min_size, image1.shape[dimension2use]+1))

        # check that large_coord coordinate is not bigger than the image size
        assert large_coord<=image1.shape[dimension2use]

        # randomly pick the small coordinate, making sure that it will not result in a cropped image of size smaller than min_size
        if large_coord == min_size:
            small_coord = 0
        else:
            small_coord = random.choice(range(0, large_coord - min_size+1))

    # security checks
    assert small_coord>=0
    assert large_coord<=image1.shape[dimension2use]
    assert large_coord-small_coord>=min_size

    return small_coord, large_coord


def random_crop(image, min_y_size=256, min_x_size=256):
    """
    NOTE: limit cases have not been tested for this function.
    """
    # verify that inputs are correct
    assert min_y_size>0, "min_y_size must be > 1"
    assert min_y_size<=image.shape[0], "min_y_size must be <= of input image y dimension"
    assert min_x_size>0, "min_y_size must be >1"
    assert min_x_size<=image.shape[1], "min_x_size must be <= of input image x dimension"

    # retun the image is min_y_size and min_x_size are equal to the sizes of image
    if min_y_size==image.shape[0] and min_x_size==image.shape[1]:
        crop_image = image

    # only crop the y dimension if min_x_size is equal to the size of image
    elif min_y_size!=image.shape[0] and min_x_size==image.shape[1]:
        
        # get the coordinates for the cropping
        top_y_row, bot_y_row = crop_dimension(image1=image, dimension2use=0, min_size=min_y_size)
        
        # crop the image
        crop_image = image[top_y_row:bot_y_row,:]

        # check that crop_image y dimension is not smaller than min_y_size
        assert crop_image.shape[0]>=min_y_size

        # check that crop_image y dimension is not larger than image y size
        assert crop_image.shape[0]<=image.shape[0]

        # check that if top and bot y_row coordinates are at the edge, the image is not cropped
        if top_y_row==0 and bot_y_row==image.shape[0]:
            assert crop_image.shape[0]==image.shape[0]

        # check that if the distance between top and bot coordinates is the min_y_size, the cropped image is of size min_y_size
        if bot_y_row-top_y_row==min_y_size:
            assert crop_image.shape[0]==min_y_size
    
    # only crop the x dimension is min_y_size is equal to the size of image
    elif min_y_size==image.shape[0] and min_x_size!=image.shape[1]:
        
        # get the coordinates for the cropping
        left_x_col , right_x_col = crop_dimension(image1=image, dimension2use=1, min_size=min_x_size)
        
        # crop the image
        crop_image = image[:,left_x_col:right_x_col]

        # check that crop_image x dimension is not smaller than min_x_size
        assert crop_image.shape[1]>=min_x_size

        # check that crop_image x dimension is not larger than image x size
        assert crop_image.shape[1]<=image.shape[1]

        # check that if left and right x_col coordinates are at the edge, the image is not cropped
        if left_x_col==0 and right_x_col==image.shape[1]:
            assert crop_image.shape[1]==image.shape[1]

        # check that if the distance between left and right coordinates is the min_x_size, the cropped image is of size min_x_size
        if right_x_col-left_x_col==min_x_size:
            assert crop_image.shape[1]==min_x_size

    # crop both the first and the second dimensions otherwise
    else:
        
        # get the coordinates for the cropping
        top_y_row, bot_y_row = crop_dimension(image1=image, dimension2use=0, min_size=min_y_size)

        # get the coordinates for the cropping
        left_x_col , right_x_col = crop_dimension(image1=image, dimension2use=1, min_size=min_x_size)

        # crop the image
        crop_image = image[top_y_row:bot_y_row,left_x_col:right_x_col]

        # check that crop_image y dimension is not smaller than min_y_size
        assert crop_image.shape[0]>=min_y_size

        # check that crop_image y dimension is not larger than image y size
        assert crop_image.shape[0]<=image.shape[0]

        # check that crop_image x dimension is not smaller than min_x_size
        assert crop_image.shape[1]>=min_x_size

        # check that crop_image x dimension is not larger than image x size
        assert crop_image.shape[1]<=image.shape[1]

        # check that if top and bot y_row coordinates are at the edge, the image is not cropped
        if top_y_row==0 and bot_y_row==image.shape[0]:
            assert crop_image.shape[0]==image.shape[0]

        # check that if the distance between top and bot coordinates is the min_y_size, the cropped image is of size min_y_size
        if bot_y_row-top_y_row==min_y_size:
            assert crop_image.shape[0]==min_y_size

        # check that if left and right x_col coordinates are at the edge, the image is not cropped
        if left_x_col==0 and right_x_col==image.shape[1]:
            assert crop_image.shape[1]==image.shape[1]

        # check that if the distance between left and right coordinates is the min_x_size, the cropped image is of size min_x_size
        if right_x_col-left_x_col==min_x_size:
            assert crop_image.shape[1]==min_x_size

    return crop_image



def measure_labelled_pixels_fraction(image):
    """
    returns the fraction of pixels which has value >0.

    Inputs:
    - image. n-dimensional array.

    Outputs:
    - float. Fraction of pixels whose value is >0 in image.
    """
    #binarize the image and rescale it the value range 0 and 1.
    bin_image = np.where(image>0, 1,0)
    #calculate the fraction of positive pixels
    pos_px_fraction = np.sum(bin_image)/image.size
    
    assert pos_px_fraction<image.size
    assert np.sum(bin_image)+np.sum(np.where(image>0, 0,1))==image.size

    return pos_px_fraction


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

def crop_spatial_dimensions(x, y, x_dim=[-2,-1], y_dim=[-2,-1]):
    """
    Center-crop x to match spatial dimensions given by y. Note: spatial dimensions the xy image dimension, their positions within x and y shapes can be
    expressed using x_dim and y_dim. By default they are expected in positions -1 and -2.
    It is assumed that no dimension of y has size > than the corresponding dimension in x.
    """

    #get the desired output size by joining all the dimensions of x (input) before the last 2 (before YX)
    #with the YX dimension (the last 2 dimensions of the size) of x (input)
    x_target_size = x.size()[:-2] + y.size()[-2:]

    #per each dimension, get the how many pixels should be added/removed (offset) from x in order to match y.
    #Note: the offset is divided by 2 because it will be added/removed from x to both sides of each dimension.
    offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

    #create a tuple of "slice" objects, with one object per dimension of the inputs.
    #each slice indicates the beginning and end (initial and final indeces) of the part of x which should be kept.
    #NOTE Because each slice starts at the index 'offset' and ends at the index 'offset + size of y' the process effectively guarantees
    #that the central part of x is maintened and matches y.
    slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

    return x[slices]


def dict2mdtable(d, key='Name', val='Value', transform_2_string=True):
    """
    retunrs a dictionary organized as a table but as a string object.
    this code is taken from https://github.com/tensorflow/tensorboard/issues/46

    inputs:
    - d. Dictionary. The dictionary to be transformed in a "table-like" string. NOTE: each key and each value of the d must be a string. If it is not
    or a string, the input transform_2_string must be set to True (default) and they will be transformed to strings. An error is raised
    when at least 1 key or value of d is not a string and transform_2_string is False.
    is not a string
    - key. Str. Optional. Default "Name". The name to give to the "column-like" corresponding to the keys of the dictionary d.
    - val. Str. Optional. Default "Value". The name to give to the "column-like" corresponding to the values of the dictionary d.
    - transform_2_string. Bool. Optional. Default True. If True, all keys and values of d are transformed to strings. If False, keys and values of d are
    expected to be strings or an error is raised.

    outputs: string. The string organised keys and values of the dictionary d as in form of a table. The table-like string has 2 columns and as many rows
    as the number of key-value pairs in the dictionary d. Keys and values of the dictionary d are listed under their respective column-like parts
    of the string under.
    """
    if transform_2_string:
        rows = [f'| {str(key)} | {str(val)} |']
    else:
        rows = [f'| {key} | {val} |']
    # rows += ['|--|--|']
    rows += [f'| {k} | {v} |' for k, v in d.items()]
    return "  \n".join(rows)


def get_current_lr(optimizer):
    """
    returns one of the valid learning rates during model training.

    Inputs:
    - optimizer. Training optimizer. A pytorch object is expected. Refer to torch.optim documentation.

    Outputs: float. Among the valide learning rates, the one in position 0.
    """
    lrs = [param_group.get('lr', None) for param_group in optimizer.param_groups]
    lrs = [lr for lr in lrs if lr is not None]
    return lrs[0]


def save_checkpoint(model, optimizer, n_epoch, path, key="checkpoint"):
    """
    save checkpoints when training model

    Inputs:
    - model. Training model. Must be derived from torch.nn.Module.
    - optimizer. The optimizer of the training process. An object from PyTorch is expected. Refer to https://pytorch.org/docs/stable/optim.html.
    - n_epoch. Int. The training epoch.
    - path. String. The directory of the folder where checkpoints will be saved.
    - key. String. Optional. Default "checkpoint". The name of the checkpoint saved file.

    The function does not have outputs, but checkpoints of trained model are saved in the path directory. Checkpoints are saved as
    .pt objects, refer to torch.save documentation https://pytorch.org/docs/stable/generated/torch.save.html.
    """
    save_path = os.path.join(path, f"{key}.pt")
    torch.save(
        {
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "epoch": n_epoch
        },
        save_path
    )


def load_checkpoint(model, path, optimizer=None, key="checkpoint", map_location=None):
    """
    loads a saved checkpoint of a pytorch model. NOTE: it is expected that the checkpoint has extension .pt () and has been saved using the torch.save function.
    Refer to the documentation https://pytorch.org/docs/stable/generated/torch.save.html.

    Inputs:
    - model. The model whose checkpoint has been saved. Must be derived from torch.nn.Module.
    - path. String. The directory of the folder where the checkpoint has been saved. NOTE: the folder can contain other files in addition to the checkpoin. The
    only file which will be opened is the file whose name corresponds to the "key" input.
    - key. String. Optional. Default "checkpoint". The name of the checkpoint saved file. NOTE: this is the name without extension. Thus, the full name is expected
    be key.pt
    - optimizer. Optional. Default None. The optimizer of the model whose checkpoints have been saved. An object from PyTorch is expected. Refer to https://pytorch.org/docs/stable/optim.html.
    - map_location. Optional. Default None. If None, nothing will be passed to map_location within torch.load. If specified, the parameter will be passed to map_location.

    Outputs:
    if optimizer==None. The model whose checkpoint has been saved.

    If optimizer!=None. The output is a tuple with:
    - position 0. The model whose checkpoint has been saved.
    - position 1. The optimizer of the training process.
    - position 2. The epoch of the saved checkpoint.
    """
    if ".pt" in key:
        load_path = os.path.join(path, f"{key}")
    else:
        load_path = os.path.join(path, f"{key}.pt")
    if map_location==None:
        checkpoint=torch.load(load_path)
    else:
        checkpoint=torch.load(load_path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch=checkpoint["epoch"]
        return model, optimizer, epoch
    
    return model


# def load_tensorboard_data_df(log_dir, tb_variable):
#     """
#     Returns the values of a single scalar variable of a tensorboard event summary register file as a pandas dataframe.
#     This code is adapted from https://stackoverflow.com/questions/42355122/can-i-export-a-tensorflow-summary-to-csv

#     inputs:
#     - log_dir. directory. One of two options can be passed: 1) the directory of the folder containing a single tensorboard summary register file. 2) the
#     directory of the tensorboard summary register file.
#     - tb_variable. String. The name of the scalar variable whose values whould be loaded, as it was entered in the tensorboard summary register file.

#     outputs: Pandas DataFrame with 2 columns: steps and tb_variable and as many rows as the values entered for the tb_variable in the summary register file.
#     """
#     #load events stored in the summary register file 
#     event_accumulator = EventAccumulator(log_dir)
#     event_accumulator.Reload()

#     #select the tb_variable
#     events = event_accumulator.Scalars(tb_variable)

#     # get steps and corresponding values for the tb_variable as numpy arrays
#     x = np.asarray([s.step for s in events])
#     y = np.asarray([v.value for v in events])

#     #form a dataframe with the values
#     df = pd.DataFrame({"step": x, tb_variable: y})
#     return df

def load_tensorboard_data_df(reloaded_event_accumul, tb_variable):
    """
    Returns the values of a single scalar variable of a tensorboard event summary register file as a pandas dataframe.
    This code is adapted from https://stackoverflow.com/questions/42355122/can-i-export-a-tensorflow-summary-to-csv

    inputs:
    - reloaded_event_accumul. The TensorBoard event accumulator object reloaded.
    - tb_variable. String. The name of the scalar variable whose values whould be loaded, as it was entered in the tensorboard summary register file.

    outputs: Pandas DataFrame with 2 columns: steps and tb_variable and as many rows as the values entered for the tb_variable in the summary register file.
    """

    #select the tb_variable
    events = reloaded_event_accumul.Scalars(tb_variable)

    # get steps and corresponding values for the tb_variable as numpy arrays
    x = np.asarray([s.step for s in events])
    y = np.asarray([v.value for v in events])

    #form a dataframe with the values
    df = pd.DataFrame({"step": x, tb_variable: y})
    return df

def interpolate_curve(x,y,n=500, **kwargs):
    X_Y_Spline = make_interp_spline(x, y, **kwargs)
    X_ = np.linspace(x.min(), x.max(), n)
    Y_ = X_Y_Spline(X_)
    return X_,Y_

