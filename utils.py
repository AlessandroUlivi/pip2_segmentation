import os
import numpy as np
import random
import torch


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

def chunk_center(image, chunk_y=256, chunk_x=256):
    """
    divides a 2D image in chunks of size chunk_y * chunk_x. When the image can't be perfectly divided, chunks are centered so that a padding remains
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


def load_checkpoint(model, path, optimizer=None, key="checkpoint"):
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

    Outputs:
    if optimizer==None. The model whose checkpoint has been saved.

    If optimizer!=None. The output is a tuple with:
    - position 0.  The model whose checkpoint has been saved.
    - position 1. The optimizer of the training process.
    - position 2. The epoch of the saved checkpoint.
    """
    load_path = os.path.join(path, f"{key}.pt")
    checkpoint=torch.load(load_path)
    model.load_state_dict(checkpoint["model"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch=checkpoint["epoch"]
        return model, optimizer, epoch
    
    return model
