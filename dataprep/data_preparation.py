# the present functions are adapted from https://github.com/dl4mia/DL4MIA_Pre-course_Webinar/blob/main/notebooks/utils.py

import os
from functools import partial
import random
import numpy as np
import torch
from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from scipy.ndimage import shift as shf
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


# def make_dataset_train_val_split(images, labels, validation_fraction=0.20, shuffle_data=True):
#     """
#     splits inputs data and matching labels into train and validation sub-sets.

#     Inputs:
#     - images. np.array.
#     - labels. np.array. The size of labels axis 0 (shape[0]) must match the size of images axis 0 (shape[0]).
#     - validation_fraction. float. Optional. Defauls 0.2. The fraction of data to use as validation dataset.
#     - shuffle_data. Bool. Optional. Defaul True. Whether or not to shuffle data before splitting.

#     Outputs: tuple.
#     - position 0. np.array. Sub-array of images along axis 0. The size of axis 0 corresponds to 1-validation_fraction of the original axis 0 size.
#     - position 1. np.array. Sub-array of images along axis 0. The size of axis 0 corresponds to validation_fraction of the original axis 0 size.
#     - position 2. np.array. Sub-array of labels along axis 0. The size of axis 0 corresponds to 1-validation_fraction of the original axis 0 size.
#     - position 3. np.array. Sub-array of labels along axis 0. The size of axis 0 corresponds to validation_fraction of the original axis 0 size.

#     """
#     #splits images and labels in train and validation datasets
#     (train_images, val_images,
#      train_labels, val_labels) = train_test_split(images, labels, shuffle=shuffle_data,
#                                                   test_size=validation_fraction)
#     #check that the splitting was correctly done
#     assert len(train_images) == len(train_labels)
#     assert len(val_images) == len(val_labels)
#     assert len(train_images) + len(val_images) == len(images)

#     return train_images, train_labels, val_images, val_labels


def add_channel(image, target, axis_to_use=0):
    """
    Add an extra dimension to images at position axis_to_use.
    NOTE: Although the position can be changed, this is meant to be done so that it could be interpreted as the color channel dimension by PyTorch. Final output (CWH)
    
    Inputs:
    - image. np.array.
    - target. np.array.
    - axis_to_use. Int. Optional. Defaul 0. The position in image.shape and target.shape where the extra dimension is added.

    Outputs: tuple.
    - position 0. np.array. Image with extra dimension of size 1 in position axis_to_use.
    - position 1. target. Target with extra dimension of size 1 in position axis_to_use.

    """
    # put channel first
    image_w_c = np.expand_dims(image, axis=axis_to_use)
    target_w_c = np.expand_dims(target, axis=axis_to_use)
    return image_w_c, target_w_c

def add_channel_and_batch(image, axis_to_use=0):
    """
    Add two extra dimension to images at position axis_to_use.
    NOTE: Although the position can be changed, this is meant to be done so that it could be interpreted as the color channel and batch dimensions by PyTorch. Final output (BCWH)
    
    Inputs:
    - image. np.array.
    - axis_to_use. Int. Optional. Defaul 0. The position in image.shape where the extra dimensions are added.

    Outputs: np.array. Image with extra dimensions of size 1 in position axis_to_use.

    """
    # put channel first
    image_w_c = np.expand_dims(image, axis=axis_to_use)
    image_w_b = np.expand_dims(image_w_c, axis=axis_to_use)
    return image_w_b

def normalize(image, target):
    """
    returns the min-max normalization of a image.
    Inputs:
    - image. 3D np.array. channel dimension in position 0.
    - target. np.array.

    Outputs: tuple.
    - position 0. np.array. min-max normalized image on axes 1 and 2. dtype float32.
    - position 1. np.array. target image rescaled to range 0 and 1. dtype float32
    """
    #initialize a small variable, to prevent a 0 division
    eps = 1.e-6
    #transform image and target to 'float32' dtype, for allowing a correct division calculation
    image = image.astype('float32')
    target = target.astype('float32')
    #calculate the minumum of the image using the axes 1 and 2 and the minimum of the target using the axis 0 and 2.
    chan_min = image.min(axis=(1, 2), keepdims=True)
    #subtract the minimum from the image
    image -= chan_min
    #calculate the maximum of the image using the axes 1 and 2
    chan_max = image.max(axis=(1, 2), keepdims=True)
    #divide the image for the maximum value NOTE: add eps to maximum value to prevent a 0 division
    image /= (chan_max + eps)
    #rescale target image to range 0 and 1
    rescaled_target = np.where(target>0, 1.0, 0.0).astype('float32')
    return image, rescaled_target


def to_tensor(image, target):
    """
    transfom image-target pair of numpy array to a pair of tensors
    Inputs:
    - image. np.array. channel dimension in position 0.
    - target. np.array.

    Outputs: tuple.
    - position 0. image transformed to tensor
    - position 1. target transformed to tensor

    """
    return torch.from_numpy(image.copy()), torch.from_numpy(target.copy())
    # return torch.from_numpy(image), torch.from_numpy(target)
    # return torch.from_numpy(image), torch.tensor([target], dtype=torch.int64)

def random_flip(image, target):
    """
    randomly flips both the image and the target along an axis. Whether or not to flip the image is chosen randomly.
    The axis to flip is chosen randomly. When the trasformation is applied, it is always applied, identical, to both image and target.
    The transformation is applied 1 time out of 2.

    inputs:
    - image. 2D array.
    - target. 2D array.

    outputs: tuple.
    - position 0. image. 1 time out of 2 it is has been flipped either on the vertical or the orizontal axis.
    - position 1. target. 1 time out of 2 it is has been flipped either on the vertical or the orizontal axis.
    """
    #randomly pick a number between 0 and 1
    coin = random.choice([0,1])
    #if the number is 1, apply the flip tranformation
    if coin==1:
        #randomly pick the axis along which to apply the transformation
        random_axis = random.choice([0,1])
        #flip image
        flipped_image = np.flip(image, axis=random_axis)
        #flip target
        flipped_target = np.flip(target, axis=random_axis)
        return flipped_image, flipped_target
    else:
        return image, target

def random_gaussian_noise(image, target):
    """
    adds some gaussian noise 1 time out of 6 to image. Target is returned unmodified.

     inputs:
    - image. n-dimensional array. Values are expected in the range 0-1.
    - target. n-dimensional array.

    outputs: tuple.
    - position 0. image. 1 time out of 6 gaussian noise is added.
    - position 1. target. Unmodified.
    """
    #randomly pick a number between 0 and 5
    dice = random.choice([0,1,2,3,4,5])
    #if the number is 1
    if dice==1:
        #create an array of the same shape of input but with values randomly drawn from a normal distribution centered on 0.5 and with a standard deviation of 0.1
        gaussian = np.random.normal(loc=0.5, scale=0.1, size=image.shape)
        #add noise to image
        noise_image = image+gaussian
        #rescale the values in the 0-1 range
        rescaled_noise_image = minmax_scale(noise_image.ravel(), feature_range=(0.0,1.0)).reshape(noise_image.shape).astype(image.dtype)
        return rescaled_noise_image, target
    else:
        return image, target

def random_uniform_noise(image, target):
    """
    adds some uniform noise 1 time out of 6 to image. Target is returned unmodified.

     inputs:
    - image. 2D array. Values are expected in the range 0-1.
    - target. n-dimensional array.

    outputs: tuple.
    - position 0. image. 1 time out of 6 uniform noise is added.
    - position 1. target. Unmodified.
    """
    #randomly pick a number between 0 and 5
    dice = random.choice([0,1,2,3,4,5])
    #if the number is 1
    if dice==1:
        #create an array of the same shape of input but with values randomly drawn from a in the 0-1 range
        uniform_noise = np.random.rand(image.shape[0],image.shape[1])
        #rescale uniform noise values in the 0.25-0.75 range
        rescaled_uniform_noise = minmax_scale(uniform_noise.ravel(), feature_range=(0.25,0.75)).reshape(image.shape)
        #add noise to image
        noise_image = image+rescaled_uniform_noise
        #rescale the values in the 0-1 range
        rescaled_noise_image = minmax_scale(noise_image.ravel(), feature_range=(0.0,1.0)).reshape(noise_image.shape).astype(image.dtype)
        return rescaled_noise_image, target
    else:
        return image, target

def random_gaussian_or_uniform_noise(image, target):
    """
    adds some noise 1 time out of 6 to image. When noise is added, 50% of the time it is gaussian noise, 50% of the times is uniform noise.
    Target is returned unmodified.

     inputs:
    - image. 2D array. Values are expected in the range 0-1.
    - target. n-dimensional array.

    outputs: tuple.
    - position 0. image. 1 time out of 6 noise is added.
    - position 1. target. Unmodified.
    """
    #randomly pick a number between 0 and 5
    dice = random.choice([0,1,2,3,4,5])
    #if the number is 1
    if dice==1:
        #randomly pick a number between 0 and 1
        coin = random.choice([0,1])
        #if 1 is picked
        if coin==1:
            #create an array of the same shape of input but with values randomly drawn from a normal distribution centered on 128 and with a standard deviation o 20
            gaussian = np.random.normal(loc=0.5, scale=0.1, size=image.shape)
            #add noise to image
            noise_image = image+gaussian
            #rescale the values in the 0-1 range
            rescaled_noise_image = minmax_scale(noise_image.ravel(), feature_range=(0.0,1.0)).reshape(noise_image.shape).astype(image.dtype)
        else:
            #create an array of the same shape of input but with values randomly drawn from a in the 0-1 range 
            uniform_noise = np.random.rand(image.shape[0],image.shape[1])
            #rescale values in the 0.25-0.75 range
            rescaled_uniform_noise = minmax_scale(uniform_noise.ravel(), feature_range=(0.25,0.75)).reshape(image.shape)
            #add noise to image
            noise_image = image+rescaled_uniform_noise
            #rescale the values in the 0-1 range
            rescaled_noise_image = minmax_scale(noise_image.ravel(), feature_range=(0.0,1.0)).reshape(noise_image.shape).astype(image.dtype)
        return rescaled_noise_image, target
    else:
        return image, target


def random_translation(image, target):
    """
    applies a random translation to image and target a little less than 1 time out of 6. Per each dimension of image, the maximum possible translation is
    half the dimension size. The translation is only applied if the target image, after the translation, still has at least 1 labelled pixel, for this reason,
    it is not possible to establish that it is applied exactely 1 time out of 6.
    When the translation is applied, it is applied identical to both image and target. Image and target are returned with the same shape,
    for this reason, a padding procedure is also applied. Pad values are 0.

    inputs:
    - image. n-dimensional array. Note that the function has only been tested on 2D arrays.
    - target. n-dimensional array of same shape of image. Labelled pixels are expected to have value >=1. Background pixels are expected to have value 0.

    outputs: tuple.
    - position 0. image. A little less than 1 time out of 6, input image has been translated.
    - position 1. target. A little less than 1 time out of 6, input target has been translated.
    """
    #randomly pick a number between 0 and 1
    dice = random.choice([0,1,2,3,4,5])
    #if the number is 1
    if dice == 1:
        #initialize a collection list for the translation of each image (and target) dimension
        translation2apply = []
        #iterate through the dimension of image
        for d in image.shape:
            #pick a random number in a range between 0 and half the dimension
            random_translation = float(random.choice(range(-d//2, d//2)))
            #add the randomly picked number to the collection list
            translation2apply.append(random_translation)
        #transform the collection list to an array
        translation2apply = np.asarray(translation2apply)
        #shift the target image
        translated_target = shf(target, translation2apply)
        #if the resulting translated target contains at least a label pixel
        if np.sum(translated_target)>=1:
            #translate also image
            translated_image = shf(image, translation2apply)
            #ensure that translated_target values are remain the same of the input image after translation
            translated_target_within_input_range = np.where(translated_target>=1.0,translated_target, 0.0)
            return translated_image, translated_target_within_input_range
        #Don't apply the translation if the translated target does not contain any labelled pixel
        else:
            return image, target
    else:
        return image, target


def compose(image, target, transforms):
    """
    applies a series of functions, sequentially, to an image-target pair.

    Inputs:
    - image. np.array.
    - target. np.array.
    - transforms. iterable. list-like object with the funtions to apply. Functions will be applied, sequentially from position 0 to position -1.

    Outputs: tuple.
    - position 0. tranformed image.
    - position 1. transformed target.
    """
    #iterate through the functions of transforms
    for trafo in transforms:
        #apply function to image and target
        image, target = trafo(image, target)
    return image, target


class DatasetWithTransform(Dataset): #Dataset (from pytorch.utils.data) is sub-classed. This allows to inherit properties from the Dataset module
    """
    Minimal dataset class. It holds data and target
    as well as optional transforms that are applied to data and target data is requested.
    """

    #initialize variables in the class
    def __init__(self, data, target, transform=None):
        assert isinstance(data, np.ndarray)
        assert isinstance(target, np.ndarray)
        self.data = data
        self.target = target
        if transform is not None:
            assert callable(transform)
        self.transform = transform

    # gets the image-target pair at index=index in the dataset. Applies transformations if required
    def __getitem__(self, index):
        data, target = self.data[index], self.target[index]
        if self.transform is not None:
            data, target = self.transform(data, target)
        return data, target

    #get the length of the dataset
    def __len__(self):
        return self.data.shape[0]


def get_default_transform():
    """
    Returns a series of the minimal transformations to images which should be done for preparing the dataset for being passed to PyTorch.
    Functions application is chained by exploiting the partial function, thus allowing to share inputs across the transformations.
    This is also possible as the transformations share the same inputs.

    The minimal transformations are, sequentially, addition of the channel dimension to data, in the position 0. The min-max normalization of the
    data. The transformation of the data to tensors.
    """
    trafos = [add_channel, normalize, to_tensor]
    trafos = partial(compose, transforms=trafos)
    return trafos


# def make_train_datasets(input_data_dir, labels_data_dir, transform=None, validation_fraction=0.20, stack_axis=0, shuffle_data=True):
#     """
#     Loads the train dataset. Splits train and validation sub-datasets. Applies tranformations, if required.

#     Inputs:
#     - input_data_dir; the directory of the input data. It is expected that only the files to be used as input data are found in the directory. Input data must
#     have the same shape.
#     - labels_data_dir; the directory of labels data. It is expected that only the files to be used as input data are found in the directory. The names of the labels
#     data is expected to match the name of the corresponding input data.
#     - transform. None or iterable. Default None. If None, minimal data transformations to have data prepared for PyTorch, will be applied
#     (see get_default_transform). If iterable, a list-like object must be passed, containing the funtions to apply. Functions will be applied,
#     sequentially from position 0 to position -1. Note that if different than None, the minimal data transformations (see get_default_transform) will be ignored.
#     - validation_fraction. float. Optional. Defauls 0.2. The fraction of data to use as validation dataset.
#     - stack_axis. Optional. Defauls 0. The axis along which data are stacked in the output arrays.
#     - shuffle_data. Bool. Optional. Defaul True. Whether or not to shuffle data before splitting.

#     Outputs: tuple.
#     - position 0. Dataset class from  DatasetWithTransform(Dataset) for the training sub-dataset (input and corresponding labels).
#     - position 1. Dataset class from  DatasetWithTransform(Dataset) for the validation sub-dataset.
#     """
#     #load input data and corresponding labels
#     images, labels = load_dataset(input_data_dir, labels_data_dir, stack_axis=stack_axis)
    
#     #split data in tran and validation datasets
#     (train_images, train_labels,
#      val_images, val_labels) = make_dataset_train_val_split(images, labels, validation_fraction=validation_fraction, shuffle_data=shuffle_data)

#     #get default the minimum transformations to apply to the dataset if tranforms is set to None
#     if transform is None:
#         transform = get_default_transform()
    
#     #Instantiate the train and validation dataset classes 
#     train_dataset = DatasetWithTransform(train_images, train_labels, transform=transform)
#     val_dataset = DatasetWithTransform(val_images, val_labels, transform=transform)

#     return train_dataset, val_dataset


# def make_test_dataset(input_data_dir, labels_data_dir, transform=None, stack_axis=0):
#     """
#     Loads the test dataset. Applies tranformations, if required.

#     Inputs:
#     - input_data_dir; the directory of the input data. It is expected that only the files to be used as input data are found in the directory. Input data must
#     have the same shape.
#     - labels_data_dir; the directory of labels data. It is expected that only the files to be used as input data are found in the directory. The names of the labels
#     data is expected to match the name of the corresponding input data.
#     - transform. None or iterable. Default None. If None, minimal data transformations to have data prepared for PyTorch, will be applied
#     (see get_default_transform). If iterable, a list-like object must be passed, containing the funtions to apply. Functions will be applied,
#     sequentially from position 0 to position -1.
#     - stack_axis. Optional. Defauls 0. The axis along which data are stacked in the output arrays.

#     Outputs: Dataset class from DatasetWithTransform(Dataset) for the input_data and corresponding labels.
#     """
#     #load input data and corresponding labels
#     images, labels = load_dataset(input_data_dir, labels_data_dir, stack_axis=stack_axis)

#     #get default the minimum transformations to apply to the dataset if tranforms is set to None
#     if transform is None:
#         transform = get_default_transform()
    
#     #Instantiate the test dataset class
#     dataset = DatasetWithTransform(images, labels, transform=transform)

#     return dataset


def make_dataset(input_data_dir, labels_data_dir, transform=None, shuffle_data=True, stack_axis=0):
    """
    Loads the a dataset and returns it as a Dataset object (sub-classed from PyTorch, inherited from DatasetWithTransform).
    Applies tranformations, if required.

    Inputs:
    - input_data_dir. The directory of the input data. It is expected that only the files to be used as input data are found in the directory. Input data must
    have the same shape.
    - labels_data_dir. The directory of labels data. It is expected that only the files to be used as input data are found in the directory. The names of the labels
    data is expected to match the name of the corresponding input data.
    - transform. None or iterable. Default None. If None, minimal data transformations to have data prepared for PyTorch, will be applied
    (see get_default_transform). If iterable, a list-like object must be passed, containing the funtions to apply. Functions will be applied,
    sequentially from position 0 to position -1.
    - shuffle_data. Bool. Optional. Default True. Whether or not to shuffle the order of the data after their loading.
    - stack_axis. Optional. Defauls 0. The axis along which data are stacked in the output arrays.

    Outputs: Dataset class from DatasetWithTransform(Dataset) for the input_data and corresponding labels.
    """
    #load input data and corresponding labels
    images, labels = load_dataset(input_data_dir, labels_data_dir, stack_axis=stack_axis)

    #implement shuffling of the data, if shuffle_data=True
    if shuffle_data:
        #generate a random order of the indexes of the images
        random_index_shuffle = random.sample(range(images.shape[stack_axis]), k=images.shape[stack_axis])

        #reorder images according to the random order
        images = images[np.asarray(random_index_shuffle), ...]

        #reorder the labels according the same random order
        labels = labels[np.asarray(random_index_shuffle), ...]

    #get default the minimum transformations to apply to the dataset if tranforms is set to None
    if transform is None:
        transform = get_default_transform()
    
    #Instantiate the test dataset class
    dataset = DatasetWithTransform(images, labels, transform=transform)

    return dataset

