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
    return torch.from_numpy(image), torch.from_numpy(target)
    # return torch.from_numpy(image), torch.tensor([target], dtype=torch.int64)

def random_flip(image, target):
    """
    randomly flips both the image and the target along a axis. Whether or not to flip the image is chosen randomly,
    the axis to flip is chosen randomly. The same transformation is applied to both image and target
    """
    coin = random.choice([0,1])
    if coin==1:
        random_axis = random.choice([0,1])
        flipped_image = np.flip(image, axis=random_axis)
        flipped_target = np.flip(target, axis=random_axis)
        return flipped_image, flipped_target
    else:
        return image, target

def random_gaussian_noise(image, target):
    """
    adds some random noise 1 time out of 6.
    """
    dice = random.choice([0,1,2,3,4,5])
    if dice==1:
        gaussian = np.random.normal(loc=128, scale=20, size=(image.shape[0],image.shape[1]))
        noise_image = image+gaussian
        return noise_image, target
    else:
        return image, target

def random_uniform_noise(image, target):
    """
    adds some random noise 1 time out of 6.
    """
    dice = random.choice([0,1,2,3,4,5])
    if dice==1:
        uniform_noise = np.random.rand(image.shape[1],image.shape[2])
        rescaled_uniform_noise = minmax_scale(uniform_noise.ravel(), feature_range=(0,255)).reshape(image.shape)
        noise_image = image+rescaled_uniform_noise
        return noise_image, target
    else:
        return image, target

def random_translation(image, target):
    """
    applies a random translation to image and target 1 time out of 6. Per each dimension of image, the maximum possible translation is
    half the dimension size. The translation is only applied if the target image, after the translation, still has some labelled pixels.
    """
    dice = random.choice([0,1,2,3,4,5])
    if dice == 1:
        translation2apply = []
        for d in image.shape:
            random_translation = float(random.choice(range(d//2)))
            translation2apply.append(random_translation)
        translation2apply = np.asarray(translation2apply)
        translated_target = shf(target, translation2apply)
        if np.sum(translated_target)>0:
            translated_image = shf(image, translation2apply)
            return translated_image, translated_target
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

