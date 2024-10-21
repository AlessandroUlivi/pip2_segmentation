#the code is mostly taken from https://github.com/AlessandroUlivi/The_segmenters/blob/main/source/train.py which
# is derived from the material of the course https://github.com/dl4mia

# import math
import torch
import torch.nn as nn
import numpy as np
from utils import crop_spatial_dimensions, get_current_lr, save_checkpoint
from test_model import test_model
import torch.optim as optim


def train(model,
          loader,
          optimizer,
          loss_function,
          epoch,
          log_interval=100,
          log_image_interval=20,
          tb_logger=None,
          device=None,
          x_dim=[-2,-1],
          y_dim=[-2,-1],
          return_loss_metric=False,
          metric=None,
          bin_threshold=None):
    """
    train the model for 1 epoch. Exploits TensorBoard to keep track of the training progress (variation of the loss function).

    Inputs:
    - model. The model to train. Must be derived from torch.nn.Module.
    - loader. Train data organized in minibatches for the epoch (with inputs and labels). A DataLoader object form torch.utils.data is expected. Refer to https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    - optimizer. The optimizer of the training process. An object from PyTorch is expected. Refer to https://pytorch.org/docs/stable/optim.html
    - loss_function. The loss_function of the training processs. An object from PyTorch is expected. Refer to https://pytorch.org/docs/stable/nn.html#loss-functions
    - epoch. Int. The epoch number within the training process.
    - log_interval. Int. Optional. Default 100. After how many batches the TensorBoard console is updated by saving the loss function,
    in order to track the progression of the training process.
    - log_image_interval. Int. Optional. Default 20. After how many batches the TensorBoard console is updated by saving the prediction results,
    in order to track the progression of the training process.
    - tb_logger. TensorBoard logger. To keep track of progress. Refer to https://www.tensorflow.org/tensorboard?hl=it
    - device. Optional. None or device. Default None. The device to use for the training. If None, it will automatically checked if a cuda gpu is available.
    If available, it will be used. If not available, the cpu will be used.
    - x_dim. List of int. Optional. Default [-2, -1]. The position of Y and X axes in x input image. The parameter is passed to the x_dim input in crop_spatial_dimensions.
    - y_dim. List of int. Optional. Default [-2, -1]. The position of Y and X axes in y input image. The parameter is passed to the y_dim input in crop_spatial_dimensions.
    - return_loss_metric. Bool. Optional. Default False. If True, the function returns the average loss and metric.
    - metric. None, function or class. Optional. Default None. The metric to use for evaluating the results. It must be provided and it is only used if return_loss_metri=True.
    - bin_threshold. None or float. Optional. Default None. The value to use as highpass threshold for binarizing the predicted image before calculating the metric. It must be provided and it is only used if return_loss_metri=True.

    Outputs:
    - if return_loss_metric=False (Default), the function has no output.
    - if return_loss_metric=True, the function returns the average loss value and the average metric value for the data.
    """
    #check that a metric is provided if return_loss_metric is set to True
    if return_loss_metric:
        assert metric!=None, "a metric must be provided if return_loss_metric is set to True"

    #if no device is passed, check if the gpu is available, else use the cpu
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model.to(device)

    # move loss function to device
    loss_function.to(device)

    #get the number of batches in the minibatch
    n_batches = len(loader)

    # log the learning rate before the epoch
    if tb_logger is not None:
        lr = get_current_lr(optimizer)
        tb_logger.add_scalar(tag='learning-rate',
                            scalar_value=lr,
                            global_step=epoch * n_batches)

    # initialize loss and metric values - they are used to obtain the cumulative sum of the loss values and metric values of the data in the loader
    if return_loss_metric:
        cum_loss_val = 0
        cum_metric_val = 0

    # iterate over the batches of the epoch
    for batch_id, (x, y) in enumerate(loader):
        
        # move input and target to the active device (either cpu or gpu)
        x = x.to(device)
        y = y.to(device)

        # zero the gradients for the iteration
        optimizer.zero_grad()

        # apply model and calculate the prediction
        prediction = model(x)

        #crop y when prediction mask is smaller than label (padding is "valid")
        if prediction.shape != y.shape:
            y = crop_spatial_dimensions(y, prediction, x_dim=x_dim, y_dim=y_dim)
        if y.dtype != prediction.dtype:
            y = y.type(prediction.dtype)
        
        #calculate the loss value
        loss = loss_function(prediction, y)

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

        #calculate metric and update cumulative loss and metric values if return_loss_metric is set to True
        if return_loss_metric:
            # calculate the metric value after binarizing the predictions
            binary_prediction = torch.where(prediction>bin_threshold, 1,0)
            metric_val = metric(binary_prediction,y)
            
            # add loss and metric_val to their cumulative respectives (cum_loss_val and cum_metric_val)
            cum_loss_val += loss.item()
            cum_metric_val += metric_val
        
        # print("=== BEFORE UNET ===")
        # print("x shape: ", x.size())
        # print("x dtype: ", x.dtype)
        # print("x max: ", np.amax(x.detach().numpy()))
        # print("x min: ", np.amin(x.detach().numpy()))
        # print("y shape: ", y.size())
        # print("y dtype: ", y.dtype)
        # print("y max: ", np.amax(y.detach().numpy()))
        # print("y min: ", np.amin(y.detach().numpy()))
        # print("pred shape: ", prediction.size())
        # print("pred dtype: ", prediction.dtype)
        # print("pred max: ", np.amax(prediction.detach().numpy()))
        # print("pred min: ", np.amin(prediction.detach().numpy()))

        # print training progression when batch_id is a multiple of log_interval
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader.dataset),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )

        # log to tensorboard if it is provided
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            # check if we log images in this iteration (when step is a multiple of log_interval)
            if step % log_image_interval == 0:
                tb_logger.add_images(
                    tag="input", img_tensor=x.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="target", img_tensor=y.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="prediction",
                    img_tensor=prediction.to("cpu").detach(),
                    global_step=step,
                )

    # get the average loss value and metric for the epoch if return_loss_metric is set to True
    if return_loss_metric:
        avg_loss_val = cum_loss_val / n_batches
        avg_metric_val = cum_metric_val / n_batches

        return avg_loss_val, avg_metric_val


def run_training(model,
                 optimizer,
                 metric, 
                 n_epochs,
                 train_loader,
                 val_loader,
                 loss_function,
                 bin_threshold=0.5,
                 logger=None,
                 log_interval=100,
                 log_image_interval=20,
                 device=None,
                 key="checkpoint",
                 path="",
                 lr_scheduler_flag = False,
                 lr_kwargs={"mode":"min", "factor": 0.1, "patience":2},
                 x_dim=[-2,-1],
                 y_dim=[-2,-1],
                 best_metric_init = 0):
    """
    trains and validate the model over multiple epochs. Exploits TensorBoard to keep track of the training progress (variation of the loss function).

    Inputs:
    - model. The model to train. Must be derived from torch.nn.Module.
    - optimizer. The optimizer of the training process. An object from PyTorch is expected. Refer to https://pytorch.org/docs/stable/optim.html
    - metric. The metric to use to evaluate the training process.
    - n_epochs. Int. The number of epochs to use for the training process.
    - train_loader. Train data organized in minibatches for the epoch (with inputs and labels). A DataLoader object form torch.utils.data is expected. Refer to https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    - val_loader. Validation data organized in minibatches for the epoch (with inputs and labels). A DataLoader object form torch.utils.data is expected. Refer to https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    - loss_function. The loss_function of the training processs. An object from PyTorch is expected. Refer to https://pytorch.org/docs/stable/nn.html#loss-functions
    - bin_threshold. Float. Optional. Default 0.5. The value to use as highpass threshold for binarizing the predicted image before calculating the metric.
    - logger. TensorBoard logger. To keep track of progress. Refer to https://www.tensorflow.org/tensorboard?hl=it
    - log_interval. Int. Optional. Default 100. After how many batches the TensorBoard console is updated by saving the loss function,
    in order to track the progression of the training process.
    - log_image_interval. Int. Optional. Default 20. After how many batches the TensorBoard console is updated by saving the prediction results,
    in order to track the progression of the training process.
    - device. Optional. None or device. Default None. The device to use for the training. If None, it will automatically checked if a cuda gpu is available.
    If available, it will be used. If not available, the cpu will be used.
    - key. String. Optional. Default 'checkpoint'. The key to use for saving the checkpoint.
    - path. String. Optional. Default "" (empty string). If empty string, no checkpoints are saved during the training process. If different than an empty string,
    the provided string will be used as directory to save checkpoints. An error is returned if the directory doesn't exist.
    - lr_scheduler_flag. Bool. Optional. Default False. If True, torch.optim.lr_scheduler.ReduceLROnPlateau will be used as learning rate scheduler. No learning scheduler will be used otherwise.
    - lr_kwargs. Dictionary. Optional. Default {"mode":"min", "factor": 0.1, "patience":2}. The kwargs parameters to be passed to torch.optim.lr_scheduler.ReduceLROnPlateau if lr_scheduler_flag==True.
    - x_dim. List of int. Optional. Default [-2, -1]. The position of Y and X axes in x input image. The parameter is passed to the x_dim input in crop_spatial_dimensions.
    - y_dim. List of int. Optional. Default [-2, -1]. The position of Y and X axes in y input image. The parameter is passed to the y_dim input in crop_spatial_dimensions.
    - best_metric_init. Int or float. Optional. Default 0. The value to use for initializing the best_metric parameter.

    The function has no output.
    """
    
    #if no device is passed, check if the gpu is available, else use the cpu
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # # send model to device
    # model = model.to(device)

    # initialize the best_metric parameter - it will be used for saving checkpoints
    best_metric = best_metric_init

    #initialize the learning scheduler, if specified
    if lr_scheduler_flag:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_kwargs)

    # train for n_epochs. During the training inspect the predictions
    for epoch in range(n_epochs):
        # train the model
        train(model=model,
              loader=train_loader,
              optimizer=optimizer,
              loss_function=loss_function,
              epoch=epoch,
              log_interval=log_interval,
              log_image_interval=log_image_interval,
              tb_logger=logger,
              device=device,
              x_dim=x_dim,
              y_dim=y_dim,
              return_loss_metric=False,
              metric=None,
              bin_threshold=None)

        #calculate the training step
        step = epoch * len(train_loader)

        # test the model
        current_loss, current_metric = test_model(model=model,
                                                  loader=val_loader,
                                                  loss_function=loss_function,
                                                  metric=metric,
                                                  bin_threshold=bin_threshold,
                                                  step=step,
                                                  tb_logger=logger,
                                                  device=device,
                                                  x_dim=x_dim,
                                                  y_dim=y_dim)

        # update the learning scheduler if it is provided
        if lr_scheduler_flag:
            lr_scheduler.step(current_loss)
            logger.add_scalar(tag="lr", scalar_value=lr_scheduler.get_last_lr()[0], global_step=step
            )
        
        #save checkpoint if a path is specified and the metric is the best
        if len(path)>0 and current_metric>best_metric:
            save_checkpoint(model, optimizer, epoch, path, key)


def run_training_no_val(model,
                        optimizer,
                        metric,
                        n_epochs,
                        train_loader,
                        loss_function,
                        bin_threshold=0.5,
                        logger=None,
                        log_interval=100,
                        log_image_interval=20,
                        device=None,
                        key="checkpoint",
                        path="",
                        lr_scheduler_flag = False,
                        lr_kwargs={"mode":"min", "factor": 0.1, "patience":2},
                        x_dim=[-2,-1],
                        y_dim=[-2,-1],
                        best_metric_init = 0):
    
    """
    trains a model over multiple epochs without validation. This function can be used to train a final model using the full data (train data + val data
    + test data).
    Exploits TensorBoard to keep track of the training progress (variation of the loss function).

    NOTE: checkpoints are saved based on the the average matric obtained at each training epoch: if the metric improves a checkpoint is saved. As the
    metric is calculated on the training data, this procedure is at risk of saving a models overfitting the data. Thus this functions must be run on an
    adequate number of epochs, established not to overfit.

    Inputs:
    - model. The model to train. Must be derived from torch.nn.Module.
    - optimizer. The optimizer of the training process. An object from PyTorch is expected. Refer to https://pytorch.org/docs/stable/optim.html
    - metric. The metric to use to evaluate the training process.
    - n_epochs. Int. The number of epochs to use for the training process.
    - train_loader. Train data organized in minibatches for the epoch (with inputs and labels). A DataLoader object form torch.utils.data is expected. Refer to https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    - loss_function. The loss_function of the training processs. An object from PyTorch is expected. Refer to https://pytorch.org/docs/stable/nn.html#loss-functions
    - bin_threshold. Float. Optional. Default 0.5. The value to use as highpass threshold for binarizing the predicted image before calculating the metric.
    - logger. TensorBoard logger. To keep track of progress. Refer to https://www.tensorflow.org/tensorboard?hl=it
    - log_interval. Int. Optional. Default 100. After how many batches the TensorBoard console is updated by saving the loss function,
    in order to track the progression of the training process.
    - log_image_interval. Int. Optional. Default 20. After how many batches the TensorBoard console is updated by saving the prediction results,
    in order to track the progression of the training process.
    - device. Optional. None or device. Default None. The device to use for the training. If None, it will automatically checked if a cuda gpu is available.
    If available, it will be used. If not available, the cpu will be used.
    - key. String. Optional. Default 'checkpoint'. The key to use for saving the checkpoint.
    - path. String. Optional. Default "" (empty string). If empty string, no checkpoints are saved during the training process. If different than an empty string,
    the provided string will be used as directory to save checkpoints. An error is returned if the directory doesn't exist.
    - lr_scheduler_flag. Bool. Optional. Default False. If True, torch.optim.lr_scheduler.ReduceLROnPlateau will be used as learning rate scheduler. No learning scheduler will be used otherwise.
    - lr_kwargs. Dictionary. Optional. Default {"mode":"min", "factor": 0.1, "patience":2}. The kwargs parameters to be passed to torch.optim.lr_scheduler.ReduceLROnPlateau if lr_scheduler_flag==True.
    - x_dim. List of int. Optional. Default [-2, -1]. The position of Y and X axes in x input image. The parameter is passed to the x_dim input in crop_spatial_dimensions.
    - y_dim. List of int. Optional. Default [-2, -1]. The position of Y and X axes in y input image. The parameter is passed to the y_dim input in crop_spatial_dimensions.
    - best_metric_init. Int or float. Optional. Default 0. The value to use for initializing the best_metric parameter.

    The function has no output.
    """
    
    #if no device is passed, check if the gpu is available, else use the cpu
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # # send model to device
    # model = model.to(device)

    #initialize the learning scheduler, if specified
    if lr_scheduler_flag:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_kwargs)

    #initialize the best_metric parameter - it will be used for saving checkpoints
    best_metric = best_metric_init

    # train for n_epochs. During the training inspect the predictions
    for epoch in range(n_epochs):
        # train the model and calculate the average loss and metric for the epoch
        avg_epoch_loss, avg_epoch_metric = train(model=model,
                                                 loader=train_loader,
                                                 optimizer=optimizer,
                                                 loss_function=loss_function,
                                                 epoch=epoch,
                                                 log_interval=log_interval,
                                                 log_image_interval=log_image_interval,
                                                 tb_logger=logger,
                                                 device=device,
                                                 x_dim=x_dim,
                                                 y_dim=y_dim,
                                                 return_loss_metric=True,
                                                 metric=metric,
                                                 bin_threshold=bin_threshold)

        # print(
        # "\nAverage loss: {:.4f}, Average Metric: {:.4f}\n".format(
        #     avg_epoch_loss, avg_epoch_metric
        # )
        # )

        #calculate the training step
        step = epoch * len(train_loader)

        # update the learning scheduler if it is provided
        if lr_scheduler_flag:
            lr_scheduler.step(avg_epoch_loss)
            logger.add_scalar(tag="lr", scalar_value=lr_scheduler.get_last_lr()[0], global_step=step
            )
        
        # log to tensorboard if it is provided
        if logger is not None:
            step = epoch * len(train_loader)
            logger.add_scalar(
                tag="train_avg_metric", scalar_value=avg_epoch_metric, global_step=step+len(train_loader)
            )

        #save checkpoint if a path is specified and the metric is the best
        if len(path)>0 and avg_epoch_metric>best_metric:
            save_checkpoint(model, optimizer, epoch, path, key)


