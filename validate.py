#the code is mostly taken from https://github.com/AlessandroUlivi/The_segmenters/blob/main/source/validation.py which
# is derived from the material of the course https://github.com/dl4mia

# import math
import torch
import torch.nn as nn
# import numpy as np


def validate(
    model,
    loader,
    loss_function,
    metric,
    step=None,
    tb_logger=None,
    device=None):

    """
    validate the model on one epoch. The function is meant to run on the validation dataset during the model training.
    It exploits TensorBoard to keep track of the training progress (variation of the loss function).

    Inputs:
    - model. The model to validate. Must be derived from torch.nn.Module.
    - loader. Validation data organized in minibatches for the epoch (with inputs and labels). A DataLoader object form torch.utils.data is expected. Refer to https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    - loss_function. The loss_function of the training processs. An object from PyTorch is expected. Refer to https://pytorch.org/docs/stable/nn.html#loss-functions
    - metric. Function or class. The metric to use for evaluating the results.
    - log_image_interval. Int. Optional. Default 20. After how many batches the TensorBoard console is updated by saving the prediction results,
    in order to track the progression of the training process.
    - tb_logger. TensorBoard logger. To keep track of progress. Refer to https://www.tensorflow.org/tensorboard?hl=it
    - device. Optional. None or device. Default None. The device to use for the validation. If None, it will automatically checked if a cuda gpu is available.
    If available, it will be used. If not available, the cpu will be used.

    Outputs: float. The average loss value for the validation data.
    """

    #if no device is passed, check if the gpu is available, else use the cpu
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set model to eval mode
    model.eval()

    # move model to device
    model.to(device)

    # initialize loss and metric values - they are used to obtain the cumulative sum of the loss values and metric values of the data in the loader
    cum_loss_val = 0
    cum_metric_val = 0

    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            # move input and target to the active device (either cpu or gpu)
            x = x.to(device)
            y = y.to(device)

            # apply model and calculate the prediction
            prediction = model(x)
            
            # calculate the loss value and the metric value
            loss_val = loss_function(prediction,y)
            metric_val = metric(prediction,y)
            
            # add loss_val and metric_val to their cumulative respectives (cum_loss_val and cum_metric_val)
            cum_loss_val += loss_val
            cum_metric_val += metric_val

    # get the average loss value and metric
    avg_loss_val = cum_loss_val / len(loader)
    avg_metric_val = cum_metric_val / len(loader)

    # # log the validation results if we have a tensorboard
    # if tb_logger is not None:
    #     assert (
    #         step is not None
    #     ), "Need to know the current step to log validation results"
    #     tb_logger.add_scalar(tag="val_loss", scalar_value=avg_loss_val, global_step=step)
    #     tb_logger.add_scalar(
    #         tag="val_metric", scalar_value=avg_metric_val, global_step=step
    #     )
    #     # we always log the last validation images
    #     tb_logger.add_images(tag="val_input", img_tensor=x.to("cpu"), global_step=step)
    #     tb_logger.add_images(tag="val_target", img_tensor=y.to("cpu"), global_step=step)
    #     tb_logger.add_images(
    #         tag="val_prediction", img_tensor=prediction.to("cpu"), global_step=step
    #     )

    # print(
    #     "\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n".format(
    #         avg_loss_val, avg_metric_val
    #     )
    # )

    return avg_loss_val

