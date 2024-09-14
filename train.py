#the code is mostly taken from https://github.com/AlessandroUlivi/The_segmenters/blob/main/source/train.py which
# is derived from the material of the course https://github.com/dl4mia

# import math
import torch
import torch.nn as nn
# import numpy as np
# from utils import crop


def train(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None):
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

    The function has no output.
    """
    #if no device is passed, check if the gpu is available, else use the cpu
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    #get the number of batches in the minibatch
    n_batches = len(loader)

    # # # log the learning rate before the epoch
    # # lr = get_current_lr(optimizer)
    # # tb_logger.add_scalar(tag='learning-rate',
    # #                      scalar_value=lr,
    # #                      global_step=epoch * n_batches)

    # iterate over the batches of the epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x = x.to(device)
        y = y.to(device)

        # zero the gradients for the iteration
        optimizer.zero_grad()

        # apply model and calculate the prediction
        prediction = model(x)

        #THIS MUST BE IMPLEMENTED, BECAUSE IF PADDING IS NOT "same" THE SHAPE IS DIFFERENT
        #THIS MUST BE IMPLEMENTED, BECAUSE IF PADDING IS NOT "same" THE SHAPE IS DIFFERENT
        #THIS MUST BE IMPLEMENTED, BECAUSE IF PADDING IS NOT "same" THE SHAPE IS DIFFERENT
        # if prediction.shape != y.shape:
        #     y = crop(y, prediction)
        # if y.dtype != prediction.dtype:
        #     y = y.type(prediction.dtype)

        #calculate the loss value
        loss = loss_function(prediction[0,0,...], y[0,...])

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()
        print("---", batch_id)

    #     # # print training progression when batch_id is a multiple of log_interval
    #     # if batch_id % log_interval == 0:
    #     #     print(
    #     #         "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
    #     #             epoch,
    #     #             batch_id * len(x),
    #     #             len(loader.dataset),
    #     #             100.0 * batch_id / len(loader),
    #     #             loss.item(),
    #     #         )
    #     #     )

    #     # # log to tensorboard if it is provided
    #     # if tb_logger is not None:
    #     #     step = epoch * len(loader) + batch_id
    #     #     tb_logger.add_scalar(
    #     #         tag="train_loss", scalar_value=loss.item(), global_step=step
    #     #     )
    #     #     # check if we log images in this iteration (when step is a multiple of log_interval)
    #     #     if step % log_image_interval == 0:
    #     #         tb_logger.add_images(
    #     #             tag="input", img_tensor=x.to("cpu"), global_step=step
    #     #         )
    #     #         tb_logger.add_images(
    #     #             tag="target", img_tensor=y.to("cpu"), global_step=step
    #     #         )
    #     #         tb_logger.add_images(
    #     #             tag="prediction",
    #     #             img_tensor=prediction.to("cpu").detach(),
    #     #             global_step=step,
    #     #         )

