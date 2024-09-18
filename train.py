#the code is mostly taken from https://github.com/AlessandroUlivi/The_segmenters/blob/main/source/train.py which
# is derived from the material of the course https://github.com/dl4mia

# import math
import torch
import torch.nn as nn
import numpy as np
from utils import crop_spatial_dimensions, get_current_lr
from validate import validate
import torch.optim as optim


def train(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None,
    x_dim=[-2,-1],
    y_dim=[-2,-1]):
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


# def run_training(model,
#                  optimizer,
#                  metric, 
#                  n_epochs,
#                  train_loader,
#                  val_loader,
#                  loss_function,
#                  logger=None,
#                  log_interval=100,
#                  device=None,
#                  key="checkpoint",
#                  path="",
#                  lr_scheduler_flag = False,
#                  lr_kwargs={"mode":"min", "factor": 0.1, "patience":2}),
#                  x_dim=[-2,-1],
#                  y_dim=[-2,-1]:
#     """
#     trains and validate the model over multiple epochs. Exploits TensorBoard to keep track of the training progress (variation of the loss function).

#     Inputs:
#     - model. The model to train. Must be derived from torch.nn.Module.
#     - optimizer. The optimizer of the training process. An object from PyTorch is expected. Refer to https://pytorch.org/docs/stable/optim.html
#     - metric. The metric to use to evaluate the training process.
#     - n_epochs. Int. The number of epochs to use for the training process.
#     - train_loader. Train data organized in minibatches for the epoch (with inputs and labels). A DataLoader object form torch.utils.data is expected. Refer to https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
#     - val_loader. Validation data organized in minibatches for the epoch (with inputs and labels). A DataLoader object form torch.utils.data is expected. Refer to https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
#     - loss_function. The loss_function of the training processs. An object from PyTorch is expected. Refer to https://pytorch.org/docs/stable/nn.html#loss-functions
#     - logger. TensorBoard logger. To keep track of progress. Refer to https://www.tensorflow.org/tensorboard?hl=it
#     - log_interval. Int. Optional. Default 100. After how many batches the TensorBoard console is updated by saving the loss function,
#     in order to track the progression of the training process.
#     - device. Optional. None or device. Default None. The device to use for the training. If None, it will automatically checked if a cuda gpu is available.
#     If available, it will be used. If not available, the cpu will be used.
#     - key. String. Optional. Default 'checkpoint'. The key to use for saving the checkpoint.
#     - path. String. Optional. Default "" (empty string). If empty string, no checkpoints are saved during the training process. If different than an empty string,
#     the provided string will be used as directory to save checkpoints. An error is returned if the directory doesn't exist.
#     - lr_scheduler_flag. Bool. Optional. Default False. If True, torch.optim.lr_scheduler.ReduceLROnPlateau will be used as learning rate scheduler. No learning scheduler will be used otherwise.
#     - lr_kwargs. Dictionary. Optional. Default {"mode":"min", "factor": 0.1, "patience":2}. The kwargs parameters to be passed to torch.optim.lr_scheduler.ReduceLROnPlateau if lr_scheduler_flag==True.
#     - x_dim. List of int. Optional. Default [-2, -1]. The position of Y and X axes in x input image. The parameter is passed to the x_dim input in crop_spatial_dimensions.
#     - y_dim. List of int. Optional. Default [-2, -1]. The position of Y and X axes in y input image. The parameter is passed to the y_dim input in crop_spatial_dimensions.

#     The function has no output.
#     """
    
#     #if no device is passed, check if the gpu is available, else use the cpu
#     if device is None:
#         if torch.cuda.is_available():
#             device = torch.device("cuda")
#         else:
#             device = torch.device("cpu")

#     # # send model to device
#     # model = model.to(device)

#     # #initialize the learning scheduler, if specified
#     # if lr_scheduler_flag:
#     #     lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_kwargs)

#     # train for n_epochs. During the training inspect the predictions
#     for epoch in range(n_epochs):
#         # train the model
#         train(
#             model=model,
#             loader=train_loader,
#             optimizer=optimizer,
#             loss_function=loss_function,
#             epoch=epoch,
#             log_interval=log_interval,
#             tb_logger=logger,
#             device=device,
#             x_dim=x_dim,
#             y_dim=y_dim
#         )

#         #calculate the training step
#         step = epoch * len(train_loader)

#         # validate the model
#         current_loss = validate(
#                                 model=model,
#                                 loader=val_loader,
#                                 loss_function=loss_function,
#                                 metric=metric,
#                                 step=step,
#                                 tb_logger=logger,
#                                 device=device,
#                                 x_dim=x_dim,
#                                 y_dim=y_dim)

#         # if lr_scheduler_flag:
#         #     lr_scheduler.step(current_loss)
#         #     logger.add_scalar(tag="lr", scalar_value=lr_scheduler.get_last_lr()[0], global_step=step
#         #     )
        
#         #save checkpoint if a path is specified
#         if len(path)>0:
#             save_checkpoint(model, optimizer, epoch, path, key)


# def run_cifar_training(model, optimizer,
#                        train_loader, val_loader,
#                        device, name, n_epochs):
#     """ Complete training logic
#     """

#     best_accuracy = 0.

#     loss_function = nn.NLLLoss()
#     loss_function.to(device)

#     scheduler = ReduceLROnPlateau(optimizer,
#                                   mode='max',
#                                   factor=0.5,
#                                   patience=1)

#     checkpoint_path = f'best_checkpoint_{name}.tar'
#     log_dir = f'runs/{name}'
#     tb_logger = SummaryWriter(log_dir)

#     for epoch in trange(n_epochs):
#         train(model, train_loader, loss_function, optimizer,
#               device, epoch, tb_logger=tb_logger)
#         step = (epoch + 1) * len(train_loader)

#         pred, labels = validate(model, val_loader, loss_function,
#                                 device, step,
#                                 tb_logger=tb_logger)
#         val_accuracy = metrics.accuracy_score(labels, pred)
#         scheduler.step(val_accuracy)

#         # otherwise, check if this is our best epoch
#         if val_accuracy > best_accuracy:
#             # if it is, save this check point
#             best_accuracy = val_accuracy
#             save_checkpoint(model, optimizer, epoch, checkpoint_path)

#     return checkpoint_path
