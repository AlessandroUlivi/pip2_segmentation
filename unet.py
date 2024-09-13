# the present functions are adapted from https://github.com/AlessandroUlivi/The_segmenters/blob/main/source/unet.py
# they are based on https://github.com/dl4mia/01_segmentation

import math
import torch
import torch.nn as nn
import numpy as np


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str = "same",
    ):
        """ A 2D convolution block for a U-Net. Contains two convolutions, each followed by a ReLU.

        Args:
            in_channels (int): The number of input channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            out_channels (int): The number of output channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            kernel_size (int): The size of the kernel. A kernel size of N signifies an
                NxN square kernel.
            padding (str, optional): The type of padding to use. Options are "same" or "valid".
                Defaults to "same".
        """
        #call the bound __init__ from the parent class (torch.nn.Module) that follows the child class (ConvBlock).
        #refer to https://stackoverflow.com/questions/222877/what-does-super-do-in-python-difference-between-super-init-and-expl
        #refer to https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods
        super().__init__()

        # determine padding size based on method
        if padding in ("VALID", "valid"):
            pad = 0
        elif padding in ("SAME", "same"):
            pad = kernel_size // 2
        else:
            raise RuntimeError("invalid string value for padding")

        # define layers in conv pass - NOTE the use of 2D convolution and ReLu activation function. Also note the fact that 2 convolution (and following ReLu) are implemented
        self.conv_pass = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=pad
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, padding=pad
            ),
            torch.nn.ReLU(),
        )

        #initialize the weights of the convolutional block
        for _name, layer in self.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    #define forward function, required for PyTorch to take advantage of the ConvBlock class behind the scene
    def forward(self, x):
        """
        applies the convolutional block to a 3D tensor (x, CYX) and returns the result.
        """
        return self.conv_pass(x)


class Downsample(torch.nn.Module):
    """
    Downsampling class for a unet. Performs a 2D downsampling using max-pooling as a default method.
    It is possible to specify the downsampling factor using the parameter downsample_factor (int). Each size of the input image must be dividable by the
    downsampling_factor (no remainder must be present).
    """
    def __init__(self, downsample_factor: int):
        #call the bound __init__ from the parent class (torch.nn.Module) that follows the child class (ConvBlock).
        #refer to https://stackoverflow.com/questions/222877/what-does-super-do-in-python-difference-between-super-init-and-expl
        #refer to https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods
        super().__init__()

        self.downsample_factor = downsample_factor

        #defines the downsampling operation (max-pooling)
        self.down = torch.nn.MaxPool2d(
            downsample_factor
        ) 

    def check_valid(self, image_size: tuple[int, int]) -> bool:
        """Check if the downsample factor evenly divides each image dimension
        """
        for dim in image_size:
            if dim % self.downsample_factor != 0:
                return False
        return True
    
    #define forward function, required for PyTorch to take advantage of the Downsample class behind the scene
    def forward(self, x):
        """
        applies downsample to a 3D tensor (x, CYX) and returns the result.
        Raises an error if the at least one of the YX dimensions is not dividable by self.downsample_factor (see above).
        """
        dim_2_check = tuple(x.size()[-2:]) #only the last 2 dimensions matter, as the output of the convolutional block is 3D where channels are in position 1
        if not self.check_valid(dim_2_check):
            raise RuntimeError(
                "Can not downsample shape %s with factor %s"
                % (x.size(), self.downsample_factor)
            )

        return self.down(x)


class CropAndConcat(torch.nn.Module):
    """
    Implements the skip connections between downsampling (descending) half and upsampling (ascending) parts of the U-net.
    NOTE: as indicated above, the function is taken from https://github.com/dl4mia/01_segmentation I am not 100% sure why there is no init function
    """
    def crop(self, x, y):
        """
        Center-crop x to match spatial dimensions given by y.
        x and y must have the same number of dimension. It is assumed that no dimension of y has size > than the corresponding dimension in x.
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

    #define forward function, required for PyTorch to take advantage of the CropAndConcat class behind the scene
    def forward(self, encoder_output, upsample_output):
        """
        applies crop in order to crop the outer part of each dimension of the encoder_output so that it matches the
        size of the corresponding dimension of the upsample_output.
        Concatenates the cropped encoder_output and the upsample_ouput on the first dimension.
        """
        encoder_cropped = self.crop(encoder_output, upsample_output)

        return torch.cat([encoder_cropped, upsample_output], dim=1)

class OutputConv(torch.nn.Module):
    """
    the final convolution block of the u-net network. It should be encoded separately because it could be different from the other blocks, notably in the
    fact that the number of channel output could be inputed by the user (in the full u-net architecture - the UNet class below - the input and output channels
    of each layer are automatically calculated) depending on their needs, and also the final activation function can be inputed by the user.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str | None = None,  # Accepts the name of any torch activation function (e.g., ``ReLU`` for ``torch.nn.ReLU``). Refer to https://docs.python.org/3/library/functions.html#getattr and to https://discuss.pytorch.org/t/call-activation-function-from-string/30857
    ):
        
        #call the bound __init__ from the parent class (torch.nn.Module) that follows the child class (ConvBlock).
        #refer to https://stackoverflow.com/questions/222877/what-does-super-do-in-python-difference-between-super-init-and-expl
        #refer to https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods
        super().__init__()
        
        #the last convolun uses a kernel of size 1 (no convolution is done) and no padding. This simply allows to define the number of channels of the output
        self.final_conv = torch.nn.Conv2d(in_channels, out_channels, 1, padding=0)
        if activation is None:
            self.activation = None
        else:
            #if an activation function is indicatded, get it among the available in pytorch
            self.activation = getattr(torch.nn, activation)()

    #define forward function, required for PyTorch to take advantage of the OutputConv class behind the scene
    def forward(self, x):
        """
        applies the final convolution and, if indicated, the final activation function
        """
        x = self.final_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class UNet(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int = 1, 
        final_activation: str | None = None,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        padding: str = "same",
        upsample_mode: str = "nearest",
    ):
        """A U-Net for 2D input that expects tensors shaped like::
            ``(batch, channels, height, width)``.
        Args:
            depth:
                The number of levels in the U-Net. 2 is the smallest that really
                makes sense for the U-Net architecture, as a one layer U-Net is
                basically just 2 conv blocks.
            in_channels:
                The number of input channels in your dataset.
            out_channels (optional):
                How many output channels you want. Depends on your task. Defaults to 1.
            final_activation (optional):
                What activation to use in your final output block. Depends on your task.
                Defaults to None.
            num_fmaps (optional):
                The number of feature maps in the first layer. Defaults to 64.
            fmap_inc_factor (optional):
                By how much to multiply the number of feature maps between
                layers. Layer ``l`` will have ``num_fmaps*fmap_inc_factor**l`` 
                feature maps. Defaults to 2.
            downsample_factor (optional):
                Factor to use for down- and up-sampling the feature maps between layers.
                Defaults to 2.
            kernel_size (optional):
                Kernel size to use in convolutions on both sides of the UNet.
                Defaults to 3.
            padding (optional):
                How to pad convolutions. Either 'same' or 'valid'. Defaults to "same."
            upsample_mode (optional):
                The upsampling mode to pass to torch.nn.Upsample. Usually "nearest" 
                or "bilinear." Defaults to "nearest."
        """

        super().__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_activation = final_activation
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factor = downsample_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.upsample_mode = upsample_mode

        # CREATE THE LEFT CONVOLUTIONAL BLOCKS - the descending/downsampling part
        # Iterate through the layers of the network (specified by the depth). Per each layer, 1) Calculate, automatically using the compute_fmaps_encoder function
        # the number of input and output channels of the convolutional block. 2) Create the convolutional block (using the ConvBlock class above) with the appropriate
        # number of input and output channels. 3) Add the convolutional block to a collection list.
        self.left_convs = torch.nn.ModuleList()
        for level in range(self.depth):
            fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
            self.left_convs.append(
                ConvBlock(
                    fmaps_in,
                    fmaps_out,
                    self.kernel_size,
                    self.padding
                )
            )

        # CREATE THE RIGHT CONVOLUTIONAL BLOCKS - the ascending/upsampling part
        # Iterate through the layers of the network (specified by the depth) - 1 (as the last, bottomw layer is created during the descending part).
        # Per each layer, 1) Calculate, automatically using the compute_fmaps_decoder function the number of input and output channels of the convolutional
        # block. 2) Create the convolutional block (using the ConvBlock class above) with the appropriate number of input and output channels.
        # 3) Add the convolutional block to a collection list.
        self.right_convs = torch.nn.ModuleList()
        for level in range(self.depth - 1):
            fmaps_in, fmaps_out = self.compute_fmaps_decoder(level)
            self.right_convs.append(
                ConvBlock(
                    fmaps_in,
                    fmaps_out,
                    self.kernel_size,
                    self.padding,
                )
            )
        
        #Instantiate the downsampling object
        self.downsample = Downsample(self.downsample_factor)

        #Instantiate the upsampling object
        #NOTE WELL! THE UPSAMPLING FUNCTION EXPECTS 4D INPUTS (MINIBATCH, CHANNEL, HEIGHT, WIDTH)...I'LL HAVE TO CHECK THAT THIS HAPPENS
        self.upsample = torch.nn.Upsample(
                    scale_factor=self.downsample_factor,
                    mode=self.upsample_mode,
                )
        
        #Instantiate the crop-and-concat object - it allows to create the skip connection across left and right parts of the u-net, and it is based on the
        #CropAndConcat class defined above
        self.crop_and_concat = CropAndConcat()

        #Instantiate the object to perform the final convolution - it allows to specify the final number of channels of the output as well as the final
        #activation function - it is based on the OutputCov class above
        #note that the input channels are computed using the compute_fmaps_decoder function for the level 0 and taking the output in position 1 (the output channels)
        self.final_conv = OutputConv(
            self.compute_fmaps_decoder(0)[1], self.out_channels, self.final_activation
        )

    def compute_fmaps_encoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for 
        a conv block at a given level of the UNet encoder (left side). 

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        #use the input in_channels as input channels of the convolutional block, if level 0
        if level == 0:  # Leave out function
            fmaps_in = self.in_channels
        else:
            #get the number of channels output by the former level as input of the current level
            fmaps_in = self.num_fmaps * self.fmap_inc_factor ** (level - 1)

        #increase the number of channels of an exponential fmap_inc_factor at each lever - identical for the encoder and the decoder
        fmaps_out = self.num_fmaps * self.fmap_inc_factor**level
        return fmaps_in, fmaps_out

    def compute_fmaps_decoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for a conv block
        at a given level of the UNet decoder (right side). Note:
        The bottom layer (depth - 1) is considered an "encoder" conv pass, 
        so this function is only valid up to depth - 2.
        
        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        #increase the number of channels of an exponential fmap_inc_factor at each level - identical for the encoder and the decoder
        fmaps_out = self.num_fmaps * self.fmap_inc_factor ** (level)  # Leave out function
        
        #at each level, the number of input channels is the channel that comes from the skip connection (the output of the level in the encoder part) plus the ouput of the level at higher depth (lower in the u-net)
        concat_fmaps = self.compute_fmaps_encoder(level)[1]  # The channels that come from the skip connection - note that it is taken the output in position 1 of the tuple output of compute_fmaps_encoder (which are the channel outputs of the same level in the encoder part)
        fmaps_in = concat_fmaps + self.num_fmaps * self.fmap_inc_factor ** (level + 1)
        return fmaps_in, fmaps_out

    #define forward function, required for PyTorch to take advantage of the Downsample class behind the scene
    def forward(self, x):
        """
        process x through the full steps of the u-net architecture.
        """
        # LEFT SIDE - DESCENDING/DOWNSAMPLING SIDE
        #initialize a list to collect the ouputs of the convolutional blocks during the descending part - it will be used for the skip connections
        convolution_outputs = []
        #initialize the layer_input as the input x
        layer_input = x
        #iterate through the descending levels, but leave out the bottom level
        for i in range(self.depth - 1):  # -1 allows to leave out the bottom level
            #get the convolutional block corresponding at level i from the list of convolutional blocks assembled by the left_convs method. Pass the layer_input to it.
            conv_out = self.left_convs[i](layer_input)
            #append the ouput to the collection list
            convolution_outputs.append(conv_out)
            #downsample the output of the convolutional block using the downsample object
            downsampled = self.downsample(conv_out)
            #update layer_input, so that the output of the downsampling will be inputed to the convolutional block of the next (lower) level
            layer_input = downsampled

        # BOTTOM LEVEL
        #get the convolutional block corresponding at bottom level from the list of convolutional blocks assembled by the left_convs method. Pass the layer_input to it.
        conv_out = self.left_convs[-1](layer_input)
        #update the layer_input with the output of the bottom convolutional block (this is different than before, where the output of the downsampling was used)
        layer_input = conv_out

        # RIGHT SIDE - ASCENDING/UPSAMPLING SIDE
        #iterate through the levels in an ascending order. Leave out the bottom level
        for j in range(0, self.depth-1)[::-1]:  # -1 allows to leave out the bottom level. The [::-1] inverts the order of the levels so that they are ascending
            #use the upsample object to upsample the data (the layer_input)
            upsampled = self.upsample(layer_input)
            #concatenate the output of the descending convolutional block from the level j (collected in the list convolution_outputs) and the upsampled data result
            concat = self.crop_and_concat(convolution_outputs[j], upsampled)
            #get the convolutional block corresponding at level j from the list of convolutional blocks assembled by the right_convs method. Pass the layer_input to it.
            conv_output = self.right_convs[j](concat)
            #update layer_input, so that the output of the convolutional block will be inputed to the upsampling operation of the next (higher) level
            layer_input = conv_output

        #output the result of the final convolutional block
        return self.final_conv(layer_input)


