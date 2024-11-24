
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
# import numpy as np
# from skimage.feature import canny


# Sorensen Dice Coefficient implemented in torch
# the coefficient takes values in two discrete arrays
# with values in {0, 1}, and produces a score in [0, 1]
# where 0 is the worst score, 1 is the best score
class DiceCoefficient(nn.Module):
    """
    this code was copied from https://github.com/AlessandroUlivi/The_segmenters/blob/main/source/metric.py which was generated in the context of
    the course https://github.com/dl4mia
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    # the dice coefficient of two sets represented as vectors a, b can be
    # computed as (2 *|a b| / (abs(a) + abs(b))
    def forward(self, prediction, target):
        intersection = torch.sum(prediction*target)
        union = torch.sum(prediction)+torch.sum(target)
        return 2 * intersection / union.clamp(min=self.eps)


class DiceLoss(nn.Module):
    """
    calculate 1-DiceCoefficient, to be used as loss function.
    Values are expected to be in the 0-1 range (e.g. obtained from a Sigmoid activation function).
    
    inputs:
    -prediction. Tensor.
    -target. Tensor. Must have the same shape of prediction.

    outpus:
    float. 1-DiceCoefficient

    NOTES: this code was copied from https://github.com/AlessandroUlivi/The_segmenters/blob/main/source/metric.py which was generated in the context of
    the course https://github.com/dl4mia
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    # the dice coefficient of two sets represented as vectors a, b can be
    # computed as (2 *|a b| / (abs(a) + abs(b))
    def forward(self, prediction, target):

        intersection = torch.sum(prediction*target)
        union = torch.sum(prediction)+torch.sum(target)
        return 1 - (2 * intersection / union.clamp(min=self.eps))


class DiceBCELoss(nn.Module):
    """
    combination of BCE and DiceLoss. This code is adapted from https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
    Values of prediction and target are expected to be in the 0-1 range (e.g. for prediction, they have been obtained from a Sigmoid activation function).
    """
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, prediction, target, eps=1e-6, reduction="mean"):    
        intersection = torch.sum(prediction*target)   
        union = torch.sum(prediction)+torch.sum(target)                       
        dice_loss = 1 - (2 * intersection / union.clamp(min=eps))
        BCE = F.binary_cross_entropy(prediction, target, reduction=reduction)
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class FocalLoss(nn.Module):
    """
    Values of prediction and target are expected to be in the 0-1 range (e.g. for prediction, they have been obtained from a Sigmoid activation function).
    this code was adapted from https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch

    Focal Loss was introduced by Lin et al of Facebook AI Research in 2017 as a means of combatting extremely imbalanced
    datasets where positive cases were relatively rare. Their paper "Focal Loss for Dense Object Detection" is retrievable here:
    https://arxiv.org/abs/1708.02002.
    In practice, the researchers used an alpha-modified version of the function so I have included it in this implementation.

    """
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, prediction, target, alpha=1, gamma=5, reduction="mean", eps=1e-6):
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(prediction, target, reduction=reduction)
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss


class TverskyLoss(nn.Module):
    """
    Values of prediction and target are expected to be in the 0-1 range (e.g. for prediction, they have been obtained from a Sigmoid activation function).
    this code was adapted from https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch

    This loss was introduced in "Tversky loss function for image segmentationusing 3D fully convolutional deep networks",
    retrievable here: https://arxiv.org/abs/1706.05721. It was designed to optimise segmentation on imbalanced medical datasets by utilising
    constants that can adjust how harshly different types of error are penalised in the loss function. From the paper:

    "in the case of α=β=0.5 the Tversky index simplifies to be the same as the Dice coefficient, which is also equal to the F1 score.
    With α=β=1, Equation 2 produces Tanimoto coefficient, and setting α+β=1 produces the set of Fβ scores. Larger βs weigh recall higher than
    precision (by placing more emphasis on false negatives)."

    To summarise, this loss function is weighted by the constants 'alpha' and 'beta' that penalise false positives and false negatives
    respectively to a higher degree in the loss function as their value is increased. The beta constant in particular has applications in
    situations where models can obtain misleadingly positive performance via highly conservative prediction.
    You may want to experiment with different values to find the optimum. With alpha==beta==0.5, this loss becomes equivalent to Dice Loss.
    """
    def __init__(self):
        super(TverskyLoss, self).__init__()

    def forward(self, prediction, target, alpha=0.5, beta=0.5, eps=1e-6):
        
        #True Positives, False Positives & False Negatives
        TP = (prediction * target).sum()    
        FP = ((1-target) * prediction).sum()
        FN = (target * (1-prediction)).sum()
       
        Tversky = (TP) / (TP + alpha*FP + beta*FN + eps)  
        
        return 1 - Tversky


class BCE_EdgeDiceLoss(nn.Module):
    """
    """
    def __init__(self):
        super(BCE_EdgeDiceLoss, self).__init__()

    def forward(self, prediction, target, reduction="mean", eps=1e-6):
        
        gx_prediction, gy_prediction = torch.gradient(prediction[0,0,...])
        gx_target, gy_target = torch.gradient(target[0,0,...])
        prediction_edge = gy_prediction*gy_prediction + gx_prediction*gx_prediction
        target_edge = gy_target*gy_target + gx_target*gx_target
        bin_prediction_edge = torch.where(prediction_edge!=0.0, 1.0,0.0)
        bin_target_edge = torch.where(target_edge!=0.0,1.0,0.0)

        edge_intersection = torch.sum(bin_prediction_edge*bin_target_edge)   
        edge_union = torch.sum(bin_prediction_edge)+torch.sum(bin_target_edge)                       
        edge_dice_loss = 1 - (2 * edge_intersection / edge_union.clamp(min=eps))
        BCE = F.binary_cross_entropy(prediction, target, reduction=reduction)
        BCE_EdgeDice = BCE + edge_dice_loss
        
        return BCE_EdgeDice

class BCE_softEdgeDiceLoss(nn.Module):
    """
    """
    def __init__(self):
        super(BCE_softEdgeDiceLoss, self).__init__()

    def forward(self, prediction, kernel_size=3, sigma=1, reduction="mean", eps=1e-6):
        
        gx_prediction, gy_prediction = torch.gradient(prediction[0,...])
        prediction_edge = gy_prediction*gy_prediction + gx_prediction*gx_prediction
        bin_prediction_edge = torch.where(prediction_edge>0, 1,0)
        unsqueezed_bin_prediction_edge = torch.unsqueeze(bin_prediction_edge, dim=0)
        gaussian_smoother = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        gau_smooth_prediction_edge = gaussian_smoother(unsqueezed_bin_prediction_edge)

        # edge_intersection = torch.sum(bin_prediction_edge*bin_target_edge)   
        # edge_union = torch.sum(bin_prediction_edge)+torch.sum(bin_target_edge)                       
        # edge_dice_loss = 1 - (2 * edge_intersection / edge_union.clamp(min=eps))
        # BCE = F.binary_cross_entropy(prediction, target, reduction=reduction)
        # BCE_EdgeDice = BCE + edge_dice_loss
        
        return bin_prediction_edge, gau_smooth_prediction_edge
