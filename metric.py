#this code was copied from https://github.com/AlessandroUlivi/The_segmenters/blob/main/source/metric.py which was generated in the context of
#the course https://github.com/dl4mia

import torch
import torch.nn as nn
# import numpy as np

# Sorensen Dice Coefficient implemented in torch
# the coefficient takes values in two discrete arrays
# with values in {0, 1}, and produces a score in [0, 1]
# where 0 is the worst score, 1 is the best score
class DiceCoefficient(nn.Module):
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
    
    inputs:
    -prediction. Tensor.
    -target. Tensor. Must have the same shape of prediction.

    outpus:
    float. 1-DiceCoefficient
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


# # define a class to weight the importance of prediction and recall within the evaluation of
# # of a prediction. The idea is to use it to balance the importance which is given to the generation of labelled pixels when the model
# # is used for predictions (and force it not to generate images with only 0 values).
# class weighted_metric(nn.Module):
#     def __init__(self, eps=1e-6):
#         super().__init__()
#         self.eps = eps

#     def get_true_positives(self, bin_prediction, target):
#         return torch.sum(bin_prediction*target)

#     def get_true_negatives(self, bin_prediction, target):
#         pred_targ_sum = bin_prediction + target
#         true_negatives = torch.sum(torch.where(pred_targ_sum==0, 1,0))
#         return true_negatives
    
#     def get_false_positives(self, bin_prediction, target):
#         return torch.sum(torch.where(target==0, bin_prediction,0))
    
#     def get_false_negatives(self, bin_prediction, target):
#         return torch.sum(torch.where(bin_prediction==0, target,0))
    
#     def forward(self, prediction, target, w_tp=0.25, w_tn=0.25, w_fp=0.25, w_fn=0.25, bin_threshold=0.5):
#         assert w_tp+w_tn+w_fp+w_fn==1
#         bin_prediction = torch.where(prediction>bin_threshold, 1,0)
#         t_p = self.get_true_positives(bin_prediction, target)
#         t_n = self.get_true_negatives(bin_prediction, target)
#         f_p = self.get_false_positives(bin_prediction, target)
#         f_n = self.get_false_negatives(self, bin_prediction, target)
#         assert t_p+t_n+f_p+f_n==target.size()
#         assert t_p+t_n+f_p+f_n==prediction.size()

#         weighted__metric = 1 - (w_tp*t_p + w_tn*t_n + w_fp*f_p + w_fn*f_n)/(t_p + t_n + f_p + f_n)
#         return weighted__metric


