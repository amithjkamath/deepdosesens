# -*- encoding: utf-8 -*-
import torch.nn as nn


class UNetLoss(nn.Module):
    """Loss function for the UNet model.
    It computes the L1 loss between the predicted doses and the ground truth doses,
    considering only the regions where the possible dose mask is greater than zero.
    """

    def __init__(self):
        """Initialize the Loss class."""
        super().__init__()
        self.lossfunc = nn.L1Loss(reduction="mean")

    def forward(self, pred, gt):
        """Compute the loss.
        Args:
            pred (tuple): A tuple containing the predicted dose from the model.
            gt (tuple): A tuple containing the ground truth dose and the possible dose mask.
        Returns:
            L1_loss (torch.Tensor): The computed L1 loss.
        """
        gt_dose = gt[0]
        possible_dose_mask = gt[1]

        pred = pred[possible_dose_mask > 0]
        gt_dose = gt_dose[possible_dose_mask > 0]
        L1_loss = self.lossfunc(pred, gt_dose)
        return L1_loss


class C3DLoss(nn.Module):
    """Loss function for the Cascaded UNet model.
    It computes the L1 loss between the predicted doses and the ground truth doses,
    considering only the regions where the possible dose mask is greater than zero.
    """

    def __init__(self):
        """Initialize the Loss class."""
        super().__init__()
        self.lossfunc = nn.L1Loss(reduction="mean")

    def forward(self, pred, gt):
        """Compute the loss.
        Args:
            pred (tuple): A tuple containing the predicted doses from both branches of the model.
            gt (tuple): A tuple containing the ground truth dose and the possible dose mask.
        Returns:
            L1_loss (torch.Tensor): The computed L1 loss.
        """
        pred_A = pred[0]
        pred_B = pred[1]
        gt_dose = gt[0]
        possible_dose_mask = gt[1]

        pred_A = pred_A[possible_dose_mask > 0]
        pred_B = pred_B[possible_dose_mask > 0]
        gt_dose = gt_dose[possible_dose_mask > 0]
        L1_loss = 0.5 * self.lossfunc(pred_A, gt_dose) + self.lossfunc(pred_B, gt_dose)
        return L1_loss
