# Code taken from: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/losses.py
# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from maad_centernet.utils.model_utils import _transpose_and_gather_feat
from torch.autograd import Variable


def _slow_neg_loss(pred, gt):
    """focal loss from CornerNet"""
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss(pred, gt):
    """Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = (
        torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    )

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    num_pos = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -= all_loss
    return loss


def _slow_reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _reg_loss(regr, gt_regr, mask):
    """L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class VeinStemLoss(nn.Module):
    def __init__(self, norm_target=False, sincos=False):
        super(VeinStemLoss, self).__init__()
        self.norm_target = norm_target
        self.reg_loss = F.l1_loss
        self.sincos = sincos

    def polar_to_cartesian(self, kp):
        kp = kp.reshape(kp.shape[0], kp.shape[1], 8, -1)

        if self.sincos:
            angle = torch.arctan2(
                ((kp[:, :, :, 2] * 2) - 1), ((kp[:, :, :, 1] * 2) - 1)
            )
        else:
            mult = torch.pi / 180
            if self.norm_target:
                mult *= 360
            angle = kp[:, :, :, 1]
            angle *= mult

        x = kp[:, :, :, 0] * torch.cos(angle)
        y = kp[:, :, :, 0] * torch.sin(angle)
        kp[:, :, :, 0] = x
        kp[:, :, :, 1] = y
        return kp[:, :, :, :2]  # kp.reshape(kp.shape[0], -1, 16)

    def cartesian_to_polar(self, kp):
        dist = torch.sqrt(kp[:, :, :, 0] ** 2 + kp[:, :, :, 1] ** 2)
        angle = torch.atan2(kp[:, :, :, 1], kp[:, :, :, 0])
        if self.sincos:
            kp_polar = torch.zeros((kp.shape[0], kp.shape[1], 8, 3)).to(
                kp.device
            )
            angle_cos = (torch.cos(angle) + 1) / 2  # [-1, 1] -> [0, 1]
            angle_sin = (torch.sin(angle) + 1) / 2  # [-1, 1] -> [0, 1]
            kp_polar[:, :, :, 0] = dist
            kp_polar[:, :, :, 1] = angle_cos
            kp_polar[:, :, :, 2] = angle_sin
        else:
            kp_polar = torch.zeros_like(kp).to(kp.device)
            mult = 180 / torch.pi
            if self.norm_target:
                mult /= 360
            angle *= mult
            angle = torch.where(angle < 0, angle + 360, angle)
            kp_polar[:, :, :, 0] = dist
            kp_polar[:, :, :, 1] = angle
        return kp_polar

    def get_closest_point_on_segment(self, p, a, b):
        ab = b - a
        ap = p - a
        t = torch.dot(ap, ab) / torch.dot(ab, ab)
        t = torch.clamp(t, 0, 1)
        closest = a + t * ab
        return closest, torch.linalg.norm(p - closest)

    def get_closest_point_on_segment_list(self, p, a, b):
        d_min = float("inf")
        closest_min = None
        for a_i, b_i in zip(a, b):
            closest, d = self.get_closest_point_on_segment(p, a_i, b_i)
            if d < d_min:
                d_min = d
                closest_min = closest
        return closest_min

    def project_prediction(self, pred, target, mask):
        # Convert to polar coordinates
        batch, count, len = pred.shape
        pred = self.polar_to_cartesian(pred)
        target = self.polar_to_cartesian(target)
        projected_points = torch.zeros_like(target)

        # Get projected points
        projected_points = torch.zeros_like(target)
        true_kp_indices = [0, 3, 7]
        for b_i in range(batch):
            for i, (m, p, t) in enumerate(
                zip(mask[b_i], pred[b_i], target[b_i])
            ):
                if torch.sum(mask[b_i, i:, :]) == 0:
                    break
                elif torch.sum(m) == 0:
                    continue
                else:
                    for j, (m_j, p_j, t_j) in enumerate(zip(m, p, t)):
                        if torch.sum(m_j) == 0:
                            continue
                        else:
                            if j in true_kp_indices:
                                projected_points[b_i, i, j, :] = t_j
                                continue
                            else:
                                # We have two segments i-1 --> i and i --> i+1
                                if torch.all((t[j - 1] == t[j + 1])):
                                    projected_points[b_i, i, j, :] = t_j
                                    continue
                                a = [t[j - 1], t_j]
                                b = [t_j, t[j + 1]]
                                # We want to find the closest point on all possible segments
                                projected_points[b_i, i, j, :] = (
                                    self.get_closest_point_on_segment_list(
                                        p_j, a, b
                                    )
                                )
        # projected_points = target
        # Do we need to transform back to polar coordinates?
        projected_points = self.cartesian_to_polar(projected_points)
        projected_points = projected_points.reshape(batch, count, len)
        return projected_points

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        # pred = output
        if mask.shape != pred.shape:
            mask = mask.unsqueeze(2).expand_as(pred).float()

        pred = pred * mask
        target = target * mask
        with torch.no_grad():
            projected_points = self.project_prediction(
                pred.clone(), target, mask
            )
        loss = self.reg_loss(
            pred * mask, projected_points * mask, size_average=False
        )
        loss = loss / (mask.sum() + 1e-4)
        return loss


class FocalLoss(nn.Module):
    """nn.Module warpper for focal loss"""

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class RegLoss(nn.Module):
    """Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        if mask.shape != pred.shape:
            mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class NormRegL1Loss(nn.Module):
    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(
            pred * mask, target * mask, reduction="elementwise_mean"
        )
        return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _transpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction="elementwise_mean")


# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction="elementwise_mean")


def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0])
        )
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0])
        )
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1])
        )
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1])
        )
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res
