import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SegmentationLosses(object):
    def __init__(self, args, nclass=3, ):
        self.args = args
        self.cuda = self.args.cuda
        self.nclass = nclass
        self.VID_branch = VIDLoss(num_input_channels=512, num_mid_channel=512,
                                  num_target_channels=512)
        self.VID_branch = self.VID_branch.cuda()

    def EnumerationLoss(self, logits, target, df, df_full, alpha_f, label):
        M = len(logits)
        loss = 0.

        MI_pred_mean, MI_log_scale = self.VID_branch(df_full)

        for l in reversed(range(1, 5)):
            for subset in itertools.combinations(list(range(M)), l):
                missing_logits = torch.stack([logits[l1] for l1 in subset], dim=0)
                alpha_m = torch.stack([alpha_f[l1] for l1 in subset], dim=0)
                df1 = torch.stack([df[l1] for l1 in subset], dim=0)

                missing_logits = torch.mean(missing_logits, dim=0)
                alpha_m = torch.mean(alpha_m, dim=0)
                df2 = torch.mean(df1, dim=0)

                loss += self.DiceCoef(missing_logits, target) + 0.1 * self.Cal_MIloss(df2, MI_pred_mean, MI_log_scale) + self.ce_loss_p(label, alpha_m)

        loss /= len(list(itertools.combinations(list(range(M)), M - self.args.loss.missing_num)))

        return loss


    def DiceCoef(self, preds, targets):
        smooth = 1.0
        class_num = self.nclass
        sigmoid = nn.Sigmoid()
        preds = sigmoid(preds)

        loss = torch.zeros(class_num, device=preds.device)
        for i in range(class_num):
            pred = preds[:, i, :, :]
            target = targets[:, i, :, :]
            intersection = (pred * target).sum()

            loss_dice = 1 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
            loss[i] = loss_dice

        return torch.mean(loss)


    def HD(self, a, alpha, c):
        # a = 2.0
        b = a / (a - 1)
        beta = torch.ones((1, c, 1, 1, 1)).cuda()

        F1_1 = torch.sum(torch.lgamma(a * alpha + beta), dim=1, keepdim=True) - torch.lgamma(
            torch.sum((a * alpha + beta), dim=1, keepdim=True))
        F2_1 = torch.sum(torch.lgamma((b + 1) * beta), dim=1, keepdim=True) - torch.lgamma(
            torch.sum(((b + 1) * beta), dim=1, keepdim=True))
        F3_1 = torch.sum(torch.lgamma(alpha + 2 * beta), dim=1, keepdim=True) - torch.lgamma(
            torch.sum((alpha + 2 * beta), dim=1, keepdim=True))

        hd1 = 1 / a * F1_1 + 1 / b * F2_1 - F3_1

        a = b
        b = a / (a - 1)
        F1_2 = torch.sum(torch.lgamma(a * alpha + beta), dim=1, keepdim=True) - torch.lgamma(
            torch.sum((a * alpha + beta), dim=1, keepdim=True))
        F2_2 = torch.sum(torch.lgamma((b + 1) * beta), dim=1, keepdim=True) - torch.lgamma(
            torch.sum(((b + 1) * beta), dim=1, keepdim=True))
        F3_2 = torch.sum(torch.lgamma(alpha + 2 * beta), dim=1, keepdim=True) - torch.lgamma(
            torch.sum((alpha + 2 * beta), dim=1, keepdim=True))

        hd2 = 1 / a * F1_2 + 1 / b * F2_2 - F3_2

        hd = 1 / 2 * (hd1 + hd2)

        return hd


    def ce_loss(self, label, alpha):
        E = alpha - 1
        alp = E * (1 - label) + 1
        B = self.HD(1.01, alp, 4)
        loss = torch.mean(B)

        return loss


    def HD_P(self, a, g, alpha, c):
        b = a / (a - 1)

        beta = torch.ones((1, c, 1, 1, 1)).cuda()
        term1 = (g / a) * alpha
        term2 = (g / b + 1) * beta
        sum_term = term1 + term2

        F1_1 = torch.sum(torch.lgamma(g * alpha + beta), dim=1, keepdim=True) - torch.lgamma(
            torch.sum((g * alpha + beta), dim=1, keepdim=True))
        F2_1 = torch.sum(torch.lgamma((g + 1) * beta), dim=1, keepdim=True) - torch.lgamma(
            torch.sum(((g + 1) * beta), dim=1, keepdim=True))
        F3_1 = torch.sum(torch.lgamma(sum_term), dim=1, keepdim=True) - torch.lgamma(
            torch.sum(sum_term, dim=1, keepdim=True))

        hd1 = 1 / a * F1_1 + 1 / b * F2_1 - F3_1

        a = b
        b = a / (a - 1)
        F1_2 = torch.sum(torch.lgamma(g * alpha + beta), dim=1, keepdim=True) - torch.lgamma(
            torch.sum((g * alpha + beta), dim=1, keepdim=True))
        F2_2 = torch.sum(torch.lgamma((g + 1) * beta), dim=1, keepdim=True) - torch.lgamma(
            torch.sum(((g + 1) * beta), dim=1, keepdim=True))
        F3_2 = torch.sum(torch.lgamma(sum_term), dim=1, keepdim=True) - torch.lgamma(
            torch.sum(sum_term, dim=1, keepdim=True))

        hd2 = 1 / a * F1_2 + 1 / b * F2_2 - F3_2

        hd = 1 / 2 * (hd1 + hd2)
        return hd


    def ce_loss_p(self, label, alpha):
        E = alpha - 1
        alp = E * (1 - label) + 1
        B = self.HD_P(1.5, 1.5, alp, 4)
        loss = torch.mean(B)

        return loss




    def Cal_MIloss(self, target, pred_mean, log_scale, eps=1e-5):
        pred_var = torch.log(1.0 + torch.exp(log_scale)) + eps
        pred_var = pred_var.view(1, -1, 1, 1, 1)
        neg_log_prob = 0.5 * ((pred_mean - target) ** 2 / pred_var + torch.log(pred_var))

        loss = torch.mean(neg_log_prob)
        return loss


class VIDLoss(nn.Module):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation"""

    def __init__(self,
                 num_input_channels,
                 num_mid_channel,
                 num_target_channels,
                 init_pred_var=5.0,
                 eps=1e-5):
        super(VIDLoss, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv3d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var - eps) - 1.0) * torch.ones(num_target_channels)
        )
        self.eps = eps

    def forward(self, x):
        pred_mean = self.regressor(x.float())
        log_scale = self.log_scale

        return pred_mean, log_scale

