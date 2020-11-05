import torch.nn as nn
import torch.nn.functional as F
from models.util import _get_num_features


class L2TWW(nn.ModuleList):
    def __init__(self, source_model, target_model, pairs):
        super(L2TWW, self).__init__()
        self.src_list = _get_num_features(source_model)
        self.tgt_list = _get_num_features(target_model)
        self.pairs = pairs

        for src_idx, tgt_idx in pairs:
            self.append(nn.Conv2d(self.tgt_list[tgt_idx], self.src_list[src_idx], 1))

    def forward(self, source_features, target_features,
                weight, beta, loss_weight):

        matching_loss = 0.0
        for i, (src_idx, tgt_idx) in enumerate(self.pairs):
            sw = source_features[src_idx].size(3)
            tw = target_features[tgt_idx].size(3)
            if sw == tw:
                diff = source_features[src_idx] - self[i](target_features[tgt_idx])
            else:
                diff = F.interpolate(
                    source_features[src_idx],
                    scale_factor=tw / sw,
                    mode='bilinear'
                ) - self[i](target_features[tgt_idx])
            diff = diff.pow(2).mean(3).mean(2)
            if loss_weight is None and weight is None:
                diff = diff.mean(1).mean(0).mul(beta[i])
            elif loss_weight is None:
                diff = diff.mul(weight[i]).sum(1).mean(0).mul(beta[i])
            elif weight is None:
                diff = (diff.sum(1)*(loss_weight[i].squeeze())).mean(0).mul(beta[i])
            else:
                diff = (diff.mul(weight[i]).sum(1)*(loss_weight[i].squeeze())).mean(0).mul(beta[i])
            matching_loss = matching_loss + diff
        return matching_loss
