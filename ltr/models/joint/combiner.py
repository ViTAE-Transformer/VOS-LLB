import torch.nn as nn
import ltr.models.layers.filter as filter_layer
import math
from pytracking import TensorList
import torch

class Combiner(nn.Module):
    """ Target model constituting a single conv layer, along with the few-shot learner used to obtain the target model
        parameters (referred to as filter), i.e. weights of the conv layer
    """
    def __init__(self, num_filters):
        super().__init__()

        self.tm_1 = nn.Linear(num_filters, num_filters, bias=False)
        self.relu = nn.ReLU()
        self.tm_2 = nn.Linear(num_filters, num_filters, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tr_1 = nn.Linear(num_filters, num_filters, bias=False)
        self.tr_2 = nn.Linear(num_filters, num_filters, bias=False)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, mask_encoding_pred_tm, mask_encoding_pred_tr):
        """ the mask should be 5d"""
        assert mask_encoding_pred_tm.dim() == 5
        assert mask_encoding_pred_tr.dim() == 5

        num_sequences = mask_encoding_pred_tm.shape[1]
        C, H, W = mask_encoding_pred_tm.shape[-3], mask_encoding_pred_tm.shape[-2], \
                  mask_encoding_pred_tm.shape[-1]

        if mask_encoding_pred_tm.dim() == 5:
            mask_encoding_pred_tm = mask_encoding_pred_tm.view(-1, C, H * W)
            mask_encoding_pred_tm = mask_encoding_pred_tm.permute(0, 2, 1)
            tm_res = self.tm_1(mask_encoding_pred_tm)
            tm_res = self.relu(tm_res)
            tm_res = self.tm_2(tm_res)
            gate_wei_tm = self.sigmoid(tm_res)
            mask_encoding_pred_tm = gate_wei_tm * mask_encoding_pred_tm
            mask_encoding_pred_tm = mask_encoding_pred_tm.permute(0, 2, 1)
            mask_encoding_pred_tm = mask_encoding_pred_tm.view(1, num_sequences, C, H, W)
        if mask_encoding_pred_tr.dim() == 5:
            mask_encoding_pred_tr = mask_encoding_pred_tr.view(-1, C, H * W)
            mask_encoding_pred_tr = mask_encoding_pred_tr.permute(0, 2, 1)
            tr_res = self.tr_1(mask_encoding_pred_tr)
            tr_res = self.relu(tr_res)
            tr_res = self.tr_2(tr_res)
            gate_wei_tr = self.sigmoid(tr_res)
            mask_encoding_pred_tr = gate_wei_tr * mask_encoding_pred_tr
            mask_encoding_pred_tr = mask_encoding_pred_tr.permute(0, 2, 1)
            mask_encoding_pred_tr = mask_encoding_pred_tr.view(1, num_sequences, C, H, W)

        mask_encodings = mask_encoding_pred_tm + mask_encoding_pred_tr

        return mask_encodings