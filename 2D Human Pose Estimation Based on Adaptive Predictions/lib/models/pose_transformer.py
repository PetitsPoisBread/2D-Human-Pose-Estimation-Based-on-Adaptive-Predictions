from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from models.transformer import build_transformer
from models.backbone import build_backbone

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class PoseTransformer(nn.Module):

    def __init__(self, cfg, backbone, transformer, **kwargs):
        super(PoseTransformer, self).__init__()
        extra = cfg.MODEL.EXTRA
        self.num_queries = extra.NUM_QUERIES
        self.num_classes = cfg.MODEL.NUM_JOINTS
        self.aux_loss = extra.AUX_LOSS

        import math
        num_queries= self.num_queries
        num_classes =self.num_classes+1
        aux_loss = self.aux_loss
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embedd = nn.Embedding(num_queries, 4) #hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.transformer.decoder.bbox_embed = self.bbox_embed
        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.bbox_embed_diff_each_layer = False
        self.query_dim = 4
        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, x):
        src, pos = self.backbone(x)
        hs,ref = self.transformer(self.input_proj(src[-1]), None,
                              self.query_embedd.weight, pos[-1],x)

        if not self.bbox_embed_diff_each_layer:
            reference_before_sigmoid = inverse_sigmoid(ref)
            tmp = self.bbox_embed(hs)
            tmp[..., :self.query_dim] += reference_before_sigmoid
            outputs_coord = tmp[...,:2].sigmoid()
            outputs_sigma = tmp[...,2:].sigmoid()
        else:
            reference_before_sigmoid = inverse_sigmoid(ref)
            outputs_coords = []
            outputs_sigmas = []
            for lvl in range(hs.shape[0]):
                tmp = self.bbox_embed[lvl](hs[lvl])
                tmp[..., :self.query_dim] += reference_before_sigmoid[lvl]
                outputs_coord = tmp[...,:2].sigmoid()
                outputs_sigma = tmp[...,2:].sigmoid()
                outputs_coords.append(outputs_coord)
                outputs_sigmas.append(outputs_sigma)
            outputs_coord = torch.stack(outputs_coords)
            outputs_sigma = torch.stack(outputs_sigmas)

        outputs_class = self.class_embed(hs)
        out = {'pred_logits': outputs_class[-1], 'pred_coords': outputs_coord[-1], 'pred_sigmas': outputs_sigma[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord,outputs_sigma)
        return out



    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_sigma):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_coords': b, 'pred_sigmas': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_sigma[:-1])]


def get_pose_net(cfg, is_train, **kwargs):
    extra = cfg.MODEL.EXTRA

    transformer = build_transformer(hidden_dim=extra.HIDDEN_DIM, dropout=extra.DROPOUT, nheads=extra.NHEADS, dim_feedforward=extra.DIM_FEEDFORWARD,
                                    enc_layers=extra.ENC_LAYERS, dec_layers=extra.DEC_LAYERS, pre_norm=extra.PRE_NORM)
    pretrained = is_train and cfg.MODEL.INIT_WEIGHTS
    backbone = build_backbone(cfg, pretrained)
    model = PoseTransformer(cfg, backbone, transformer, **kwargs)

    return model
