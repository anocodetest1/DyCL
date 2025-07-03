import torch
import torch.nn as nn
import torch.nn.functional as F

import dycl.lib as lib

from dycl.models.get_pooling import get_pooling
from dycl.models.get_backbone import get_backbone


def flatten(tens):
    if tens.ndim == 2:
        return tens.squeeze(1)
    if tens.ndim == 3:
        return tens.squeeze(2).squeeze(1)
    if tens.ndim == 4:
        return tens.squeeze(3).squeeze(2).squeeze(1)


class RetrievalNet(nn.Module):

    def __init__(
        self,
        backbone_name,
        embed_dim=512,
        normalize=True,
        norm_features=False,
        without_fc=False,
        with_autocast=True,
        pooling='default',
        projection_normalization_layer='none',
        pretrained=True,
        **kwargs,
    ):
        super().__init__()

        norm_features = lib.str_to_bool(norm_features)
        without_fc = lib.str_to_bool(without_fc)
        with_autocast = lib.str_to_bool(with_autocast)

        self.embed_dim = embed_dim
        self.normalize = normalize
        self.norm_features = norm_features
        self.without_fc = without_fc
        self.with_autocast = with_autocast
        if with_autocast:
            lib.LOGGER.info("Using mixed precision")

        self.backbone_1, self.backbone_2, self.backbone_3, default_pooling, out_features = get_backbone(backbone_name, pretrained=pretrained, **kwargs)
        
        self.pooling = get_pooling(default_pooling, pooling)
        lib.LOGGER.info(f"Pooling is {self.pooling}")

        if self.norm_features:
            lib.LOGGER.info("Using a LayerNorm layer")
            self.standardize = nn.LayerNorm(out_features, elementwise_affine=False)
        else:
            self.standardize = nn.Identity()

        if not self.without_fc:
            self.fc = nn.Linear(out_features, embed_dim)
            lib.LOGGER.info(f"Projection head : \n{self.fc}")
        else:
            self.fc = nn.Identity()
            lib.LOGGER.info("Not using a linear projection layer")

    def forward(self, X1, X2, X3, return_before_fc=False):
        with torch.cuda.amp.autocast(enabled=self.with_autocast or (not self.training)):
            X1 = self.backbone_1(X1)
            X1 = self.pooling(X1)

            X1 = flatten(X1)
            X1 = self.standardize(X1)
            if return_before_fc:
                return X1

            X1 = self.fc(X1)
            if self.normalize or (not self.training):
                dtype = X1.dtype
                X1 = F.normalize(X1, p=2, dim=-1).to(dtype)


            X2 = self.backbone_2(X2)
            X2 = self.pooling(X2)

            X2 = flatten(X2)
            X2 = self.standardize(X2)
            if return_before_fc:
                return X2

            X2 = self.fc(X2)
            if self.normalize or (not self.training):
                dtype = X2.dtype
                X2 = F.normalize(X2, p=2, dim=-1).to(dtype)


            X3 = self.backbone_3(X3)
            X3 = self.pooling(X3)

            X3 = flatten(X3)
            X3 = self.standardize(X3)
            if return_before_fc:
                return X3

            X3 = self.fc(X3)
            if self.normalize or (not self.training):
                dtype = X3.dtype
                X3 = F.normalize(X3, p=2, dim=-1).to(dtype)              
            return (X1, X2, X3)
