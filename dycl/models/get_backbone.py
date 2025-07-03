import torch.nn as nn
import torchvision.models as models

import dycl.lib as lib


def get_backbone(name, pretrained=True, **kwargs):
    if name == 'resnet34':
        lib.LOGGER.info("using ResNet-34")
        out_dim = 512
        backbone = models.resnet34(pretrained=pretrained)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    elif name == 'resnet50':
        lib.LOGGER.info("using ResNet-50")
        out_dim = 2048
        backbone_1 = models.resnet50(pretrained=pretrained)
        backbone_1 = nn.Sequential(*list(backbone_1.children())[:-2])
        backbone_2 = models.resnet50(pretrained=pretrained)
        backbone_2 = nn.Sequential(*list(backbone_2.children())[:-2])
        backbone_3 = models.resnet50(pretrained=pretrained)
        backbone_3 = nn.Sequential(*list(backbone_3.children())[:-2])
        pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    return (backbone_1, backbone_2, backbone_3, pooling, out_dim)
