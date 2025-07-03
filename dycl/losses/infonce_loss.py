import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
import dycl.lib as lib
import math

class InfoNCE(nn.Module):

    def __init__(self,hierarchy_level, num_classes, embedding_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.temperature = 1
        self.hierarchy_level = hierarchy_level 
        self.loss_function = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale=1):
        image_features1 = F.normalize(image_features1, dim=0)
        image_features2 = F.normalize(image_features2, dim=0)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss  
    