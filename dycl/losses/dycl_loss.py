import torch
import torch.nn as nn
import torch.nn.functional as F

import dycl.lib as lib


class DyCLLoss(nn.Module):

    def __init__(self,hierarchy_level, num_classes, embedding_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.temperature = 1
        self.hierarchy_level = hierarchy_level 

        self.device = device


    def forward(self, image_features1, image_features2, labels, matrix, logit_scale=1):
        image_features1 = F.normalize(image_features1, dim=0)
        image_features2 = F.normalize(image_features2, dim=0)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        loss = (self.calculate(logits_per_image1, labels, matrix) + self.calculate(logits_per_image2, labels, matrix))/2

        return loss  
    
    def calculate(self, embeddings, labels, matrix):

        losses = []
        scale = 32
        margins = [0.25, 0.35, 0.45]  

        target = lib.create_label(labels, labels, matrix)

        index_0 = target == 0
        index_1 = target >= 1
        index_2 = target >= 2
        index_3 = target == 3
        LSE_neg = lib.mask_logsumexp(embeddings * scale, index_0)
        LSE_pos_high_3 = lib.mask_logsumexp(- embeddings * scale, index_1)
        LSE_pos_high_2 = lib.mask_logsumexp(- embeddings * scale, index_2)
        LSE_pos_high_1 = lib.mask_logsumexp(- embeddings * scale, index_3)


        lss_3 = F.softplus(LSE_neg + LSE_pos_high_3 + margins[0] * scale).mean()
        losses.append(lss_3)
        lss_2 = F.softplus(LSE_neg + LSE_pos_high_2 + margins[1] * scale).mean()
        losses.append(lss_2)
        lss_1 = F.softplus(LSE_neg + LSE_pos_high_1 + margins[2] * scale).mean()
        losses.append(lss_1)
        loss = sum(losses)
        

        return loss