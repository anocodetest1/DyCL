from typing import Optional, Union
from geopy.distance import distance 
import torch
import numpy as np


def create_label(
    labels: torch.Tensor,
    other_labels: Optional[torch.Tensor] = None,
    matrix:Optional[torch.Tensor] = None,
    hierarchy_level: Optional[Union[int, str]] = None,
    dtype: torch.dtype = torch.float,
):
   
    label_matrix = torch.zeros((len(labels), len(other_labels)),dtype=torch.float)

    for i in range(len(labels)):
         for j in range(len(other_labels)):
            #  if i == j:
            #     label_matrix[i, i] = 3
            #  else:
               distances = matrix[labels[i], other_labels[j]]
               label_matrix[i, j] = distances
              #  label_matrix[j, i] = distances

    if (hierarchy_level is not None) and (hierarchy_level != "MULTI"):
          if hierarchy_level == 0 :
            label_matrix[label_matrix == 3] = 1
            label_matrix[label_matrix < 3] = 0 
          if hierarchy_level == 1 :
            label_matrix[label_matrix >= 2] = 1
            label_matrix[label_matrix < 2] = 0 
          if hierarchy_level == 2 :
            label_matrix[label_matrix >= 1] = 1
            label_matrix[label_matrix < 1] = 0 
          new_matrix = label_matrix.to(dtype)    
          return new_matrix
    else:

      new_matrix = label_matrix.to(dtype)
    #print(label_matrix)

      return new_matrix