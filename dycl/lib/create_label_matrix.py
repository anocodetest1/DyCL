from typing import Optional, Union
from geopy.distance import distance 
import torch
import numpy as np


def create_label_matrix(
    coords: torch.Tensor,
    other_coords: Optional[torch.Tensor] = None,
    hierarchy_level: Optional[Union[int, str]] = None,
    dtype: torch.dtype = torch.float,
):
   

    label_matrix = torch.zeros((len(coords), len(other_coords)),dtype=torch.float)

    for i in range(len(coords)):
#         for j in range(i, len(other_coords)):
        for j in range(len(other_coords)):
             if i == j:
                label_matrix[i, i] = 0
             else:
               distances = distance(coords[i], other_coords[j]).m
               label_matrix[i, j] = distances
#               label_matrix[j, i] = distances


    if (hierarchy_level is not None) and (hierarchy_level != "MULTI"):
        if hierarchy_level == 0 :
            label_matrix[label_matrix == 0] = 1
            label_matrix[label_matrix > 1] = 0 
        if hierarchy_level == 1 :
            label_matrix[label_matrix <= 200] = 1
            label_matrix[label_matrix > 200] = 0 
        if hierarchy_level == 2 :
            label_matrix[label_matrix <= 500] = 1
            label_matrix[label_matrix > 500] = 0 
        new_matrix = label_matrix.to(dtype)    
        return new_matrix
    else:
#
      label_matrix[label_matrix < 3] = 3  #
      label_matrix[(label_matrix > 3) & (label_matrix <= 200)] = 2  #
      label_matrix[(label_matrix > 200) & (label_matrix <= 500)] = 1  #
      label_matrix[label_matrix > 500] = 0  #
      new_matrix = label_matrix.to(dtype)
    #print(label_matrix)

      return new_matrix
