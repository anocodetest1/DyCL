from typing import Optional, Union
from geopy.distance import distance 
import torch
import numpy as np


def create_label_test(
    labels: torch.Tensor,
    other_labels: Optional[torch.Tensor] = None,
    matrix:Optional[torch.Tensor] = None,
    hierarchy_level: Optional[Union[int, str]] = None,
    dtype: torch.dtype = torch.float,
):
   
   
    if other_labels.ndim == 2:
       label_matrix = torch.zeros((len(labels), other_labels.size(1)),dtype=torch.float)
       for i in range(len(labels)):
          line = other_labels[i,:]
          for j in range(other_labels.size(1)):
               value = matrix[labels[i]-450, line[j]-450]
               label_matrix[i, j] = value
       if (hierarchy_level is not None) and (hierarchy_level != "MULTI"):
          if hierarchy_level == 0 :
            label_matrix[label_matrix < 3] = 0 
            label_matrix[label_matrix == 3] = 1
          if hierarchy_level == 1 :
            label_matrix[label_matrix < 2] = 0 
            label_matrix[label_matrix >= 2] = 1
          if hierarchy_level == 2 :
            #label_matrix[label_matrix < 1] = 0 
            label_matrix[label_matrix >= 1] = 1
          new_matrix = label_matrix.to(dtype)    
          return new_matrix
       else:

          new_matrix = label_matrix.to(dtype)  
          return new_matrix

    else:
       label_matrix = torch.zeros((len(labels), len(other_labels)),dtype=torch.float)


       for i in range(len(labels)):
         for j in range(len(other_labels)):
             value = matrix[labels[i]-450, other_labels[j]-450]
             label_matrix[i, j] = value
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