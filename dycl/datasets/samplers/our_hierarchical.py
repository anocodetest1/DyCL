import itertools
from collections import defaultdict
import copy
import time
import random
import torch
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm
import numpy as np
from pytorch_metric_learning.utils import common_functions as c_f


# Inspired by
# https://github.com/kunhe/Deep-Metric-Learning-Baselines/blob/master/datasets.py
class ourHierarchicalSampler(BatchSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        samples_per_class,
        batches_per_super_tuple=4,
        super_classes_per_batch=2,
        inner_label=0,
        outer_label=1,
    ):
        """
        labels: 2D array, where rows correspond to elements, and columns correspond to the hierarchical labels
        batch_size: because this is a BatchSampler the batch size must be specified
        samples_per_class: number of instances to sample for a specific class. set to "all" if all element in a class
        batches_per_super_tuples: number of batches to create for a pair of categories (or super labels)
        inner_label: columns index corresponding to classes
        outer_label: columns index corresponding to the level of hierarchy for the pairs
        """
        coords = dataset.coords
        labels = dataset.labels
        pairs = dataset.pairs
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        self.pairs = pairs
        self.batch_size = batch_size
        self.batches_per_super_tuple = batches_per_super_tuple
        self.samples_per_class = samples_per_class
        self.super_classes_per_batch = super_classes_per_batch
        self.reshuffle()

    def __iter__(
        self,
    ):
        self.reshuffle()
        for batch in self.batches:
            yield batch

    def __len__(
        self,
    ):
        return len(self.batches)

    def reshuffle(self):
            print("\nShuffle Dataset:")
            
            pair_pool = copy.deepcopy(self.pairs)
              
            # Shuffle pairs order
            random.shuffle(pair_pool)
           
            
            # Lookup if already used in epoch
            pairs_epoch = []
            
     
            # buckets
            batches = []
            idx_batch = []
            label_batch = []
             
            # counter
            break_counter = 0
            
            # progressbar
            pbar = tqdm()

            start_time = time.time()


            max_run_time = 60  
            while True:

                current_time = time.time()


                elapsed_time = current_time - start_time
                if elapsed_time > max_run_time:
                     print("The operation took longer than 1 minute and has been stopped.")
                     break
                
                pbar.update()
                
                if len(pair_pool) > self.batch_size:
                    pair = pair_pool.pop(0)
                    
                    index, idx, _, _, _ = pair
                    
                    # if idx not in label_batch and pair not in pairs_epoch:
                        
                    label_batch.append(idx)
                    idx_batch.append(index)
                        
                    pairs_epoch.append(pair)
            
                    break_counter = 0
                           
                            
                else:
                    break

                if len(idx_batch) >= self.batch_size:
                
                    # empty current_batch bucket to batches
                    batches.append(idx_batch)
                    idx_batch = []
                    label_batch = []
       
            pbar.close()
            
            # wait before closing progress bar
            time.sleep(0.3)
            c_f.NUMPY_RANDOM.shuffle(batches)
            self.batches = batches               
