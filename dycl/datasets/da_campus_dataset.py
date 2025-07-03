from os.path import join

import pandas as pd
import numpy as np
import copy
import dycl.lib as lib
from dycl.datasets.base_dataset import BaseDataset


class DaCampusDataset(BaseDataset):

    HIERARCHY_LEVEL = 3

    def __init__(self, data_dir, mode, platform_1,  platform_3, transform=None, **kwargs):

        dir = lib.expand_path(data_dir)
        self.mode = mode
        self.data_dir = dir
        self.transform = transform
        self.platform_1 = platform_1
        self.platform_3 = platform_3
        self.data_dir_1 = join(dir, self.platform_1)
        self.data_dir_3 = join(dir, self.platform_3)
        if mode == 'train':
            mode = ['train']
        elif mode == 'test':
            mode = ['test']
        elif mode == 'all':
            mode = ['train', 'test']
        else:
            raise ValueError(f"Mode unrecognized {mode}")

        self.paths_3 = []
        self.paths_1 = []
        self.labels = []
        labels = []
        super_labels = []
        self.pairs = []
        latitude = []
        longitude = []
        self.coords = []
        for splt in mode:
            gt = pd.read_csv(join(self.data_dir_3, f'{splt}.txt'), sep=' ')
            self.paths_3.extend(gt["path"].apply(lambda x: join(self.data_dir_3, x)).tolist())
            self.labels.extend((gt["class_id"] - 1).tolist())
            
        for splt in mode:
            gt = pd.read_csv(join(self.data_dir_1, f'{splt}.txt'), sep=' ')
            self.paths_1.extend(gt["path"].apply(lambda x: join(self.data_dir_1, x)).tolist())
            #self.labels.extend((gt["class_id"] - 1).tolist())
            latitude.extend((gt["latitude"]).tolist())
            longitude.extend((gt["longitude"]).tolist())
            self.coords = np.stack([latitude, longitude], axis=1)
        k=0 
        #j=1 
        m=0
        
        for path in self.paths_1:
                coord = self.coords[m]
#                label = self.labels[m]
                for i in range(60):
                    label = self.labels[k]
                    self.pairs.append((k, label, coord, path, self.paths_3[k]))
                    k=k+1
                    #j=j+1
                m=m+1
        

        
        self.samples = copy.deepcopy(self.pairs)
        super().__init__(**kwargs)