from os.path import join

import pandas as pd
import numpy as np

import dycl.lib as lib
from dycl.datasets.test_dataset import TestDataset


class TestDaCampusDataset(TestDataset):

    HIERARCHY_LEVEL = 3

    def __init__(self, data_dir, mode, platform, platform_ref, transform=None, **kwargs):

        dir = lib.expand_path(data_dir)
        self.dir = dir
        self.mode = mode
        self.transform = transform
        self.platform = platform
        self.platform_ref = platform_ref
        self.data_dir = join(dir, self.platform)
        self.data_dir_ref = join(dir, self.platform_ref)
        if mode == 'train':
            mode = ['train']
        elif mode == 'test':
            mode = ['test']
        elif mode == 'all':
            mode = ['train', 'test']
        else:
            raise ValueError(f"Mode unrecognized {mode}")

        self.paths = []
        self.labels = []
        super_labels = []
        latitude = []
        longitude = []
        self.coords = []
        latitude_ref = []
        longitude_ref = []
        self.coords_ref = []
        for splt in mode:
            gt = pd.read_csv(join(self.data_dir, f'{splt}.txt'), sep=' ')
            self.paths.extend(gt["path"].apply(lambda x: join(self.data_dir, x)).tolist())
            self.labels.extend((gt["class_id"] - 1).tolist())
            latitude.extend((gt["latitude"]).tolist())
            longitude.extend((gt["longitude"]).tolist())
            self.coords = np.stack([latitude, longitude], axis=1)

            gt_ref = pd.read_csv(join(self.data_dir_ref, f'{splt}.txt'), sep=' ')
            latitude_ref.extend((gt_ref["latitude"]).tolist())
            longitude_ref.extend((gt_ref["longitude"]).tolist())
            self.coords_ref = np.stack([latitude_ref, longitude_ref], axis=1)


        super().__init__(**kwargs)
