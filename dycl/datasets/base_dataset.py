import os
from os.path import join, isfile
import time
import cv2
import copy
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import time
import random
import dycl.lib as lib
import dycl.engine as eng


class BaseDataset(Dataset):

    RELEVANCE_BS = 80
    CACHE_FILE = "DA_Campus_matrix_{mode}_{relevance_type}_{alpha}.trch"

    def __init__(
        self,
        restict_hierarchy=None,
        compute_relevances=True,
        alpha=1.0,
        alpha_train=None,
        relevance_type='pop',
        relevance_type_train=None,
        cache_relevances=True,
        force_reload=False,
        cache_matrix=True,
        compute_matrix=True,
    ):
        super().__init__()
        self.restict_hierarchy = restict_hierarchy
        self.alpha = alpha
        self.alpha_train = alpha if alpha_train is None else alpha_train
        self.relevance_type = relevance_type
        self.relevance_type_train = relevance_type if relevance_type_train is None else relevance_type_train
        self.shuffle_batch_size =80
        self.epoch = 0

        if compute_relevances and (self.mode != 'train'):
             cache_path = None

        else:
             self.relevances = None
        if compute_matrix and (self.mode != 'test'):
            cache_path_matrix = None
            if cache_matrix:
               cache_path_matrix = join(
                    self.data_dir,
                    self.CACHE_FILE.format(mode=self.mode, relevance_type=self.relevance_type, alpha=self.alpha)
                )                
               if isfile(cache_path_matrix) and not force_reload:
                    self.matrix = torch.load(cache_path_matrix, map_location='cpu')
               else:
                    self.get_matrix(cache=cache_path_matrix, verbose=True)

            else:
                self.get_matrix(cache=None, verbose=True)
        else:
            self.matrix = None            


    def __len__(self,):
        return len(self.paths_3)

    @property
    def my_sub_repr(self,):
        return ""

    def get_matrix(
        self,
        verbose=False,
        cache=None,
    ):
        matrix = []
        BS = self.RELEVANCE_BS
        lib.LOGGER.info("Launching matrix computation")
        iterator = tqdm(
            range(len(self.coords) // BS + (len(self.coords) % BS != 0)),
            f"Creating matrix for {self.__class__.__name__} mode={self.mode} relevance_type={self.relevance_type} alpha={self.alpha}",
            disable=not (verbose and (not os.getenv('TQDM_DISABLE')))
        )
        torch_coords = torch.from_numpy(self.coords).to('cuda' if os.getenv("USE_CUDA_FOR_RELEVANCE") else 'cpu')
        for i in iterator:
            mask = torch.ones(len(torch_coords), dtype=torch.bool)
            mask[i*BS:(i+1)*BS] = False
            target = lib.create_label_matrix(torch_coords[~mask], torch_coords)

            matrix.append(target)
        self.matrix = torch.cat(matrix).cpu()
        torch.cuda.empty_cache()

        if cache is not None:
            torch.save(self.matrix, cache)


    def get_all_relevance(
        self,
        verbose=False,
        cache=None,
    ):
        all_relevances = []
        BS = self.RELEVANCE_BS
        lib.LOGGER.info("Launching relevance computation")
        iterator = tqdm(
            range(len(self) // BS + (len(self) % BS != 0)),
            f"Creating relevance for {self.__class__.__name__} mode={self.mode} relevance_type={self.relevance_type} alpha={self.alpha}",
            disable=not (verbose and (not os.getenv('TQDM_DISABLE')))
        )
        torch_coords = torch.from_numpy(self.coords).to('cuda' if os.getenv("USE_CUDA_FOR_RELEVANCE") else 'cpu')
        for i in iterator:
            mask = torch.ones(len(torch_coords), dtype=torch.bool)
            mask[i*BS:(i+1)*BS] = False
            target = lib.create_label_matrix(torch_coords[~mask], torch_coords)
            rel = eng.relevance_for_batch(
                target,
                alpha=self.alpha,
                check_for=range(self.HIERARCHY_LEVEL+1),
                type=self.relevance_type,
            )
            all_relevances.append(rel)
        self.relevances = torch.cat(all_relevances).cpu()
        torch.cuda.empty_cache()

        if cache is not None:
            torch.save(self.relevances, cache)

    def compute_relevance_on_the_fly(self, target, train=True):
        relevance_type = self.relevance_type_train if train else self.relevance_type
        alpha = self.alpha_train if train else self.alpha

        return eng.compute_relevance_on_the_fly(
            target,
            alpha=alpha,
            check_for=range(self.HIERARCHY_LEVEL+1),
            type=relevance_type,
        )

    def set_epoch(self, e):
        self.epoch = e

    def __getitem__(self, idx):
        #if idx == 4321:
           # idx =4320
        _, label, coord, pth_s, pth_d = self.samples[idx]
        img_s = Image.open(pth_s).convert('RGB')
        img_d = Image.open(pth_d).convert('RGB')
        if self.transform:
            img_s = self.transform(img_s)
            img_d = self.transform(img_d)

       # label = torch.tensor(self.labels[idx, :])
        label = torch.tensor(label)
        coord = torch.tensor(coord)
        out = {"image_s": img_s, "image_d": img_d,"label": label,"coord":coord, "path_s": pth_s, "path_d": pth_d, "index": idx}

        if self.relevances is not None:
            relevances = self.relevances[idx, :]
            out["relevance"] = relevances

        return out

    def __repr__(self):
        repr = (
            f"{self.__class__.__name__}(\n"
            f"    mode={self.mode},\n"
            f"    len={len(self)},\n"
            f"    restict_hierarchy={self.restict_hierarchy},\n"
        )

        if self.relevances is not None:
            repr = repr + f"    alpha={self.alpha}\n"
            repr = repr + f"    relevance_type={self.relevance_type}\n"

        repr = repr + self.my_sub_repr + ')'
        return repr
