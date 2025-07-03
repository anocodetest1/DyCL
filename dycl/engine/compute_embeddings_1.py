import os

import torch
from tqdm import tqdm

import dycl.lib as lib


def compute_embeddings_1(
    net,
    loader,
    convert_to_cuda=False,
    with_paths=False,
):
    features = []

    mode = net.training
    net.eval()
    lib.LOGGER.info("Computing embeddings")
    for i, batch in enumerate(tqdm(loader, disable=os.getenv("TQDM_DISABLE"))):
        with torch.no_grad():
          X, X3 = net(batch["image"].cuda(), batch["image"].cuda())

        features.append(X)

    features = torch.cat(features)
    labels = torch.tensor(loader.dataset.labels).to('cuda' if convert_to_cuda else 'cpu')
    coords = torch.from_numpy(loader.dataset.coords).to('cuda' if convert_to_cuda else 'cpu')
    if loader.dataset.relevances is not None:
        relevances = loader.dataset.relevances.to('cuda' if convert_to_cuda else 'cpu')
    else:
        relevances = None

    net.train(mode)
    if with_paths:
        return features, labels, coords, relevances, loader.dataset.paths
    else:
        return features, labels, coords, relevances
