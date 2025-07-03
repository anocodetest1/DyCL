import os
from os.path import join
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import dycl.lib as lib

from .get_knn import get_knn
from .get_knn_rerank import get_knn_rerank
from .metrics import get_metrics_dict
from .compute_embeddings_1 import compute_embeddings_1
from .compute_embeddings_3 import compute_embeddings_3
from .overall_accuracy_hook import overall_accuracy_hook


class AccuracyCalculator:

    def __init__(
        self,
        compute_for_hierarchy_levels=[0],
        exclude=[],
        recall_rate=[],
        hard_ap_for_level=[],
        with_binary_asi=False,
        overall_accuracy=False,
        metric_batch_size=256,
        inference_batch_size=256,
        num_workers=16,
        pin_memory=True,
        convert_to_cuda=True,
        with_faiss=False,
        with_rerank=False,
        data_dir=None,
        **kwargs,
    ):
        self.compute_for_hierarchy_levels = sorted(set(compute_for_hierarchy_levels))
        self.exclude = exclude
        self.recall_rate = recall_rate
        self.hard_ap_for_level = hard_ap_for_level
        self.with_binary_asi = with_binary_asi
        self.overall_accuracy = overall_accuracy
        self.metric_batch_size = metric_batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.convert_to_cuda = convert_to_cuda
        self.with_faiss = with_faiss
        self.with_rerank = with_rerank
        self.data_dir = data_dir
        

        self.METRICS_DICT = get_metrics_dict(
            recall_rate=self.recall_rate,
            hard_ap_for_level=self.hard_ap_for_level,
            with_binary_asi=self.with_binary_asi,
            **kwargs,
        )

    def get_embeddings_1(self, net, dts):
        loader = DataLoader(
            dts,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return compute_embeddings_1(
            net,
            loader,
            self.convert_to_cuda,
        )
    def get_embeddings_3(self, net, dts):
        loader = DataLoader(
            dts,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return compute_embeddings_3(
            net,
            loader,
            self.convert_to_cuda,
        )
    
    def batch_metrics(
        self,
        features,
        labels,
        coords,
        embeddings_come_from_same_source,
        relevances=None,
        ref_features=None,
        ref_labels=None,
        ref_coords=None,
        split=None,
    ):
        lib.LOGGER.info("Launching batched metrics")
        if ref_features is None:
            assert ref_labels is None
            ref_features, ref_labels = features, labels

        BS = self.metric_batch_size
        with_rerank = self.with_rerank
        dir = lib.expand_path(self.data_dir)
        matrix_dir = join(dir,f'DA_Campus_matrix_test_pop_1.0.trch')
        matrix = torch.load(matrix_dir, map_location=coords.device)        

        out = lib.DictAverage()
        iterator = tqdm(range(features.size(0) // BS + (features.size(0) % BS != 0)), disable=os.getenv("TQDM_DISABLE"))
        for i in iterator:
            logs = {}
            if with_rerank:
              indices, distances = get_knn_rerank(
                ref_features, features,
                ref_features.size(0)-1,
                split,
                i, BS,
                dir,
                embeddings_come_from_same_source=embeddings_come_from_same_source,
                with_faiss=self.with_faiss,
              )
            else:
              indices, distances = get_knn(
                ref_features, features[i*BS:(i+1)*BS],
                ref_features.size(0)-1,
                embeddings_come_from_same_source=embeddings_come_from_same_source,
                with_faiss=self.with_faiss,
              )
            in_dict = {}
            in_dict['sorted_target'] = lib.create_label_test(labels[i*BS:(i+1)*BS], ref_labels[indices], matrix, dtype=torch.int64)
            if relevances is not None:
                in_dict['relevances'] = relevances[i*BS:(i+1)*BS]
                in_dict['sorted_rel'] = lib.create_relevance_matrix(
                    in_dict['sorted_target'],
                    in_dict['relevances']
                )
            for key, metric in self.METRICS_DICT["multi_level"].items():
                if key not in self.exclude:
                    logs[f"{key}_multi"] = metric(**in_dict)

            for key, metric in self.METRICS_DICT["exclude_level"].items():
                if key not in self.exclude:
                    logs[f"{key}_exclude"] = metric(**in_dict)

            for hierarchy_level in self.compute_for_hierarchy_levels:
                # creates a binary label matrix corresponding to hierarchy_level
                binary_sorted_target = lib.create_label_test(
                    labels[i*BS:(i+1)*BS],
                    ref_labels[indices],
                    matrix,
                    hierarchy_level,
                ).float()
                for key, metric in self.METRICS_DICT["binary"].items():
                    if key not in self.exclude:
                        logs[f"{key}_level{hierarchy_level}"] = metric(binary_sorted_target)

            out.update(logs, in_dict['sorted_target'].size(0))
            iterator.set_postfix(out.avg)

        return out.avg

    def evaluate(
        self,
        net,
        dataset_dict,
        epoch=None,
    ):
        if epoch is not None:
            lib.LOGGER.info(f"Evaluating for epoch {epoch}")

        logs = {}
        for split, dts in dataset_dict.items():
            if isinstance(dts, Dataset):
                lib.LOGGER.info(f"Getting embeddings for the {split} set")
                features, labels, relevances = self.get_embeddings(net, dts)
                ref_features = ref_labels = None
                embeddings_come_from_same_source = True

            elif isinstance(dts, dict):
                if "satellite_drone" == split:
                    # gallery and queries are disjoint
                    lib.LOGGER.info(f"Getting embeddings for the queries of the {split} set")
                    features, labels, coords, relevances = self.get_embeddings_1(net, dts["query"])
                    lib.LOGGER.info(f"Getting embeddings for the gallery of the {split} set")
                    ref_features, ref_labels, ref_coords, ref_relevances = self.get_embeddings_3(net, dts["gallery"])
                    embeddings_come_from_same_source = False
                    features_1 = features
                    labels_1 = labels
                    coords_1 = coords
                    relevances_1 = relevances

                    features_3 = ref_features
                    labels_3 = ref_labels
                    coords_3 = ref_coords
                    relevances_3 = ref_relevances

                elif "drone_satellite" == split:
                    # gallery and queries are disjoint
                    lib.LOGGER.info(f"Getting embeddings for the queries of the {split} set")
                    #features, labels, coords, relevances = self.get_embeddings_3(net, dts["query"])
                    features = features_3
                    labels = labels_3
                    coords = coords_3
                    relevances = relevances_3
                    lib.LOGGER.info(f"Getting embeddings for the gallery of the {split} set")
                    #ref_features, ref_labels, ref_coords, _ = self.get_embeddings_1(net, dts["gallery"])
                    ref_features = features_1
                    ref_labels = labels_1
                    ref_coords = coords_1
                    ref_relevances = relevances_1
                    embeddings_come_from_same_source = False
                elif "drone_drone" == split:
                    # gallery and queries are disjoint
                    lib.LOGGER.info(f"Getting embeddings for the queries of the {split} set")
                    #features, labels, coords, relevances = self.get_embeddings_3(net, dts["query"])
                    features = features_3
                    labels = labels_3
                    coords = coords_3
                    relevances = relevances_3
                    lib.LOGGER.info(f"Getting embeddings for the gallery of the {split} set")
                    #ref_features, ref_labels, ref_coords, _ = self.get_embeddings_3(net, dts["gallery"])
                    ref_features = features_3
                    ref_labels = labels_3
                    ref_coords = coords_3
                    ref_relevances = relevances_3
                    embeddings_come_from_same_source = False
                elif "satellite_satellite" == split:
                    # gallery and queries are disjoint
                    lib.LOGGER.info(f"Getting embeddings for the queries of the {split} set")
                    #features, labels, coords, relevances = self.get_embeddings_1(net, dts["query"])
                    features = features_1
                    labels = labels_1
                    coords = coords_1
                    relevances = relevances_1
                    lib.LOGGER.info(f"Getting embeddings for the gallery of the {split} set")
                    #ref_features, ref_labels, ref_coords, _ = self.get_embeddings_1(net, dts["gallery"])
                    ref_features = features_1
                    ref_labels = labels_1
                    ref_coords = coords_1
                    ref_relevances = relevances_1
                    embeddings_come_from_same_source = False


            else:
                raise ValueError(f"Unknown type for dataset: {type(dts)}")

            logs[split] = self.batch_metrics(
                features,
                labels,
                coords,
                embeddings_come_from_same_source,
                relevances,
                ref_features,
                ref_labels,
                ref_coords,
                split,
            )

        if self.overall_accuracy:
            overall_logs = overall_accuracy_hook(logs)
            logs["test_overall"] = overall_logs

        return logs

    def __repr__(self):
        repr = (
            f"{self.__class__.__name__}(\n"
            f"    compute_for_hierarchy_levels={self.compute_for_hierarchy_levels},\n"
            f"    exclude={self.exclude},\n"
            f"    recall_rate={self.recall_rate},\n"
            f"    hard_ap_for_level={self.hard_ap_for_level},\n"
            f"    with_binary_asi={self.with_binary_asi},\n"
            f"    overall_accuracy={self.overall_accuracy},\n"
            f"    metric_batch_size={self.metric_batch_size},\n"
            f"    inference_batch_size={self.inference_batch_size},\n"
            f"    num_workers={self.num_workers},\n"
            f"    pin_memory={self.pin_memory},\n"
            f"    convert_to_cuda={self.convert_to_cuda},\n"
            f"    with_faiss={self.with_faiss},\n"
            ")"
        )
        return repr


@lib.get_set_random_state
def evaluate(
    net,
    dataset_dict,
    acc=None,
    epoch=None,
    **kwargs,
):
    if acc is None:
        acc = AccuracyCalculator(**kwargs)

    return acc.evaluate(
        net,
        dataset_dict,
        epoch=epoch,
    )
