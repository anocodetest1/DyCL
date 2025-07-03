import os
from os.path import join, isfile
import numpy as np
import torch
import faiss
import pytorch_metric_learning.utils.common_functions as c_f
from scipy.stats import norm
import dycl.lib as lib


def get_knn_rerank(references, queries, num_k, split, i, BS, dir, embeddings_come_from_same_source, with_faiss=False):
    with_faiss = lib.str_to_bool(os.getenv("WITH_FAISS", with_faiss))

    num_k += embeddings_come_from_same_source

    if with_faiss:
        distances, indices = get_knn_faiss(references, queries, num_k)
    else:
        distances, indices = get_knn_torch(references, queries, num_k, split, i, BS, dir)

    if embeddings_come_from_same_source:
        return indices[:, 1:], distances[:, 1:]

    return indices, distances


def get_knn_faiss(references, queries, num_k):
    lib.LOGGER.debug("Computing k-nn with faiss")

    d = references.size(-1)
    device = references.device
    references = c_f.to_numpy(references).astype(np.float32)
    queries = c_f.to_numpy(queries).astype(np.float32)

    index = faiss.IndexFlatL2(d)
    try:
        if torch.cuda.device_count() > 1:
            co = faiss.GpuMultipleClonerOptions()
            co.shards = True
            index = faiss.index_cpu_to_all_gpus(index, co)
        else:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
    except AttributeError:
        # Only faiss CPU is installed
        pass

    index.add(references)
    distances, indices = index.search(queries, num_k)
    distances = c_f.to_device(torch.from_numpy(distances), device=device)
    indices = c_f.to_device(torch.from_numpy(indices), device=device)
    index.reset()
    return distances, indices


def get_knn_torch(references, queries, num_k, split, i, BS, dir):
    lib.LOGGER.debug("Computing k-nn with torch")
    # scores = queries @ references.t()
    if "satellite_drone" == split:
        scores_dir = join(dir,f'kr_sd_o.pt')
    else:
        scores_dir = join(dir,f'kr_ds_o.pt')        

    if i == 0 and isfile(scores_dir):
        os.remove(scores_dir)       

    if isfile(scores_dir):
        scores = torch.load(scores_dir, map_location='cpu')
    else:
        scores =re_ranking(queries, references, split, dir, k1=20, k2=6, lambda_value=0.3, beta=1.)
        # distance -> cosine
        scores = 1.0-scores
        torch.save(scores, scores_dir)  

    scores = scores[i*BS:(i+1)*BS] 
    distances, indices = torch.topk(scores, num_k)
    return distances, indices


def re_ranking(queries, references, split, dir, k1=20, k2=6, lambda_value=0.3, beta=1.):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    queries_np = queries.cpu().numpy()
    references_np = references.cpu().numpy()
    if "satellite_drone" == split:
       q_g_dist_dir = join(dir,f'q_g_dist.npy')
       g_q_dist_dir = join(dir,f'g_q_dist.npy')       
       q_q_dist_dir = join(dir,f'q_q_dist.npy')
       g_g_dist_dir = join(dir,f'g_g_dist.npy')
    else:
       q_g_dist_dir = join(dir,f'g_q_dist.npy')       
       g_q_dist_dir = join(dir,f'q_g_dist.npy')
       q_q_dist_dir = join(dir,f'g_g_dist.npy')
       g_g_dist_dir = join(dir,f'q_q_dist.npy')    
    if isfile(q_g_dist_dir) and isfile(q_q_dist_dir) and isfile(g_g_dist_dir):
            q_g_dist= np.load(q_g_dist_dir)
            g_q_dist= q_g_dist.T
            q_q_dist= np.load(q_q_dist_dir)
            g_g_dist= np.load(g_g_dist_dir)
    else:
        q_g_dist= queries_np @ references_np.T
        g_q_dist= q_g_dist.T        
        q_q_dist= queries_np @ queries_np.T
        g_g_dist= references_np @ references_np.T
        np.save(q_g_dist_dir, q_g_dist)
        np.save(g_q_dist_dir, g_q_dist)        
        np.save(q_q_dist_dir, q_q_dist)       
        np.save(g_g_dist_dir, g_g_dist)

    # cosine-distance
    q_g_dist = 1.0 - q_g_dist
    q_q_dist = 1.0 - q_q_dist
    g_g_dist = 1.0 - g_g_dist
    final_dist = re_ranking_in(q_g_dist, q_q_dist, g_g_dist, k1, k2, lambda_value, beta)
    final_dist_tensor = torch.tensor(final_dist, device=queries.device)

    return final_dist_tensor


def build_expanded_reciprocal_index(initial_rank, original_dist_vec, k1, idx, beta):
    neighbor_idx_f_k = initial_rank[idx, :k1 + 1]  
    neighbor_idx_b_k = initial_rank[neighbor_idx_f_k, :k1 + 1] 
    fi = np.where(neighbor_idx_b_k == idx)[0]
    k_rec_idx = neighbor_idx_f_k[fi]
    k_rec_idx_expand = k_rec_idx

    for j in range(len(k_rec_idx)):
        candidate = k_rec_idx[j]
        candidate_neighbor_idx_f_k = initial_rank[candidate, :int(np.around(k1 / 2.)) + 1]
        candidate_neighbor_idx_b_k = initial_rank[candidate_neighbor_idx_f_k, :int(np.around(k1 / 2.)) + 1]
        fi_candidate = np.where(candidate_neighbor_idx_b_k == candidate)[0]
        candidate_k_rec_idx = candidate_neighbor_idx_f_k[fi_candidate]
        if len(np.intersect1d(candidate_k_rec_idx, k_rec_idx)) > 2. / 3 * len(candidate_k_rec_idx):
            k_rec_idx_expand = np.append(k_rec_idx_expand, candidate_k_rec_idx)

    k_rec_idx_expand = np.unique(k_rec_idx_expand)
    weight = np.exp(-original_dist_vec[k_rec_idx_expand] * beta)

    return k_rec_idx_expand, weight


# MSRerank
def re_ranking_in(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3, beta=1.):
    original_dist = np.concatenate(
        [
            np.concatenate([q_q_dist, q_g_dist], axis=1),
            np.concatenate([q_g_dist.T, g_g_dist], axis=1)
        ],
        axis=0
    )

    original_dist = np.power(original_dist, 2).astype(np.float32)
    # original_dist = original_dist.astype(np.float32)
    original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    final_dist = np.zeros_like(original_dist[:query_num, ]).astype(np.float32)


    k1s = [k1, 62, 292]
    k2s = [k2, 6, 6]
    idss = []

    for level in range(3):

        V = np.zeros_like(original_dist).astype(np.float32)

        for i in range(all_num):

            k_re_idx, weight = build_expanded_reciprocal_index(initial_rank, original_dist[i, :], k1s[level], i, beta)
            V[i, k_re_idx] = 1. * weight / np.sum(weight)

        if k2s[level] != 1:
            V_qe = np.zeros_like(V, dtype=np.float32)
            for i in range(all_num):
                V_qe[i, :] = np.mean(V[initial_rank[i, :k2s[level]], :], axis=0)
            V = V_qe
            del V_qe

        invIndex = []
        for i in range(gallery_num):
            invIndex.append(np.where(V[:, i] != 0)[0])

        jaccard_dist = np.zeros_like(original_dist[:query_num, ], dtype=np.float32)
        for i in range(query_num):
            temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
            indNonZero = np.where(V[i, :] != 0)[0]
            indImages = [invIndex[ind] for ind in indNonZero]

            for j in range(len(indNonZero)):
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])

            jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

        jaccard_dist = jaccard_dist * (1 - lambda_value) + original_dist[:query_num, ] * lambda_value
        # if np.any(jaccard_dist > 0):
        #     print("yes")
        del V

        if level == 0:
            final_dist = jaccard_dist
            final_rank = np.argsort(jaccard_dist)
            idss0 = final_rank[:, :k1s[0] + 1]
            del final_rank
        elif level == 1:
            for i in range(query_num):
                ids_i = np.ones([all_num], dtype=bool)
                ids_i[idss0[i, :]] = False
                final_dist[i, ids_i] = jaccard_dist[i, ids_i] + level

            final_rank = np.argsort(final_dist)
            idss1 = final_rank[:, :k1s[level] + 1]
            del final_rank
        else:
            for i in range(query_num):
                ids_i = np.ones([all_num], dtype=bool)
                idss = np.concatenate((idss0, idss1), axis=1)
                ids_i[idss[i, :]] = False
                final_dist[i, ids_i] = jaccard_dist[i, ids_i] + level
        del jaccard_dist

    del initial_rank
    del original_dist

    final_dist = final_dist[:query_num, query_num:]

    return final_dist


# rerank
# def re_ranking_in(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3, beta=1.):

#     # The following naming, e.g. gallery_num, is different from outer scope.
#     # Don't care about it.

#     original_dist = np.concatenate(
#         [
#             np.concatenate([q_q_dist, q_g_dist], axis=1),
#             np.concatenate([q_g_dist.T, g_g_dist], axis=1)
#         ],
#         axis=0
#     )
#     # original_dist = np.power(original_dist, 2).astype(np.float32)
#     original_dist = np.power(original_dist, 2).astype(np.float32)
#     original_dist = np.transpose(
#         1. * original_dist / np.max(original_dist, axis=0)
#     )
#     V = np.zeros_like(original_dist).astype(np.float32)
#     initial_rank = np.argsort(original_dist).astype(np.int32)

#     query_num = q_g_dist.shape[0]
#     gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
#     all_num = gallery_num

#     for i in range(all_num):
#         k_re_idx, weight = build_expanded_reciprocal_index(
#             initial_rank, original_dist[i], k1, i, beta
#         )
#         V[i, k_re_idx] = 1. * weight / np.sum(weight)

#     original_dist = original_dist[:query_num, ]
#     if k2 != 1:
#         V_qe = np.zeros_like(V, dtype=np.float32)
#         for i in range(all_num):
#             V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
#         V = V_qe
#         del V_qe
#     del initial_rank
#     invIndex = []
#     for i in range(gallery_num):
#         invIndex.append(np.where(V[:, i] != 0)[0])

#     jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

#     for i in range(query_num):
#         temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
#         indNonZero = np.where(V[i, :] != 0)[0]
#         indImages = []
#         indImages = [invIndex[ind] for ind in indNonZero]
#         for j in range(len(indNonZero)):
#             temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
#                 V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
#             )
#         jaccard_dist[i] = 1 - temp_min / (2.-temp_min)

#     final_dist = jaccard_dist * (1-lambda_value) + original_dist * lambda_value
#     # final_dist = original_dist
#     del original_dist
#     del V
#     del jaccard_dist
#     final_dist = final_dist[:query_num, query_num:]
#     return final_dist