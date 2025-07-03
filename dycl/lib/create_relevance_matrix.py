import torch


def create_relevance_matrix(label_matrix, relevance):
    label_matrix = label_matrix.to(torch.int64)
    label_matrix = label_matrix.to(relevance.device)
    return torch.gather(relevance, 1, label_matrix)
