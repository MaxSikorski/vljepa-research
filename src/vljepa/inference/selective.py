"""
Selective Decoding for VL-JEPA.

The key efficiency innovation: VL-JEPA only invokes the text decoder
when generative output is actually needed. For most tasks (classification,
retrieval, VQA with known answer set), inference happens purely in
embedding space — no decoder required.

For video tasks, Ward agglomerative clustering identifies which temporal
segments are relevant to a query, and decoding is applied only to those
segments. This achieves 2.85x reduction in decoding operations.

Algorithm:
1. Extract visual features for all temporal segments
2. Compute embedding similarity between each segment and the query
3. Apply Ward clustering to group temporally similar segments
4. Select top-k clusters by relevance score
5. Decode only selected segments
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster, ward
from scipy.spatial.distance import pdist


def selective_decode(
    segment_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    num_segments_to_decode: int | None = None,
    reduction_factor: float = 2.85,
) -> torch.Tensor:
    """
    Select which video segments to decode based on relevance.

    Args:
        segment_embeddings: (T, D) embeddings for each temporal segment
        query_embedding: (D,) query embedding
        num_segments_to_decode: explicit number, or computed from reduction_factor
        reduction_factor: target speedup (default 2.85x per paper)

    Returns:
        selected_indices: (K,) indices of segments to decode
    """
    T = segment_embeddings.shape[0]

    if num_segments_to_decode is None:
        num_segments_to_decode = max(1, int(T / reduction_factor))

    # Compute relevance scores (cosine similarity with query)
    segment_embeddings = F.normalize(segment_embeddings, dim=-1)
    query_embedding = F.normalize(query_embedding, dim=-1)
    relevance = segment_embeddings @ query_embedding  # (T,)

    if T <= num_segments_to_decode:
        return torch.arange(T, device=segment_embeddings.device)

    # Ward agglomerative clustering on embeddings
    embeddings_np = segment_embeddings.cpu().numpy()
    distances = pdist(embeddings_np, metric="cosine")
    linkage = ward(distances)

    # Form clusters
    num_clusters = min(num_segments_to_decode, T)
    cluster_labels = fcluster(linkage, t=num_clusters, criterion="maxclust")

    # Score each cluster by max relevance of its members
    cluster_scores = {}
    for i in range(T):
        cluster_id = cluster_labels[i]
        score = relevance[i].item()
        if cluster_id not in cluster_scores or score > cluster_scores[cluster_id][1]:
            cluster_scores[cluster_id] = (i, score)

    # Select top-k clusters by score
    sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1][1], reverse=True)
    selected = sorted_clusters[:num_segments_to_decode]

    # Collect all segment indices from selected clusters
    selected_cluster_ids = {c[0] for c in selected}
    selected_indices = [
        i for i in range(T) if cluster_labels[i] in selected_cluster_ids
    ]

    return torch.tensor(sorted(selected_indices), device=segment_embeddings.device)


def batch_selective_decode(
    segment_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    reduction_factor: float = 2.85,
) -> list[torch.Tensor]:
    """
    Batch version of selective decoding.

    Args:
        segment_embeddings: (B, T, D)
        query_embeddings: (B, D)

    Returns:
        List of selected index tensors, one per batch element
    """
    B = segment_embeddings.shape[0]
    results = []

    for i in range(B):
        indices = selective_decode(
            segment_embeddings[i],
            query_embeddings[i],
            reduction_factor=reduction_factor,
        )
        results.append(indices)

    return results
