"""
Utility functions for the Two-Tower model.
"""

import random
import numpy as np
import torch
import os
from typing import List, Tuple, Dict


def setup_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device=None):
    """Get the device to use for training."""
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def create_ranking_metrics(scores, ground_truth, k_values=[1, 5, 10, 20, 50, 100]):
    """
    Compute ranking metrics for recommendation.
    
    Args:
        scores: Predicted scores for each item
        ground_truth: Ground truth relevance for each item (binary)
        k_values: List of k values for precision/recall@k
    
    Returns:
        metrics: Dictionary of ranking metrics
    """
    metrics = {}
    
    # Sort by score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_ground_truth = ground_truth[sorted_indices]
    
    # Calculate metrics
    for k in k_values:
        # Limit to top k results
        top_k = sorted_ground_truth[:k]
        
        # Precision@k
        precision = np.mean(top_k) if len(top_k) > 0 else 0.0
        metrics[f'precision@{k}'] = precision
        
        # Recall@k
        recall = np.sum(top_k) / np.sum(ground_truth) if np.sum(ground_truth) > 0 else 0.0
        metrics[f'recall@{k}'] = recall
        
        # NDCG@k
        dcg = np.sum(top_k / np.log2(np.arange(2, len(top_k) + 2)))
        
        # Ideal DCG
        idcg = np.sum(np.sort(ground_truth)[::-1][:k] / np.log2(np.arange(2, min(k, np.sum(ground_truth)) + 2)))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics[f'ndcg@{k}'] = ndcg
    
    return metrics


def batch_cosine_similarity(a, b):
    """
    Compute cosine similarity between two batches of vectors.
    
    Args:
        a: Tensor of shape (batch_size, vector_dim)
        b: Tensor of shape (batch_size, vector_dim)
    
    Returns:
        Tensor of shape (batch_size,) with cosine similarities
    """
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.sum(a_norm * b_norm, dim=1)


def generate_hard_negatives(user_embeddings, place_embeddings, positive_edges, num_negatives=5):
    """
    Generate hard negative samples using embeddings.
    
    Args:
        user_embeddings: Dictionary of user_id -> embedding
        place_embeddings: Dictionary of place_id -> embedding
        positive_edges: List of (user_id, place_id) pairs representing positive interactions
        num_negatives: Number of negative samples per positive
    
    Returns:
        List of (user_id, place_id, 0.0) tuples for negative samples
    """
    negative_samples = []
    
    # Convert embeddings to tensors for efficient computation
    user_ids = list(user_embeddings.keys())
    place_ids = list(place_embeddings.keys())
    
    user_embedding_tensor = torch.stack([user_embeddings[uid] for uid in user_ids])
    place_embedding_tensor = torch.stack([place_embeddings[pid] for pid in place_ids])
    
    # Normalize embeddings
    user_embedding_tensor = torch.nn.functional.normalize(user_embedding_tensor, p=2, dim=1)
    place_embedding_tensor = torch.nn.functional.normalize(place_embedding_tensor, p=2, dim=1)
    
    # Create set of positive edges for fast lookup
    positive_set = set((u, p) for u, p in positive_edges)
    
    # Process each user
    for i, user_id in enumerate(user_ids):
        # Get user embedding
        user_emb = user_embedding_tensor[i].unsqueeze(0)
        
        # Compute similarity to all places
        similarities = torch.matmul(user_emb, place_embedding_tensor.t()).squeeze()
        
        # Get top similar places
        top_indices = torch.argsort(similarities, descending=True)
        
        # Find hard negatives (high similarity but not interacted)
        neg_count = 0
        for idx in top_indices:
            place_id = place_ids[idx]
            
            # Check if this is a positive interaction
            if (user_id, place_id) not in positive_set:
                negative_samples.append((user_id, place_id, 0.0))
                neg_count += 1
                
                # Stop when we have enough negatives for this user
                if neg_count >= num_negatives:
                    break
    
    return negative_samples


def compare_embeddings(old_embeddings, new_embeddings):
    """
    Compare old and new embeddings to measure distribution shifts.
    
    Args:
        old_embeddings: Dictionary of entity_id -> old embedding
        new_embeddings: Dictionary of entity_id -> new embedding
    
    Returns:
        metrics: Dictionary of comparison metrics
    """
    metrics = {}
    
    # Find common entity IDs
    common_ids = set(old_embeddings.keys()) & set(new_embeddings.keys())
    
    if not common_ids:
        return {'error': 'No common entities found'}
    
    # Compute average cosine similarity
    similarities = []
    
    for entity_id in common_ids:
        old_emb = old_embeddings[entity_id]
        new_emb = new_embeddings[entity_id]
        
        # Normalize embeddings
        old_emb = old_emb / torch.norm(old_emb)
        new_emb = new_emb / torch.norm(new_emb)
        
        # Compute cosine similarity
        similarity = torch.dot(old_emb, new_emb).item()
        similarities.append(similarity)
    
    metrics['avg_similarity'] = np.mean(similarities)
    metrics['min_similarity'] = np.min(similarities)
    metrics['max_similarity'] = np.max(similarities)
    metrics['std_similarity'] = np.std(similarities)
    
    # Compute average L2 distance
    distances = []
    
    for entity_id in common_ids:
        old_emb = old_embeddings[entity_id]
        new_emb = new_embeddings[entity_id]
        
        # Compute L2 distance
        distance = torch.norm(old_emb - new_emb).item()
        distances.append(distance)
    
    metrics['avg_distance'] = np.mean(distances)
    metrics['max_distance'] = np.max(distances)
    
    return metrics