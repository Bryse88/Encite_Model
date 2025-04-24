"""
Evaluation utilities for the Two-Tower model.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import logging

from utils import create_ranking_metrics


logger = logging.getLogger('two_tower_evaluation')


def evaluate_recommendations(model, user_features, place_features, test_interactions, top_k=[5, 10, 20], device='cpu'):
    """
    Evaluate model on recommendation task.
    
    Args:
        model: Trained Two-Tower model
        user_features: Dictionary of user_id -> feature tensor
        place_features: Dictionary of place_id -> feature tensor
        test_interactions: List of (user_id, place_id, label) tuples
        top_k: List of k values for top-k metrics
        device: Device to use
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    # Group interactions by user
    user_positives = {}
    for user_id, place_id, label in test_interactions:
        if label > 0:
            if user_id not in user_positives:
                user_positives[user_id] = []
            user_positives[user_id].append(place_id)
    
    # Metrics to track
    metrics = {
        f'precision@{k}': [] for k in top_k
    }
    metrics.update({
        f'recall@{k}': [] for k in top_k
    })
    metrics.update({
        f'ndcg@{k}': [] for k in top_k
    })
    
    # Process each user
    for user_id, positive_places in tqdm(user_positives.items(), desc="Evaluating users"):
        # Skip users with no positive places
        if not positive_places or user_id not in user_features:
            continue
        
        # Get user embedding
        user_feature = user_features[user_id].to(device)
        with torch.no_grad():
            user_embedding = model.get_user_embedding(user_feature.unsqueeze(0)).squeeze(0)
        
        # Compute scores for all places
        all_places = list(place_features.keys())
        all_scores = []
        
        # Process in batches to avoid OOM
        batch_size = 1000
        for i in range(0, len(all_places), batch_size):
            batch_places = all_places[i:i + batch_size]
            batch_features = torch.stack([place_features[pid] for pid in batch_places]).to(device)
            
            with torch.no_grad():
                place_embeddings = model.get_place_embedding(batch_features)
                batch_scores = torch.matmul(user_embedding.unsqueeze(0), place_embeddings.t()).squeeze(0)
            
            all_scores.append(batch_scores.cpu().numpy())
        
        # Combine scores
        all_scores = np.concatenate(all_scores)
        
        # Create ground truth array
        ground_truth = np.zeros(len(all_places))
        for pid in positive_places:
            if pid in all_places:
                idx = all_places.index(pid)
                ground_truth[idx] = 1
        
        # Compute ranking metrics
        user_metrics = create_ranking_metrics(all_scores, ground_truth, k_values=top_k)
        
        # Add to overall metrics
        for k in top_k:
            metrics[f'precision@{k}'].append(user_metrics[f'precision@{k}'])
            metrics[f'recall@{k}'].append(user_metrics[f'recall@{k}'])
            metrics[f'ndcg@{k}'].append(user_metrics[f'ndcg@{k}'])
    
    # Compute average metrics
    for k in top_k:
        metrics[f'precision@{k}'] = np.mean(metrics[f'precision@{k}'])
        metrics[f'recall@{k}'] = np.mean(metrics[f'recall@{k}'])
        metrics[f'ndcg@{k}'] = np.mean(metrics[f'ndcg@{k}'])
    
    return metrics


def evaluate_cold_start(model, user_features, place_features, test_interactions, cold_start_type='user', device='cpu'):
    """
    Evaluate model on cold-start scenarios.
    
    Args:
        model: Trained Two-Tower model
        user_features: Dictionary of user_id -> feature tensor
        place_features: Dictionary of place_id -> feature tensor
        test_interactions: List of (user_id, place_id, label) tuples
        cold_start_type: Type of cold-start scenario ('user', 'place', or 'both')
        device: Device to use
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    # Identify cold-start entities (those with few interactions)
    if cold_start_type in ['user', 'both']:
        # Count interactions per user
        user_counts = {}
        for user_id, _, _ in test_interactions:
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
        
        # Identify cold-start users (bottom 25% by interaction count)
        counts = sorted(user_counts.values())
        threshold = counts[min(len(counts) // 4, len(counts) - 1)]
        cold_users = {uid for uid, count in user_counts.items() if count <= threshold}
        
        logger.info(f"Identified {len(cold_users)} cold-start users (≤{threshold} interactions)")
    else:
        cold_users = set()
    
    if cold_start_type in ['place', 'both']:
        # Count interactions per place
        place_counts = {}
        for _, place_id, _ in test_interactions:
            place_counts[place_id] = place_counts.get(place_id, 0) + 1
        
        # Identify cold-start places (bottom 25% by interaction count)
        counts = sorted(place_counts.values())
        threshold = counts[min(len(counts) // 4, len(counts) - 1)]
        cold_places = {pid for pid, count in place_counts.items() if count <= threshold}
        
        logger.info(f"Identified {len(cold_places)} cold-start places (≤{threshold} interactions)")
    else:
        cold_places = set()
    
    # Filter interactions for cold-start scenarios
    cold_interactions = []
    for user_id, place_id, label in test_interactions:
        if cold_start_type == 'user' and user_id in cold_users:
            cold_interactions.append((user_id, place_id, label))
        elif cold_start_type == 'place' and place_id in cold_places:
            cold_interactions.append((user_id, place_id, label))
        elif cold_start_type == 'both' and (user_id in cold_users or place_id in cold_places):
            cold_interactions.append((user_id, place_id, label))
    
    logger.info(f"Evaluating {len(cold_interactions)} cold-start interactions")
    
    # Skip if no cold-start interactions
    if not cold_interactions:
        return {"error": "No cold-start interactions found"}
    
    # Prepare data for evaluation
    all_scores = []
    all_labels = []
    
    # Process each interaction
    for user_id, place_id, label in tqdm(cold_interactions, desc="Evaluating cold-start"):
        # Skip if features not available
        if user_id not in user_features or place_id not in place_features:
            continue
        
        # Get embeddings
        user_feature = user_features[user_id].to(device)
        place_feature = place_features[place_id].to(device)
        
        with torch.no_grad():
            user_embedding = model.get_user_embedding(user_feature.unsqueeze(0)).squeeze(0)
            place_embedding = model.get_place_embedding(place_feature.unsqueeze(0)).squeeze(0)
            
            # Compute score
            score = torch.dot(user_embedding, place_embedding).item()
        
        all_scores.append(score)
        all_labels.append(label)
    
    # Compute metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    
    return {
        'auc': auc,
        'ap': ap,
        'num_interactions': len(all_scores)
    }


def evaluate_embedding_variance(model, features, entity_type='user', num_runs=5, device='cpu'):
    """
    Evaluate the variance of embeddings due to dropout.
    This helps assess the model's uncertainty.
    
    Args:
        model: Trained Two-Tower model
        features: Dictionary of entity_id -> feature tensor
        entity_type: Type of entity ('user' or 'place')
        num_runs: Number of forward passes to perform
        device: Device to use
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.train()  # Set to train mode to enable dropout
    model = model.to(device)
    
    # Sample entities to evaluate
    entity_ids = list(features.keys())
    if len(entity_ids) > 100:
        sampled_ids = np.random.choice(entity_ids, 100, replace=False)
    else:
        sampled_ids = entity_ids
    
    # Collect embeddings from multiple runs
    all_embeddings = {eid: [] for eid in sampled_ids}
    
    for _ in range(num_runs):
        for eid in sampled_ids:
            feature = features[eid].to(device)
            
            with torch.no_grad():
                if entity_type == 'user':
                    embedding = model.get_user_embedding(feature.unsqueeze(0)).squeeze(0)
                else:
                    embedding = model.get_place_embedding(feature.unsqueeze(0)).squeeze(0)
            
            all_embeddings[eid].append(embedding.cpu())
    
    # Compute variance metrics
    cosine_sims = []
    euclidean_dists = []
    
    for eid in sampled_ids:
        embs = all_embeddings[eid]
        
        # Compute pairwise similarities and distances
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                # Cosine similarity
                sim = torch.nn.functional.cosine_similarity(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item()
                cosine_sims.append(sim)
                
                # Euclidean distance
                dist = torch.norm(embs[i] - embs[j]).item()
                euclidean_dists.append(dist)
    
    # Compute statistics
    metrics = {
        'mean_cosine_sim': np.mean(cosine_sims),
        'std_cosine_sim': np.std(cosine_sims),
        'min_cosine_sim': np.min(cosine_sims),
        'mean_euclidean_dist': np.mean(euclidean_dists),
        'std_euclidean_dist': np.std(euclidean_dists),
        'max_euclidean_dist': np.max(euclidean_dists)
    }
    
    return metrics