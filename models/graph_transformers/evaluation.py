"""
Evaluation utilities for the HGT model.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm
import torch.nn.functional as F
from utils import create_negative_edges


def evaluate_model(model, graph, device, task_types=None, neg_ratio=1.0):
    """
    Evaluate model on link prediction tasks.
    
    Args:
        model: Trained HGT model
        graph: Full graph data
        device: Device to use
        task_types: List of task types to evaluate ('user_place', 'user_user', etc.)
        neg_ratio: Ratio of negative to positive edges
    
    Returns:
        metrics: Dictionary of evaluation metrics for each task
    """
    model.eval()
    metrics = {}
    
    # Default to all available tasks if none specified
    if task_types is None:
        task_types = []
        if ('user', 'visited', 'place') in graph.edge_index_dict:
            task_types.append('user_place')
        if ('user', 'friends_with', 'user') in graph.edge_index_dict:
            task_types.append('user_user')
    
    # Get node embeddings
    with torch.no_grad():
        graph = graph.to(device)
        out_dict = model(graph)
    
    # Evaluate user-place prediction
    if 'user_place' in task_types:
        if ('user', 'visited', 'place') in graph.edge_index_dict:
            # Get positive edges
            edge_index = graph['user', 'visited', 'place'].edge_index
            
            with torch.no_grad():
                # Compute scores for positive edges
                src, dst = edge_index
                pos_score = model.predict_user_place(
                    out_dict['user'][src], 
                    out_dict['place'][dst]
                ).cpu().numpy()
                
                # Generate negative edges
                num_neg = int(edge_index.size(1) * neg_ratio)
                neg_src, neg_dst = create_negative_edges(
                    edge_index, 
                    graph['user'].num_nodes,
                    graph['place'].num_nodes,
                    num_samples=num_neg
                )
                
                # Compute scores for negative edges
                neg_score = model.predict_user_place(
                    out_dict['user'][neg_src], 
                    out_dict['place'][neg_dst]
                ).cpu().numpy()
            
            # Combine positive and negative scores with labels
            scores = np.concatenate([pos_score, neg_score])
            labels = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
            
            # Compute metrics
            auc = roc_auc_score(labels, scores)
            ap = average_precision_score(labels, scores)
            
            metrics['user_place'] = {
                'auc': auc,
                'ap': ap
            }
    
    # Evaluate user-user prediction
    if 'user_user' in task_types:
        if ('user', 'friends_with', 'user') in graph.edge_index_dict:
            # Get positive edges
            edge_index = graph['user', 'friends_with', 'user'].edge_index
            
            with torch.no_grad():
                # Compute scores for positive edges
                src, dst = edge_index
                pos_score = model.predict_user_user(
                    out_dict['user'][src], 
                    out_dict['user'][dst]
                ).cpu().numpy()
                
                # Generate negative edges
                num_neg = int(edge_index.size(1) * neg_ratio)
                neg_src, neg_dst = create_negative_edges(
                    edge_index, 
                    graph['user'].num_nodes,
                    graph['user'].num_nodes,
                    num_samples=num_neg
                )
                
                # Compute scores for negative edges
                neg_score = model.predict_user_user(
                    out_dict['user'][neg_src], 
                    out_dict['user'][neg_dst]
                ).cpu().numpy()
            
            # Combine positive and negative scores with labels
            scores = np.concatenate([pos_score, neg_score])
            labels = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
            
            # Compute metrics
            auc = roc_auc_score(labels, scores)
            ap = average_precision_score(labels, scores)
            
            metrics['user_user'] = {
                'auc': auc,
                'ap': ap
            }
    
    return metrics


def evaluate_place_recommendations(model, graph, user_ids, device, k=10):
    """
    Evaluate place recommendations for specific users.
    
    Args:
        model: Trained HGT model
        graph: Full graph data
        user_ids: List of user IDs to evaluate
        device: Device to use
        k: Number of recommendations to return
    
    Returns:
        precision@k, recall@k, and ndcg@k for each user
    """
    model.eval()
    metrics = {
        'precision@k': [],
        'recall@k': [],
        'ndcg@k': []
    }
    
    # Get node embeddings
    with torch.no_grad():
        graph = graph.to(device)
        out_dict = model(graph)
    
    # Get all place embeddings
    place_embeddings = out_dict['place']
    
    for user_id in user_ids:
        # Get user embedding
        user_embedding = out_dict['user'][user_id:user_id+1]
        
        # Compute scores for all places
        with torch.no_grad():
            scores = model.predict_user_place(
                user_embedding.repeat(place_embeddings.size(0), 1),
                place_embeddings
            ).squeeze()
        
        # Get ground truth - places the user has visited
        visited_places = set()
        if ('user', 'visited', 'place') in graph.edge_index_dict:
            edge_index = graph['user', 'visited', 'place'].edge_index
            mask = edge_index[0] == user_id
            visited_places = set(edge_index[1, mask].cpu().numpy())
        
        # Get top-k recommendations (excluding visited places)
        # First set scores of visited places to -inf
        scores_clone = scores.clone()
        for place_id in visited_places:
            scores_clone[place_id] = -float('inf')
        
        # Get top-k recommendations
        topk_scores, topk_indices = torch.topk(scores_clone, k=k)
        recommended_places = set(topk_indices.cpu().numpy())
        
        # Compute metrics
        # For this evaluation, we'll consider all visited places as relevant
        # In a real system, you might want to filter for highly rated visits
        
        # Precision@k: What fraction of recommended items are relevant?
        precision = len(visited_places.intersection(recommended_places)) / k
        
        # Recall@k: What fraction of relevant items are recommended?
        if visited_places:
            recall = len(visited_places.intersection(recommended_places)) / len(visited_places)
        else:
            recall = 0.0
        
        # NDCG@k: Normalized Discounted Cumulative Gain
        dcg = 0.0
        idcg = 0.0
        
        for i, place_id in enumerate(topk_indices.cpu().numpy()):
            if place_id in visited_places:
                dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed
        
        # Ideal DCG - if all top-k items were relevant
        for i in range(min(k, len(visited_places))):
            idcg += 1.0 / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        # Add metrics for this user
        metrics['precision@k'].append(precision)
        metrics['recall@k'].append(recall)
        metrics['ndcg@k'].append(ndcg)
    
    # Average metrics across users
    for key in metrics:
        metrics[key] = np.mean(metrics[key])
    
    return metrics


def evaluate_cold_start(model, graph, device, test_edges, edge_type=('user', 'visited', 'place')):
    """
    Evaluate model performance on cold-start scenarios.
    
    Args:
        model: Trained HGT model
        graph: Full graph data
        device: Device to use
        test_edges: Dictionary of test edges for different cold-start scenarios
        edge_type: Type of edge to evaluate
    
    Returns:
        Dictionary of metrics for each cold-start scenario
    """
    model.eval()
    metrics = {}
    
    # Get node embeddings
    with torch.no_grad():
        graph = graph.to(device)
        out_dict = model(graph)
    
    # Evaluate each cold-start scenario
    for scenario, edges in test_edges.items():
        pos_src, pos_dst = edges['positive']
        neg_src, neg_dst = edges['negative']
        
        with torch.no_grad():
            # Compute scores for positive edges
            if edge_type[0] == 'user' and edge_type[2] == 'place':
                pos_score = model.predict_user_place(
                    out_dict['user'][pos_src], 
                    out_dict['place'][pos_dst]
                ).cpu().numpy()
                
                # Compute scores for negative edges
                neg_score = model.predict_user_place(
                    out_dict['user'][neg_src], 
                    out_dict['place'][neg_dst]
                ).cpu().numpy()
            elif edge_type[0] == 'user' and edge_type[2] == 'user':
                pos_score = model.predict_user_user(
                    out_dict['user'][pos_src], 
                    out_dict['user'][pos_dst]
                ).cpu().numpy()
                
                # Compute scores for negative edges
                neg_score = model.predict_user_user(
                    out_dict['user'][neg_src], 
                    out_dict['user'][neg_dst]
                ).cpu().numpy()
        
        # Combine positive and negative scores with labels
        scores = np.concatenate([pos_score, neg_score])
        labels = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
        
        # Compute metrics
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        
        metrics[scenario] = {
            'auc': auc,
            'ap': ap
        }
    
    return metrics