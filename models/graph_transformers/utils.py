"""
Utility functions for the HGT model.
"""

import random
import numpy as np
import torch
import os
from typing import Tuple, List


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


def create_negative_edges(
    edge_index: torch.Tensor,
    num_nodes_src: int,
    num_nodes_dst: int,
    num_samples: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create negative edges by sampling from non-existing edges.
    
    Args:
        edge_index: Positive edge indices
        num_nodes_src: Number of source nodes
        num_nodes_dst: Number of destination nodes
        num_samples: Number of negative samples to generate
    
    Returns:
        src, dst: Source and destination indices for negative edges
    """
    # Convert edge_index to set of tuples for fast lookup
    edge_set = set(map(tuple, edge_index.t().tolist()))
    
    # Generate negative samples
    neg_src = []
    neg_dst = []
    
    while len(neg_src) < num_samples:
        # Sample random source and destination nodes
        src = torch.randint(0, num_nodes_src, (1,)).item()
        dst = torch.randint(0, num_nodes_dst, (1,)).item()
        
        # Check if this edge already exists
        if (src, dst) not in edge_set:
            neg_src.append(src)
            neg_dst.append(dst)
            edge_set.add((src, dst))  # Add to set to avoid duplicates
    
    return torch.tensor(neg_src, device=edge_index.device), torch.tensor(neg_dst, device=edge_index.device)


def create_train_val_test_split(edge_index, val_ratio=0.1, test_ratio=0.1):
    """
    Split edges into train, validation, and test sets.
    
    Args:
        edge_index: Edge indices
        val_ratio: Ratio of edges to use for validation
        test_ratio: Ratio of edges to use for testing
    
    Returns:
        train_edge_index, val_edge_index, test_edge_index
    """
    num_edges = edge_index.size(1)
    
    # Create random permutation of edges
    perm = torch.randperm(num_edges)
    
    # Calculate split indices
    test_size = int(num_edges * test_ratio)
    val_size = int(num_edges * val_ratio)
    train_size = num_edges - test_size - val_size
    
    # Split edges
    train_indices = perm[:train_size]
    val_indices = perm[train_size:train_size + val_size]
    test_indices = perm[train_size + val_size:]
    
    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]
    
    return train_edge_index, val_edge_index, test_edge_index


def create_cold_start_splits(graph, cold_node_ratio=0.1):
    """
    Create splits for cold-start evaluation.
    
    Args:
        graph: Full graph data
        cold_node_ratio: Ratio of nodes to consider "cold"
    
    Returns:
        Dictionary of test edges for different cold-start scenarios
    """
    results = {}
    
    # Case 1: Cold-start places (existing users, new places)
    if ('user', 'visited', 'place') in graph.edge_index_dict:
        edge_index = graph['user', 'visited', 'place'].edge_index
        
        # Identify "cold" places - those with few interactions
        place_counts = torch.bincount(edge_index[1], minlength=graph['place'].num_nodes)
        percentile = int(cold_node_ratio * 100)
        threshold = torch.kthvalue(place_counts, percentile).values.item()
        
        cold_places = torch.where(place_counts <= threshold)[0]
        
        # Get edges involving cold places
        mask = torch.isin(edge_index[1], cold_places)
        cold_edges = edge_index[:, mask]
        
        # Split into positive and negative examples
        train_edges, val_edges, test_edges = create_train_val_test_split(cold_edges)
        
        # Create negative examples for test edges
        neg_src, neg_dst = create_negative_edges(
            test_edges, 
            graph['user'].num_nodes,
            graph['place'].num_nodes,
            num_samples=test_edges.size(1)
        )
        
        results['cold_places'] = {
            'positive': (test_edges[0], test_edges[1]),
            'negative': (neg_src, neg_dst)
        }
    
    # Case 2: Cold-start users (new users, existing places)
    if ('user', 'visited', 'place') in graph.edge_index_dict:
        edge_index = graph['user', 'visited', 'place'].edge_index
        
        # Identify "cold" users - those with few interactions
        user_counts = torch.bincount(edge_index[0], minlength=graph['user'].num_nodes)
        percentile = int(cold_node_ratio * 100)
        threshold = torch.kthvalue(user_counts, percentile).values.item()
        
        cold_users = torch.where(user_counts <= threshold)[0]
        
        # Get edges involving cold users
        mask = torch.isin(edge_index[0], cold_users)
        cold_edges = edge_index[:, mask]
        
        # Split into positive and negative examples
        train_edges, val_edges, test_edges = create_train_val_test_split(cold_edges)
        
        # Create negative examples for test edges
        neg_src, neg_dst = create_negative_edges(
            test_edges, 
            graph['user'].num_nodes,
            graph['place'].num_nodes,
            num_samples=test_edges.size(1)
        )
        
        results['cold_users'] = {
            'positive': (test_edges[0], test_edges[1]),
            'negative': (neg_src, neg_dst)
        }
    
    return results


def save_embeddings_to_firestore(embeddings, node_mapping, collection, batch_size=500):
    """
    Save embeddings to Firestore.
    
    Args:
        embeddings: Node embeddings
        node_mapping: Mapping from node indices to Firestore IDs
        collection: Firestore collection name
        batch_size: Number of documents to update in each batch
    """
    from google.cloud import firestore
    
    db = firestore.Client()
    batch = db.batch()
    count = 0
    total = 0
    
    inv_mapping = {v: k for k, v in node_mapping.items()}
    
    for idx in range(embeddings.shape[0]):
        if idx in inv_mapping:
            doc_id = inv_mapping[idx]
            doc_ref = db.collection(collection).document(doc_id)
            
            # Convert embedding to list and store
            embedding_list = embeddings[idx].tolist()
            batch.update(doc_ref, {'embedding': embedding_list})
            
            count += 1
            total += 1
            
            # Commit batch when it reaches the batch size
            if count >= batch_size:
                batch.commit()
                batch = db.batch()
                count = 0
                print(f"Saved {total} embeddings so far")
    
    # Commit any remaining updates
    if count > 0:
        batch.commit()
    
    print(f"Saved a total of {total} embeddings to Firestore")


def visualize_embeddings(embeddings, labels, output_path=None):
    """
    Visualize embeddings using t-SNE.
    
    Args:
        embeddings: Node embeddings
        labels: Node labels
        output_path: Path to save the visualization
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=labels, cmap='tab10', alpha=0.7)
        
        # Add legend
        unique_labels = np.unique(labels)
        plt.legend(handles=scatter.legend_elements()[0], 
                   labels=[f'Class {i}' for i in unique_labels])
        
        plt.title('t-SNE Visualization of Node Embeddings')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        # Save or show the plot
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install matplotlib and scikit-learn to use this function.")