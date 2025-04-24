"""
Training script for the Two-Tower model.
"""

import os
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import logging

from model import create_model
from dataset import (
    UserPlaceInteractionDataset, 
    FirestoreDataLoader, 
    create_train_val_test_split,
    create_context_features
)
from utils import setup_seeds, get_device


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('two_tower_train')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Two-Tower model for Encite')
    parser.add_argument('--config', type=str, default='configs/two_tower_config.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def in_batch_train_step(model, batch, optimizer, device, use_context=False):
    """
    Training step using in-batch negatives.
    
    Args:
        model: Two-Tower model
        batch: Batch of user-place pairs
        optimizer: Optimizer
        device: Device to use
        use_context: Whether to use context features
    
    Returns:
        loss: Training loss for this batch
    """
    optimizer.zero_grad()
    
    # Move batch to device
    user_features = batch['user_features'].to(device)
    place_features = batch['pos_place_features'].to(device)
    
    # Get similarity scores
    if use_context and 'pos_context' in batch:
        context_features = batch['pos_context'].to(device)
        scores = model(user_features, place_features, context_features)
    else:
        scores = model(user_features, place_features)
    
    # For in-batch training, we treat diagonal elements as positive examples
    # and off-diagonal elements as negative examples
    batch_size = user_features.size(0)
    
    # Create target matrix: 1 on diagonal, 0 elsewhere
    targets = torch.eye(batch_size).to(device)
    
    # Compute cross-entropy loss
    loss = nn.BCEWithLogitsLoss()(scores, targets)
    
    # Backward and optimize
    loss.backward()
    optimizer.step()
    
    return loss.item()


def explicit_train_step(model, batch, optimizer, device, use_context=False):
    """
    Training step using explicit negatives.
    
    Args:
        model: Two-Tower model
        batch: Batch of user-place pairs with explicit negatives
        optimizer: Optimizer
        device: Device to use
        use_context: Whether to use context features
    
    Returns:
        loss: Training loss for this batch
    """
    optimizer.zero_grad()
    
    # Move batch to device
    user_features = batch['user_features'].to(device)
    pos_place_features = batch['pos_place_features'].to(device)
    neg_place_features = batch['neg_place_features'].to(device)
    
    batch_size = user_features.size(0)
    neg_samples = neg_place_features.size(1)
    
    # Reshape for batch processing
    user_features_expanded = user_features.unsqueeze(1).expand(-1, neg_samples + 1, -1)
    user_features_flat = user_features_expanded.reshape(-1, user_features.size(1))
    
    # Combine positive and negative place features
    all_place_features = torch.cat([
        pos_place_features.unsqueeze(1),
        neg_place_features
    ], dim=1)
    all_place_features_flat = all_place_features.reshape(-1, pos_place_features.size(1))
    
    # Get similarity scores
    if use_context and 'pos_context' in batch and 'neg_contexts' in batch:
        pos_context = batch['pos_context'].to(device)
        neg_contexts = batch['neg_contexts'].to(device)
        
        # Combine contexts
        all_contexts = torch.cat([
            pos_context.unsqueeze(1),
            neg_contexts
        ], dim=1)
        all_contexts_flat = all_contexts.reshape(-1, pos_context.size(1))
        
        scores = model(user_features_flat, all_place_features_flat, all_contexts_flat)
    else:
        scores = model(user_features_flat, all_place_features_flat)
    
    # Reshape scores
    scores = scores.reshape(batch_size, neg_samples + 1)
    
    # Create targets: first column is positive (1), rest are negative (0)
    targets = torch.zeros_like(scores)
    targets[:, 0] = 1.0
    
    # Compute cross-entropy loss
    loss = nn.BCEWithLogitsLoss()(scores, targets)
    
    # Backward and optimize
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate_model(model, dataloader, device, use_context=False):
    """
    Evaluate model on validation set.
    
    Args:
        model: Two-Tower model
        dataloader: DataLoader for validation set
        device: Device to use
        use_context: Whether to use context features
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            user_features = batch['user_features'].to(device)
            pos_place_features = batch['pos_place_features'].to(device)
            neg_place_features = batch['neg_place_features'].to(device)
            
            batch_size = user_features.size(0)
            neg_samples = neg_place_features.size(1)
            
            # Reshape for batch processing
            user_features_expanded = user_features.unsqueeze(1).expand(-1, neg_samples + 1, -1)
            user_features_flat = user_features_expanded.reshape(-1, user_features.size(1))
            
            # Combine positive and negative place features
            all_place_features = torch.cat([
                pos_place_features.unsqueeze(1),
                neg_place_features
            ], dim=1)
            all_place_features_flat = all_place_features.reshape(-1, pos_place_features.size(1))
            
            # Get similarity scores
            if use_context and 'pos_context' in batch and 'neg_contexts' in batch:
                pos_context = batch['pos_context'].to(device)
                neg_contexts = batch['neg_contexts'].to(device)
                
                # Combine contexts
                all_contexts = torch.cat([
                    pos_context.unsqueeze(1),
                    neg_contexts
                ], dim=1)
                all_contexts_flat = all_contexts.reshape(-1, pos_context.size(1))
                
                scores = model(user_features_flat, all_place_features_flat, all_contexts_flat)
            else:
                scores = model(user_features_flat, all_place_features_flat)
            
            # Reshape scores
            scores = scores.reshape(batch_size, neg_samples + 1)
            
            # Convert to numpy for metric computation
            scores_np = scores.cpu().numpy()
            
            # Create labels: first is positive, rest are negative
            labels_np = np.zeros_like(scores_np)
            labels_np[:, 0] = 1.0
            
            # Flatten for metrics
            all_scores.append(scores_np.flatten())
            all_labels.append(labels_np.flatten())
    
    # Concatenate results
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    
    # Compute metrics
    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    
    # Compute precision@k and recall@k
    k_values = [1, 5, 10]
    precision_k = {}
    recall_k = {}
    
    for k in k_values:
        precision_k[k] = compute_precision_at_k(all_scores, all_labels, k)
        recall_k[k] = compute_recall_at_k(all_scores, all_labels, k)
    
    return {
        'auc': auc,
        'ap': ap,
        'precision@k': precision_k,
        'recall@k': recall_k
    }


def compute_precision_at_k(scores, labels, k):
    """Compute precision@k for binary labels."""
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    
    # Get top-k items
    top_k_indices = sorted_indices[:k]
    
    # Compute precision@k
    precision = np.mean(labels[top_k_indices])
    
    return precision


def compute_recall_at_k(scores, labels, k):
    """Compute recall@k for binary labels."""
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    
    # Get top-k items
    top_k_indices = sorted_indices[:k]
    
    # Compute recall@k
    if np.sum(labels) > 0:
        recall = np.sum(labels[top_k_indices]) / np.sum(labels)
    else:
        recall = 0.0
    
    return recall


def main():
    """Main training function."""
    args = parse_args()
    config = load_config(args.config)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup random seeds for reproducibility
    setup_seeds(config.get('seed', 42))
    
    # Setup device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load data from Firestore
    logger.info("Loading data from Firestore...")
    data_loader = FirestoreDataLoader(cache_path=config.get('data', {}).get('cache_path'))
    user_features, place_features, interactions = data_loader.load_all_data(
        limit=config.get('data', {}).get('limit')
    )
    
    # Split data
    train_interactions, val_interactions, test_interactions = create_train_val_test_split(
        interactions,
        val_ratio=config.get('data', {}).get('val_ratio', 0.1),
        test_ratio=config.get('data', {}).get('test_ratio', 0.1),
        by_user=config.get('data', {}).get('split_by_user', True)
    )
    
    logger.info(f"Split data: {len(train_interactions)} train, {len(val_interactions)} val, {len(test_interactions)} test interactions")
    
    # Create context features if needed
    use_context = config.get('model', {}).get('use_context', False)
    if use_context:
        logger.info("Creating context features...")
        context_dim = config.get('model', {}).get('context_dim', 8)
        train_contexts = create_context_features(train_interactions, context_dim)
        val_contexts = create_context_features(val_interactions, context_dim)
    else:
        train_contexts = None
        val_contexts = None
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = UserPlaceInteractionDataset(
        user_features=user_features,
        place_features=place_features,
        interactions=train_interactions,
        context_features=train_contexts,
        negative_sampling=config.get('data', {}).get('negative_sampling', 'random'),
        neg_samples_per_pos=config.get('data', {}).get('neg_samples_per_pos', 4)
    )
    
    val_dataset = UserPlaceInteractionDataset(
        user_features=user_features,
        place_features=place_features,
        interactions=val_interactions,
        context_features=val_contexts,
        negative_sampling='random',  # Always use random for validation
        neg_samples_per_pos=config.get('data', {}).get('neg_samples_per_pos', 4)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('training', {}).get('batch_size', 128),
        shuffle=True,
        num_workers=config.get('training', {}).get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('training', {}).get('batch_size', 128),
        shuffle=False,
        num_workers=config.get('training', {}).get('num_workers', 4)
    )
    
    # Create model
    logger.info("Creating model...")
    model_config = config.get('model', {})
    model_config['user_input_dim'] = data_loader.user_dim
    model_config['place_input_dim'] = data_loader.place_dim
    
    model = create_model(model_config)
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('training', {}).get('learning_rate', 0.001),
        weight_decay=config.get('training', {}).get('weight_decay', 0.0001)
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=config.get('training', {}).get('patience', 5),
        verbose=True
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_auc = 0
    
    for epoch in range(config.get('training', {}).get('num_epochs', 50)):
        logger.info(f"Epoch {epoch+1}/{config.get('training', {}).get('num_epochs', 50)}")
        
        # Training
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc="Training"):
            if config.get('training', {}).get('use_in_batch_negatives', False):
                loss = in_batch_train_step(model, batch, optimizer, device, use_context)
            else:
                loss = explicit_train_step(model, batch, optimizer, device, use_context)
            
            train_losses.append(loss)
        
        avg_train_loss = np.mean(train_losses)
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        logger.info("Evaluating on validation set...")
        val_metrics = evaluate_model(model, val_loader, device, use_context)
        
        logger.info(f"Validation AUC: {val_metrics['auc']:.4f}, AP: {val_metrics['ap']:.4f}")
        logger.info(f"Precision@1: {val_metrics['precision@k'][1]:.4f}, "
                   f"Precision@5: {val_metrics['precision@k'][5]:.4f}, "
                   f"Precision@10: {val_metrics['precision@k'][10]:.4f}")
        
        # Update learning rate
        scheduler.step(val_metrics['auc'])
        
        # Save model if it's the best so far
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, checkpoint_path)
            
            logger.info(f"New best model saved to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'epoch': config.get('training', {}).get('num_epochs', 50) - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, final_path)
    
    logger.info(f"Final model saved to {final_path}")
    
    # Final evaluation on test set
    logger.info("Creating test dataset...")
    test_contexts = create_context_features(test_interactions, context_dim) if use_context else None
    
    test_dataset = UserPlaceInteractionDataset(
        user_features=user_features,
        place_features=place_features,
        interactions=test_interactions,
        context_features=test_contexts,
        negative_sampling='random',
        neg_samples_per_pos=config.get('data', {}).get('neg_samples_per_pos', 4)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('training', {}).get('batch_size', 128),
        shuffle=False,
        num_workers=config.get('training', {}).get('num_workers', 4)
    )
    
    logger.info("Loading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device, use_context)
    
    logger.info(f"Test AUC: {test_metrics['auc']:.4f}, AP: {test_metrics['ap']:.4f}")
    logger.info(f"Precision@1: {test_metrics['precision@k'][1]:.4f}, "
               f"Precision@5: {test_metrics['precision@k'][5]:.4f}, "
               f"Precision@10: {test_metrics['precision@k'][10]:.4f}")
    
    # Save test metrics
    with open(os.path.join(args.output_dir, 'test_metrics.yaml'), 'w') as f:
        yaml.dump(test_metrics, f)


if __name__ == '__main__':
    main()