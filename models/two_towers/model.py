"""
Two-Tower model for Encite recommendations.
This model consists of separate encoders for users and places/events,
producing embeddings that can be efficiently compared for recommendations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TowerBase(nn.Module):
    """Base class for both user and item towers."""
    
    def __init__(self, input_dim, hidden_layers, output_dim, dropout=0.1):
        """
        Initialize the tower.
        
        Args:
            input_dim: Input embedding dimension
            hidden_layers: List of hidden layer dimensions
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the tower."""
        return self.layers(x)


class UserTower(TowerBase):
    """Tower for encoding user features."""
    
    def __init__(self, input_dim, hidden_layers, output_dim, dropout=0.1):
        super().__init__(input_dim, hidden_layers, output_dim, dropout)


class PlaceTower(TowerBase):
    """Tower for encoding place/event features."""
    
    def __init__(self, input_dim, hidden_layers, output_dim, dropout=0.1):
        super().__init__(input_dim, hidden_layers, output_dim, dropout)


class TwoTowerModel(nn.Module):
    """
    Two-Tower model for recommendation.
    
    This model consists of two separate towers (neural networks):
    1. User tower: Encodes user features
    2. Place tower: Encodes place/event features
    
    The similarity between user and place embeddings determines recommendation scores.
    """
    
    def __init__(self, 
                 user_input_dim, 
                 place_input_dim, 
                 hidden_layers,
                 output_dim=64, 
                 dropout=0.1,
                 temperature=0.1):
        """
        Initialize the Two-Tower model.
        
        Args:
            user_input_dim: Dimension of user input features
            place_input_dim: Dimension of place input features
            hidden_layers: List of hidden layer dimensions for both towers
            output_dim: Output embedding dimension
            dropout: Dropout rate
            temperature: Temperature for scaling dot products
        """
        super().__init__()
        
        self.user_tower = UserTower(user_input_dim, hidden_layers, output_dim, dropout)
        self.place_tower = PlaceTower(place_input_dim, hidden_layers, output_dim, dropout)
        self.temperature = temperature
    
    def forward(self, user_features, place_features):
        """
        Forward pass through the model.
        
        Args:
            user_features: User features tensor (batch_size, user_input_dim)
            place_features: Place features tensor (batch_size, place_input_dim)
        
        Returns:
            similarity_scores: Similarity scores between users and places
        """
        # Get embeddings from each tower
        user_embeddings = self.user_tower(user_features)
        place_embeddings = self.place_tower(place_features)
        
        # Normalize embeddings
        user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
        place_embeddings = F.normalize(place_embeddings, p=2, dim=1)
        
        # Compute similarity scores (dot product)
        # For in-batch pairs: compute all combinations of users and places
        similarity = torch.matmul(user_embeddings, place_embeddings.transpose(0, 1))
        
        # Apply temperature scaling
        similarity = similarity / self.temperature
        
        return similarity
    
    def get_user_embedding(self, user_features):
        """Get normalized user embedding."""
        user_embedding = self.user_tower(user_features)
        return F.normalize(user_embedding, p=2, dim=1)
    
    def get_place_embedding(self, place_features):
        """Get normalized place embedding."""
        place_embedding = self.place_tower(place_features)
        return F.normalize(place_embedding, p=2, dim=1)
    
    def predict_scores(self, user_embedding, place_embeddings):
        """
        Predict scores for a user and multiple places.
        
        Args:
            user_embedding: Single user embedding (output_dim)
            place_embeddings: Multiple place embeddings (num_places, output_dim)
        
        Returns:
            scores: Recommendation scores for each place
        """
        # Ensure embeddings are normalized
        user_embedding = F.normalize(user_embedding, p=2, dim=0)
        place_embeddings = F.normalize(place_embeddings, p=2, dim=1)
        
        # Compute dot product scores
        scores = torch.matmul(place_embeddings, user_embedding)
        
        # Scale by temperature
        scores = scores / self.temperature
        
        return scores


class TwoTowerWithContext(nn.Module):
    """
    Extended Two-Tower model that incorporates contextual information.
    
    This model adds context features (time, weather, budget, etc.) to the
    scoring function, allowing for context-aware recommendations.
    """
    
    def __init__(self, 
                 user_input_dim, 
                 place_input_dim, 
                 context_dim,
                 hidden_layers,
                 output_dim=64, 
                 context_layers=[64, 32],
                 dropout=0.1):
        """
        Initialize the context-aware Two-Tower model.
        
        Args:
            user_input_dim: Dimension of user input features
            place_input_dim: Dimension of place input features
            context_dim: Dimension of context features
            hidden_layers: List of hidden layer dimensions for both towers
            output_dim: Output embedding dimension
            context_layers: Hidden layers for context MLP
            dropout: Dropout rate
        """
        super().__init__()
        
        # Core two-tower model
        self.base_model = TwoTowerModel(
            user_input_dim=user_input_dim,
            place_input_dim=place_input_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # Context processing network
        context_net = []
        prev_dim = output_dim * 2 + context_dim  # Concatenated embeddings + context
        
        for hidden_dim in context_layers:
            context_net.append(nn.Linear(prev_dim, hidden_dim))
            context_net.append(nn.ReLU())
            context_net.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final scoring layer
        context_net.append(nn.Linear(prev_dim, 1))
        
        self.context_net = nn.Sequential(*context_net)
    
    def forward(self, user_features, place_features, context_features):
        """
        Forward pass with context features.
        
        Args:
            user_features: User features tensor
            place_features: Place features tensor
            context_features: Context features tensor
        
        Returns:
            scores: Context-aware recommendation scores
        """
        # Get embeddings from base towers
        user_embeddings = self.base_model.get_user_embedding(user_features)
        place_embeddings = self.base_model.get_place_embedding(place_features)
        
        # Prepare for context network
        batch_size = user_features.shape[0]
        
        # Concatenate embeddings with context
        combined = torch.cat([
            user_embeddings,
            place_embeddings,
            context_features
        ], dim=1)
        
        # Compute context-aware scores
        scores = self.context_net(combined)
        
        return scores.squeeze(-1)
    
    def get_user_embedding(self, user_features):
        """Get user embedding from base model."""
        return self.base_model.get_user_embedding(user_features)
    
    def get_place_embedding(self, place_features):
        """Get place embedding from base model."""
        return self.base_model.get_place_embedding(place_features)
    
    def predict_base_similarity(self, user_features, place_features):
        """Get base similarity without context."""
        return self.base_model(user_features, place_features)


def create_model(config):
    """Factory function to create a model from configuration."""
    if config.get('use_context', False):
        return TwoTowerWithContext(
            user_input_dim=config['user_input_dim'],
            place_input_dim=config['place_input_dim'],
            context_dim=config['context_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            context_layers=config.get('context_layers', [64, 32]),
            dropout=config.get('dropout', 0.1)
        )
    else:
        return TwoTowerModel(
            user_input_dim=config['user_input_dim'],
            place_input_dim=config['place_input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            dropout=config.get('dropout', 0.1),
            temperature=config.get('temperature', 0.1)
        )