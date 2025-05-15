"""
RL model implementations for the schedule optimizer.

This module contains the core model definitions used by the
reinforcement learning scheduler, including:
1. Base model class
2. MLP-based policy implementation
3. Transformer-based policy implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Any, Optional
import numpy as np


class BaseRLModel(nn.Module):
    """
    Base class for all RL models.
    
    This provides common functionality for different policy implementations.
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initialize base RL model.
        
        Args:
            state_dim: Dimension of the latent state representation
            action_dim: Dimension of the action space
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def _normalize_state(self, state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Normalize raw state inputs to tensors.
        
        Args:
            state: Dictionary of state observations
            
        Returns:
            Dictionary of normalized state tensors
        """
        normalized = {}
        
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                # Add batch dimension if needed
                if len(value.shape) == 1:
                    value = np.expand_dims(value, 0)
                normalized[key] = torch.FloatTensor(value).to(next(self.parameters()).device)
            elif isinstance(value, torch.Tensor):
                # Ensure tensor is on correct device
                normalized[key] = value.to(next(self.parameters()).device)
            else:
                normalized[key] = value
                
        return normalized
    
    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            state: Dictionary of state observations
            
        Returns:
            Tuple of (action_probs, state_value)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def act(self, state: Dict[str, Any], deterministic: bool = False) -> Tuple[int, float]:
        """
        Select an action based on the current state.
        
        Args:
            state: Dictionary of state observations
            deterministic: Whether to select actions deterministically
            
        Returns:
            Tuple of (selected_action, value)
        """
        with torch.no_grad():
            # Normalize state
            normalized_state = self._normalize_state(state)
            
            # Get action probabilities and value
            action_probs, value = self(normalized_state)
            
            if deterministic:
                # Select action with highest probability
                action = torch.argmax(action_probs, dim=-1).item()
            else:
                # Sample from probability distribution
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()
                
            return action, value.item()


class MLPModel(BaseRLModel):
    """
    Multi-layer Perceptron policy for schedule optimization.
    
    This policy uses a simple feed-forward neural network architecture
    to map state observations to action probabilities and value estimates.
    """
    
    def __init__(self, 
                 state_dim: int = 128, 
                 action_dim: int = 50,  # Number of candidate places + 1
                 hidden_dims: Optional[List[int]] = None):
        """
        Initialize the MLP policy network.
        
        Args:
            state_dim: Dimension of the processed state
            action_dim: Dimension of the action space
            hidden_dims: Dimensions of hidden layers
        """
        super().__init__(state_dim, action_dim)
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
            
        # Input processing layers
        self.time_embedding = nn.Linear(4, 16)
        self.location_embedding = nn.Linear(2, 8)
        self.budget_embedding = nn.Linear(1, 4)
        self.remaining_time_embedding = nn.Linear(1, 4)
        self.weather_embedding = nn.Linear(3, 12)
        
        # Candidate mask and schedule processing
        self.schedule_encoder = nn.Sequential(
            nn.Linear(10, 32),  # Encode schedule indices
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Calculate input dimension to main network
        input_dim = 16 + 8 + 4 + 4 + 12 + 32  # Sum of all embedding dimensions
        
        # Main network layers
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
            
        self.main_network = nn.Sequential(*layers)
        
        # Output heads
        self.policy_head = nn.Linear(prev_dim, action_dim)
        self.value_head = nn.Linear(prev_dim, 1)
        
    def _process_state(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process the state dictionary into a single vector.
        
        Args:
            state: Dictionary of state components
            
        Returns:
            Processed state tensor
        """
        # Process different state components
        time_embed = F.relu(self.time_embedding(state['time_features']))
        loc_embed = F.relu(self.location_embedding(state['location']))
        budget_embed = F.relu(self.budget_embedding(state['remaining_budget']))
        time_remaining_embed = F.relu(self.remaining_time_embedding(state['remaining_time_minutes']))
        weather_embed = F.relu(self.weather_embedding(state['weather_features']))
        
        # Process schedule history
        schedule_embed = F.relu(self.schedule_encoder(state['schedule_so_far']))
        
        # Concatenate all embeddings
        combined = torch.cat([
            time_embed, 
            loc_embed, 
            budget_embed, 
            time_remaining_embed, 
            weather_embed, 
            schedule_embed
        ], dim=-1)
        
        return combined
        
    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            state: Dictionary of state observations
            
        Returns:
            Tuple of (action_probs, state_value)
        """
        # Process state
        x = self._process_state(state)
        
        # Main network
        features = self.main_network(x)
        
        # Get logits and apply mask
        logits = self.policy_head(features)
        
        # Apply available actions mask
        mask = state['candidates_mask']
        mask_expanded = mask.expand_as(logits)
        masked_logits = logits - (1 - mask_expanded) * 1e9  # Subtract large value from masked actions
        
        # Get action probabilities and value
        probs = F.softmax(masked_logits, dim=-1)
        value = self.value_head(features)
        
        return probs, value


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer inputs.
    
    This adds information about the position in the sequence to each embedding,
    allowing the transformer to understand sequence order.
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 10):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Embedding dimension
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (won't be updated during training)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, embedding_dim]
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]


class TransformerModel(BaseRLModel):
    """
    Transformer-based policy for schedule optimization.
    
    This policy uses a transformer architecture to better model the
    relationships between items in the schedule and make more contextual
    decisions about the next activity to add.
    """
    
    def __init__(self, 
                 state_dim: int = 128, 
                 action_dim: int = 50,  # Number of candidate places + 1
                 embedding_dim: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Initialize transformer policy.
        
        Args:
            state_dim: Dimension of the processed state
            action_dim: Dimension of the action space
            embedding_dim: Dimension of embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__(state_dim, action_dim)
        
        # Input embeddings for different parts of the state
        self.time_embedding = nn.Linear(4, embedding_dim // 4)
        self.location_embedding = nn.Linear(2, embedding_dim // 4)
        self.budget_embedding = nn.Linear(1, embedding_dim // 8)
        self.remaining_time_embedding = nn.Linear(1, embedding_dim // 8)
        self.weather_embedding = nn.Linear(3, embedding_dim // 4)
        
        # Schedule history embeddings
        self.schedule_item_embedding = nn.Embedding(
            action_dim, embedding_dim, padding_idx=-1
        )
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_length=10)
        
        # Transformer encoder to process the schedule history
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Output layers
        self.fc_state = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
            nn.ReLU()
        )
        
        self.value_head = nn.Linear(state_dim, 1)
        self.policy_head = nn.Linear(state_dim, action_dim)
        
    def _encode_state(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode the state into a fixed-size embedding.
        
        Args:
            state: Dictionary of state components
            
        Returns:
            Encoded state tensor
        """
        # Embed different parts of the state
        time_emb = self.time_embedding(state['time_features'])
        loc_emb = self.location_embedding(state['location'])
        budget_emb = self.budget_embedding(state['remaining_budget'])
        time_remaining_emb = self.remaining_time_embedding(state['remaining_time_minutes'])
        weather_emb = self.weather_embedding(state['weather_features'])
        
        # Concatenate all parts
        state_emb = torch.cat([
            time_emb, loc_emb, budget_emb, time_remaining_emb, weather_emb
        ], dim=-1)
        
        return state_emb
    
    def _encode_schedule(self, schedule: torch.Tensor) -> torch.Tensor:
        """
        Encode the schedule history using a transformer.
        
        Args:
            schedule: Tensor of place indices in the schedule
            
        Returns:
            Encoded schedule tensor
        """
        # Get embeddings
        schedule_emb = self.schedule_item_embedding(schedule)
        
        # Add positional encoding
        schedule_emb = self.positional_encoding(schedule_emb)
        
        # Create mask for padding positions
        padding_mask = (schedule == -1)
        
        # Apply transformer encoder
        encoded_schedule = self.transformer_encoder(
            schedule_emb,
            src_key_padding_mask=padding_mask
        )
        
        # Pool across the sequence dimension
        # Use mean pooling of non-padding tokens
        mask = ~padding_mask
        mask_expanded = mask.unsqueeze(-1).expand_as(encoded_schedule)
        sum_encoded = torch.sum(encoded_schedule * mask_expanded, dim=1)
        sum_mask = torch.sum(mask, dim=1, keepdim=True)
        sum_mask = torch.clamp(sum_mask, min=1)  # Avoid division by zero
        
        pooled_schedule = sum_encoded / sum_mask
        
        return pooled_schedule
    
    def forward(self, 
                state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get action probabilities and value estimates.
        
        Args:
            state: Dictionary of state observations
            
        Returns:
            Tuple of (action_probs, state_value)
        """
        # Encode state components
        state_embedding = self._encode_state(state)
        schedule_embedding = self._encode_schedule(state['schedule_so_far'])
        
        # Combine state and schedule embeddings
        combined = torch.cat([state_embedding, schedule_embedding], dim=-1)
        features = self.fc_state(combined)
        
        # Get action logits and mask invalid actions
        action_logits = self.policy_head(features)
        
        # Apply mask: set logits for already visited places to a large negative value
        candidates_mask = state['candidates_mask']
        mask_value = -1e9
        masked_logits = action_logits + (1 - candidates_mask) * mask_value
        
        # Get policy and value outputs
        policy = F.softmax(masked_logits, dim=-1)
        value = self.value_head(features)
        
        return policy, value


class HybridTransformerMLP(BaseRLModel):
    """
    Hybrid policy combining transformer and MLP components.
    
    This policy uses a transformer to process the schedule history
    and an MLP to process the current state. The outputs are then
    combined to make action decisions.
    """
    
    def __init__(self, 
                 state_dim: int = 128, 
                 action_dim: int = 50,
                 embedding_dim: int = 64,
                 hidden_dims: Optional[List[int]] = None,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Initialize hybrid transformer-MLP policy.
        
        Args:
            state_dim: Dimension of the processed state
            action_dim: Dimension of the action space
            embedding_dim: Dimension of embeddings
            hidden_dims: Dimensions of hidden layers in MLP
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__(state_dim, action_dim)
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        # Transformer component for processing schedule history
        self.schedule_item_embedding = nn.Embedding(
            action_dim, embedding_dim, padding_idx=-1
        )
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_length=10)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # MLP component for processing current state
        self.time_embedding = nn.Linear(4, 16)
        self.location_embedding = nn.Linear(2, 8)
        self.budget_embedding = nn.Linear(1, 4)
        self.remaining_time_embedding = nn.Linear(1, 4)
        self.weather_embedding = nn.Linear(3, 12)
        
        # Calculate input dimension to main network
        mlp_input_dim = 16 + 8 + 4 + 4 + 12  # Sum of all state embedding dimensions
        combined_input_dim = mlp_input_dim + embedding_dim  # Add transformer output
        
        # Main network layers
        layers = []
        prev_dim = combined_input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
            
        self.main_network = nn.Sequential(*layers)
        
        # Output heads
        self.policy_head = nn.Linear(prev_dim, action_dim)
        self.value_head = nn.Linear(prev_dim, 1)
        
    def _encode_schedule(self, schedule: torch.Tensor) -> torch.Tensor:
        """
        Encode the schedule history using a transformer.
        
        Args:
            schedule: Tensor of place indices in the schedule
            
        Returns:
            Encoded schedule tensor
        """
        # Get embeddings
        schedule_emb = self.schedule_item_embedding(schedule)
        
        # Add positional encoding
        schedule_emb = self.positional_encoding(schedule_emb)
        
        # Create mask for padding positions
        padding_mask = (schedule == -1)
        
        # Apply transformer encoder
        encoded_schedule = self.transformer_encoder(
            schedule_emb,
            src_key_padding_mask=padding_mask
        )
        
        # Pool across the sequence dimension
        mask = ~padding_mask
        mask_expanded = mask.unsqueeze(-1).expand_as(encoded_schedule)
        sum_encoded = torch.sum(encoded_schedule * mask_expanded, dim=1)
        sum_mask = torch.sum(mask, dim=1, keepdim=True)
        sum_mask = torch.clamp(sum_mask, min=1)  # Avoid division by zero
        
        pooled_schedule = sum_encoded / sum_mask
        
        return pooled_schedule
    
    def _encode_state(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode the current state using MLP embeddings.
        
        Args:
            state: Dictionary of state components
            
        Returns:
            Encoded state tensor
        """
        # Process different state components
        time_embed = F.relu(self.time_embedding(state['time_features']))
        loc_embed = F.relu(self.location_embedding(state['location']))
        budget_embed = F.relu(self.budget_embedding(state['remaining_budget']))
        time_remaining_embed = F.relu(self.remaining_time_embedding(state['remaining_time_minutes']))
        weather_embed = F.relu(self.weather_embedding(state['weather_features']))
        
        # Concatenate all embeddings
        combined = torch.cat([
            time_embed, 
            loc_embed, 
            budget_embed, 
            time_remaining_embed, 
            weather_embed
        ], dim=-1)
        
        return combined
    
    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the hybrid policy network.
        
        Args:
            state: Dictionary of state observations
            
        Returns:
            Tuple of (action_probs, state_value)
        """
        # Encode schedule with transformer
        schedule_embedding = self._encode_schedule(state['schedule_so_far'])
        
        # Encode current state with MLP
        state_embedding = self._encode_state(state)
        
        # Combine embeddings
        combined = torch.cat([state_embedding, schedule_embedding], dim=-1)
        
        # Main network
        features = self.main_network(combined)
        
        # Get logits and apply mask
        logits = self.policy_head(features)
        
        # Apply available actions mask
        mask = state['candidates_mask']
        mask_expanded = mask.expand_as(logits)
        masked_logits = logits - (1 - mask_expanded) * 1e9  # Subtract large value from masked actions
        
        # Get action probabilities and value
        probs = F.softmax(masked_logits, dim=-1)
        value = self.value_head(features)
        
        return probs, value


def create_model(model_type: str, 
                action_dim: int,
                config: Dict[str, Any]) -> BaseRLModel:
    """
    Factory function to create a policy model.
    
    Args:
        model_type: Type of model ('mlp', 'transformer', or 'hybrid')
        action_dim: Dimension of the action space
        config: Model configuration parameters
        
    Returns:
        Instantiated model
    """
    if model_type == 'mlp':
        return MLPModel(
            state_dim=config.get('state_dim', 128),
            action_dim=action_dim,
            hidden_dims=config.get('hidden_dims', [256, 256])
        )
    elif model_type == 'transformer':
        return TransformerModel(
            state_dim=config.get('state_dim', 128),
            action_dim=action_dim,
            embedding_dim=config.get('embedding_dim', 64),
            nhead=config.get('nhead', 4),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1)
        )
    elif model_type == 'hybrid':
        return HybridTransformerMLP(
            state_dim=config.get('state_dim', 128),
            action_dim=action_dim,
            embedding_dim=config.get('embedding_dim', 64),
            hidden_dims=config.get('hidden_dims', [256, 256]),
            nhead=config.get('nhead', 4),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")