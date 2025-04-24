"""
Heterogeneous Graph Transformer model for Encite.
This model learns embeddings for users, places, and events from their interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData


class NodeEncoder(nn.Module):
    """Initial encoder for node features."""
    
    def __init__(self, in_channels_dict, hidden_channels):
        super().__init__()
        self.encoders = nn.ModuleDict()
        
        # Create a linear projection for each node type
        for node_type, in_channels in in_channels_dict.items():
            self.encoders[node_type] = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
    
    def forward(self, x_dict):
        """Forward pass for initial node encoding."""
        encoded_dict = {}
        for node_type, x in x_dict.items():
            encoded_dict[node_type] = self.encoders[node_type](x)
        return encoded_dict


class TemporalEncoding(nn.Module):
    """Temporal encoding for capturing time patterns."""
    
    def __init__(self, hidden_channels, max_time_delta=365):
        super().__init__()
        self.max_time_delta = max_time_delta
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_channels // 4),
            nn.ReLU(),
            nn.Linear(hidden_channels // 4, hidden_channels)
        )
    
    def forward(self, time_deltas):
        """Forward pass for temporal encoding."""
        # Normalize time deltas to [0, 1]
        normalized_deltas = (time_deltas / self.max_time_delta).clamp(0, 1).unsqueeze(-1)
        return self.time_encoder(normalized_deltas)


class HGTModel(nn.Module):
    """
    Heterogeneous Graph Transformer model for learning node embeddings.
    
    This model uses multiple layers of graph attention to capture relationships
    between different types of nodes (users, places, events) and different types
    of edges (visited, liked, friends_with, etc.).
    """
    
    def __init__(self, 
                 metadata, 
                 in_channels_dict, 
                 hidden_channels=128, 
                 out_channels=64, 
                 num_heads=4, 
                 num_layers=3, 
                 dropout=0.2, 
                 use_temporal=True):
        super().__init__()
        
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.use_temporal = use_temporal
        
        # Initial node encoders
        self.node_encoder = NodeEncoder(in_channels_dict, hidden_channels)
        
        # Optional temporal encoding
        if use_temporal:
            self.temporal_encoder = TemporalEncoding(hidden_channels)
        
        # HGT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = HGTConv(
                hidden_channels, 
                hidden_channels,
                metadata, 
                num_heads, 
                group='sum'
            )
            self.convs.append(conv)
        
        # Output projection
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, out_channels)
            )
    
    def forward(self, x_dict, edge_index_dict, edge_time_dict=None):
        """
        Forward pass for the HGT model.
        
        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
            edge_time_dict: Optional dictionary of edge timestamps for temporal encoding
        
        Returns:
            Dictionary of node embeddings for each node type
        """
        # Initial encoding
        h_dict = self.node_encoder(x_dict)
        
        # Apply temporal encoding if available
        if self.use_temporal and edge_time_dict is not None:
            for edge_type, edge_time in edge_time_dict.items():
                src, _, dst = edge_type
                time_embedding = self.temporal_encoder(edge_time)
                
                # Add temporal encoding to source nodes
                src_idx = edge_index_dict[edge_type][0]
                h_dict[src][src_idx] = h_dict[src][src_idx] + time_embedding
        
        # Apply HGT layers
        for i in range(self.num_layers):
            h_dict_new = {}
            
            # Apply the i-th HGT layer
            conv_h_dict = self.convs[i](h_dict, edge_index_dict)
            
            # Apply residual connection
            for node_type in h_dict:
                h_dict_new[node_type] = conv_h_dict[node_type] + h_dict[node_type]
                h_dict_new[node_type] = F.relu(h_dict_new[node_type])
                h_dict_new[node_type] = F.dropout(h_dict_new[node_type], p=0.2, training=self.training)
            
            h_dict = h_dict_new
        
        # Final projection
        out_dict = {}
        for node_type, h in h_dict.items():
            out_dict[node_type] = self.lin_dict[node_type](h)
            
            # Normalize embeddings to unit length
            out_dict[node_type] = F.normalize(out_dict[node_type], p=2, dim=1)
        
        return out_dict


class HGTWithPredictor(nn.Module):
    """
    HGT model with predictor head for multiple downstream tasks:
    - User-place interaction prediction
    - User-user relationship prediction
    - Place category classification
    
    This version allows for multi-task training with different loss functions.
    """
    
    def __init__(self, hgt_model, task_heads=None):
        super().__init__()
        self.hgt_model = hgt_model
        
        # Create prediction heads if not provided
        if task_heads is None:
            task_heads = {}
            out_channels = hgt_model.out_channels
            
            # User-place prediction head
            task_heads['user_place'] = nn.Sequential(
                nn.Linear(out_channels * 2, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, 1)
            )
            
            # User-user prediction head
            task_heads['user_user'] = nn.Sequential(
                nn.Linear(out_channels * 2, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, 1)
            )
            
            # Place category prediction head (assuming 20 categories)
            task_heads['place_category'] = nn.Linear(out_channels, 20)
        
        self.task_heads = nn.ModuleDict(task_heads)
    
    def forward(self, data):
        """Forward pass through both the HGT and prediction heads."""
        # Get embeddings from HGT
        embeddings = self.hgt_model(
            data.x_dict, 
            data.edge_index_dict, 
            getattr(data, 'edge_time_dict', None)
        )
        
        return embeddings
    
    def predict_user_place(self, user_emb, place_emb):
        """Predict user-place interaction probability."""
        x = torch.cat([user_emb, place_emb], dim=1)
        return torch.sigmoid(self.task_heads['user_place'](x))
    
    def predict_user_user(self, user1_emb, user2_emb):
        """Predict user-user relationship probability."""
        x = torch.cat([user1_emb, user2_emb], dim=1)
        return torch.sigmoid(self.task_heads['user_user'](x))
    
    def predict_place_category(self, place_emb):
        """Predict place category probabilities."""
        return F.softmax(self.task_heads['place_category'](place_emb), dim=1)


def create_model(config, metadata, in_channels_dict):
    """Factory function to create the HGT model with the right configuration."""
    hgt_model = HGTModel(
        metadata=metadata,
        in_channels_dict=in_channels_dict,
        hidden_channels=config['hidden_dim'],
        out_channels=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_temporal=config.get('use_temporal', True)
    )
    
    model = HGTWithPredictor(hgt_model)
    return model