"""
State normalization utilities for RL scheduler.

This module contains utilities for normalizing observation states
to tensors compatible with PyTorch models.
"""

import torch
import numpy as np
from typing import Dict, Any

class StateNormalizer:
    """Utility for normalizing states to tensors with batch dimension."""
    
    def __init__(self, device='cpu'):
        """
        Initialize state normalizer.
        
        Args:
            device: Device to place tensors on
        """
        self.device = device
        
    def normalize(self, state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Convert state dictionary to tensors with batch dimension.
        
        Args:
            state: Dictionary of state observations
            
        Returns:
            Dictionary of normalized tensors
        """
        normalized = {}
        
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                # Add batch dimension if needed
                if len(value.shape) == 1:
                    value = np.expand_dims(value, 0)
                normalized[key] = torch.FloatTensor(value).to(self.device)
            elif isinstance(value, torch.Tensor):
                # Ensure tensor is on correct device and has batch dim
                if len(value.shape) == 1:
                    value = value.unsqueeze(0)
                normalized[key] = value.to(self.device)
            else:
                normalized[key] = value
                
        return normalized