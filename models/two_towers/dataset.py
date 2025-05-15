"""
Dataset implementation for the Two-Tower model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
try:
    from google.cloud import firestore
except ImportError:
    firestore = None
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Union


class UserPlaceInteractionDataset(Dataset):
    """
    Dataset of user-place interactions for training the Two-Tower model.
    
    Each sample contains:
    - User features
    - Place features
    - Interaction label (1 for positive interaction, 0 for negative)
    - Optional context features
    """
    
    def __init__(self, 
                 user_features: Dict[str, torch.Tensor],
                 place_features: Dict[str, torch.Tensor],
                 interactions: List[Tuple[str, str, float]],
                 context_features: Optional[Dict[Tuple[str, str], torch.Tensor]] = None,
                 negative_sampling: str = 'random',
                 neg_samples_per_pos: int = 4):
        """
        Initialize the dataset.
        
        Args:
            user_features: Dictionary mapping user IDs to feature tensors
            place_features: Dictionary mapping place IDs to feature tensors
            interactions: List of (user_id, place_id, label) tuples
            context_features: Optional dictionary mapping (user_id, place_id) to context features
            negative_sampling: Strategy for negative sampling ('random', 'popular', or 'hard')
            neg_samples_per_pos: Number of negative samples per positive interaction
        """
        self.user_features = user_features
        self.place_features = place_features
        self.context_features = context_features
        self.neg_samples_per_pos = neg_samples_per_pos
        
        # Extract positive interactions
        self.pos_interactions = [(user_id, place_id) for user_id, place_id, label in interactions if label > 0]
        
        # Keep track of all user-place interactions
        self.user_interacted_places = {}
        for user_id, place_id, _ in interactions:
            if user_id not in self.user_interacted_places:
                self.user_interacted_places[user_id] = set()
            self.user_interacted_places[user_id].add(place_id)
        
        # For negative sampling
        self.all_place_ids = list(place_features.keys())
        self.all_user_ids = list(user_features.keys())
        
        # Set up negative sampling strategy
        self.negative_sampling = negative_sampling
        
        if negative_sampling == 'popular':
            # Count place occurrences for popularity-based sampling
            place_counts = {}
            for _, place_id, _ in interactions:
                place_counts[place_id] = place_counts.get(place_id, 0) + 1
            
            # Calculate sampling weights (more popular places are more likely to be sampled)
            self.place_weights = np.array([place_counts.get(pid, 0) + 1 for pid in self.all_place_ids])
            self.place_weights = self.place_weights / self.place_weights.sum()
        
        elif negative_sampling == 'hard':
            # Hard negative sampling will be done during batch generation
            # For now, initialize with random sampling
            self.place_weights = np.ones(len(self.all_place_ids)) / len(self.all_place_ids)
        
        else:  # 'random'
            self.place_weights = np.ones(len(self.all_place_ids)) / len(self.all_place_ids)
    
    def __len__(self):
        """Get number of positive interactions."""
        return len(self.pos_interactions)
    
    def __getitem__(self, idx):
        """
        Get a training sample (one positive + multiple negative interactions).
        
        Args:
            idx: Index of the positive interaction
        
        Returns:
            Dictionary containing:
            - user_features: User feature tensor
            - pos_place_features: Positive place feature tensor
            - neg_place_features: List of negative place feature tensors
            - context_features: Optional context feature tensors
        """
        # Get positive interaction
        user_id, pos_place_id = self.pos_interactions[idx]
        
        # Get user and positive place features
        user_feat = self.user_features[user_id]
        pos_place_feat = self.place_features[pos_place_id]
        
        # Sample negative places for this user
        neg_place_ids = self._sample_negatives(user_id, self.neg_samples_per_pos)
        neg_place_feats = [self.place_features[pid] for pid in neg_place_ids]
        
        # Stack negative place features
        neg_place_feats = torch.stack(neg_place_feats)
        
        # Get context features if available
        if self.context_features is not None:
            pos_context = self.context_features.get((user_id, pos_place_id), 
                                                   torch.zeros(self.context_dim))
            
            neg_contexts = []
            for neg_pid in neg_place_ids:
                neg_context = self.context_features.get((user_id, neg_pid), 
                                                      torch.zeros(self.context_dim))
                neg_contexts.append(neg_context)
            
            neg_contexts = torch.stack(neg_contexts)
            
            return {
                'user_features': user_feat,
                'pos_place_features': pos_place_feat,
                'neg_place_features': neg_place_feats,
                'pos_context': pos_context,
                'neg_contexts': neg_contexts
            }
        
        return {
            'user_features': user_feat,
            'pos_place_features': pos_place_feat,
            'neg_place_features': neg_place_feats
        }
    
    def _sample_negatives(self, user_id, n_samples):
        """
        Sample negative places for a user, ensuring no positives are sampled.
        """
        interacted = self.user_interacted_places.get(user_id, set())

        candidates = [pid for pid in self.all_place_ids if pid not in interacted]

        # If no candidates left (rare), fallback to all places
        if not candidates:
            candidates = self.all_place_ids

        if self.negative_sampling == 'popular':
            candidate_indices = [self.all_place_ids.index(pid) for pid in candidates]
            candidate_weights = self.place_weights[candidate_indices]
            candidate_weights /= candidate_weights.sum()
            return np.random.choice(candidates, n_samples, replace=(len(candidates) < n_samples), p=candidate_weights)

        return np.random.choice(candidates, n_samples, replace=(len(candidates) < n_samples))




class FirestoreDataLoader:
    """
    Loads user and place data from Firestore for the Two-Tower model.
    """
    
    def __init__(self, db=None, cache_path=None):
        """
        Initialize the loader.
        
        Args:
            db: Firestore client instance
            cache_path: Optional path to cache the data
        """
        if firestore is None:
            raise ImportError("google-cloud-firestore is not installed. Install it or disable Firestore loading.")
        self.db = db or firestore.Client()

        self.cache_path = cache_path
        
        # Data storage
        self.users = {}
        self.places = {}
        self.events = {}
        self.interactions = []
        
        # Feature dimensions
        self.user_dim = None
        self.place_dim = None
        self.event_dim = None
        self.context_dim = None
        
        # Feature mappings
        self.user_categorical_mappings = {}
        self.place_categorical_mappings = {}
        
        # Statistics
        self.stats = {
            'num_users': 0,
            'num_places': 0,
            'num_events': 0,
            'num_interactions': 0
        }
    
    def load_all_data(self, limit=None):
        """
        Load all data from Firestore.
        
        Args:
            limit: Optional limit on the number of documents to load
        
        Returns:
            user_features, place_features, interactions
        """
        print("Loading data from Firestore...")
        
        self._load_users(limit)
        self._load_places(limit)
        self._load_events(limit)
        self._load_interactions(limit)
        
        # Convert to feature tensors
        user_features = self._prepare_user_features()
        place_features = self._prepare_place_features()
        
        print("Data loading statistics:")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
        
        return user_features, place_features, self.interactions
    
    def _load_users(self, limit=None):
        """Load user data from Firestore."""
        query = self.db.collection('users')
        if limit:
            query = query.limit(limit)
        
        for user in query.stream():
            user_id = user.id
            user_data = user.to_dict()
            
            # Extract relevant user features
            features = {
                'age': user_data.get('age', 30),
                'budget_preference': user_data.get('budget_preference', 3),
                'active_level': user_data.get('active_level', 3),
            }
            
            # Add preference features if available
            preferences = user_data.get('preferences', {})
            for category, score in preferences.items():
                features[f'pref_{category}'] = score
            
            # Store user embedding if available
            if 'embedding' in user_data:
                features['embedding'] = user_data['embedding']
            
            self.users[user_id] = features
        
        self.stats['num_users'] = len(self.users)
    
    def _load_places(self, limit=None):
        """Load place data from Firestore."""
        query = self.db.collection('places')
        if limit:
            query = query.limit(limit)
        
        for place in query.stream():
            place_id = place.id
            place_data = place.to_dict()
            
            # Extract relevant place features
            features = {
                'price_level': place_data.get('price_level', 2),
                'rating': place_data.get('rating', 3),
                'popularity': place_data.get('popularity', 50),
                'indoor': float(place_data.get('indoor', True)),
            }
            
            # Add category features
            categories = place_data.get('categories', [])
            for category in categories:
                features[f'cat_{category}'] = 1.0
            
            # Store place embedding if available
            if 'embedding' in place_data:
                features['embedding'] = place_data['embedding']
            
            self.places[place_id] = features
        
        self.stats['num_places'] = len(self.places)
    
    def _load_events(self, limit=None):
        """Load event data from Firestore."""
        query = self.db.collection('events')
        if limit:
            query = query.limit(limit)
        
        for event in query.stream():
            event_id = event.id
            event_data = event.to_dict()
            
            # Extract relevant event features
            features = {
                'price': event_data.get('price', 30),
                'popularity': event_data.get('popularity', 50),
                'indoor': float(event_data.get('indoor', True)),
            }
            
            # Add category features
            categories = event_data.get('categories', [])
            for category in categories:
                features[f'cat_{category}'] = 1.0
            
            # Store event embedding if available
            if 'embedding' in event_data:
                features['embedding'] = event_data['embedding']
            
            # Store as place (for compatibility)
            self.places[f"event_{event_id}"] = features
            
            # Also store in events
            self.events[event_id] = features
        
        self.stats['num_events'] = len(self.events)
    
    def _load_interactions(self, limit=None):
        """Load interaction data from Firestore."""
        query = self.db.collection('interactions')
        if limit:
            query = query.limit(limit)
        
        for interaction in query.stream():
            interaction_data = interaction.to_dict()
            interaction_type = interaction_data.get('type')
            
            if interaction_type in ['visit', 'like', 'interested_in']:
                user_id = interaction_data.get('user_id')
                place_id = interaction_data.get('place_id')
                event_id = interaction_data.get('event_id')
                
                # Skip if user or place not found
                if user_id not in self.users:
                    continue
                
                # Determine target ID (place or event)
                target_id = place_id if place_id else f"event_{event_id}" if event_id else None
                
                if target_id not in self.places:
                    continue
                
                # Determine interaction weight
                weight = 1.0
                if interaction_type == 'visit':
                    weight = interaction_data.get('rating', 4) / 5.0 if 'rating' in interaction_data else 0.8
                elif interaction_type == 'like':
                    weight = 1.0
                elif interaction_type == 'interested_in':
                    weight = 0.6
                
                self.interactions.append((user_id, target_id, weight))
        
        self.stats['num_interactions'] = len(self.interactions)
    
    def _prepare_user_features(self):
        """Convert user data to tensors."""
        # Collect all feature keys
        feature_keys = set()
        for user_data in self.users.values():
            feature_keys.update(user_data.keys())
        
        # Remove embedding if present
        if 'embedding' in feature_keys:
            feature_keys.remove('embedding')
        
        # Sort keys for consistent ordering
        feature_keys = sorted(feature_keys)
        
        # Convert categorical features
        categorical_features = []
        
        # Create feature vectors
        user_features = {}
        for user_id, user_data in self.users.items():
            if 'embedding' in user_data:
                # Use pre-computed embedding if available
                user_features[user_id] = torch.tensor(user_data['embedding'], dtype=torch.float)
            else:
                # Create feature vector
                features = []
                
                # Add numerical features
                for key in feature_keys:
                    value = user_data.get(key, 0.0)
                    
                    if key in categorical_features:
                        # One-hot encode categorical features
                        encoding = self._one_hot_encode(value, key, 'user')
                        features.extend(encoding)
                    else:
                        # Add numerical feature
                        features.append(value)
                
                user_features[user_id] = torch.tensor(features, dtype=torch.float)
        
        # Set feature dimension
        if user_features:
            first_id = next(iter(user_features))
            self.user_dim = user_features[first_id].shape[0]
        
        return user_features
    
    def _prepare_place_features(self):
        """Convert place data to tensors."""
        # Collect all feature keys
        feature_keys = set()
        for place_data in self.places.values():
            feature_keys.update(place_data.keys())
        
        # Remove embedding if present
        if 'embedding' in feature_keys:
            feature_keys.remove('embedding')
        
        # Sort keys for consistent ordering
        feature_keys = sorted(feature_keys)
        
        # Convert categorical features
        categorical_features = []
        
        # Create feature vectors
        place_features = {}
        for place_id, place_data in self.places.items():
            if 'embedding' in place_data:
                # Use pre-computed embedding if available
                place_features[place_id] = torch.tensor(place_data['embedding'], dtype=torch.float)
            else:
                # Create feature vector
                features = []
                
                # Add numerical features
                for key in feature_keys:
                    value = place_data.get(key, 0.0)
                    
                    if key in categorical_features:
                        # One-hot encode categorical features
                        encoding = self._one_hot_encode(value, key, 'place')
                        features.extend(encoding)
                    else:
                        # Add numerical feature
                        features.append(value)
                
                place_features[place_id] = torch.tensor(features, dtype=torch.float)
        
        # Set feature dimension
        if place_features:
            first_id = next(iter(place_features))
            self.place_dim = place_features[first_id].shape[0]
        
        return place_features
    
    def _one_hot_encode(self, value, key, entity_type):
        """One-hot encode a categorical feature."""
        mappings = self.user_categorical_mappings if entity_type == 'user' else self.place_categorical_mappings
        
        if key not in mappings:
            mappings[key] = {}
        
        if value not in mappings[key]:
            mappings[key][value] = len(mappings[key])
        
        index = mappings[key][value]
        encoding = [0.0] * len(mappings[key])
        encoding[index] = 1.0
        
        return encoding


def create_train_val_test_split(interactions, val_ratio=0.1, test_ratio=0.1, by_user=True):
    """
    Split interactions into train, validation, and test sets.
    
    Args:
        interactions: List of (user_id, place_id, label) tuples
        val_ratio: Ratio of interactions to use for validation
        test_ratio: Ratio of interactions to use for testing
        by_user: Whether to split by users rather than interactions
    
    Returns:
        train_interactions, val_interactions, test_interactions
    """
    if not by_user:
        # Simple random split
        interactions = np.array(interactions)
        indices = np.random.permutation(len(interactions))
        
        test_size = int(len(interactions) * test_ratio)
        val_size = int(len(interactions) * val_ratio)
        train_size = len(interactions) - test_size - val_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_interactions = interactions[train_indices].tolist()
        val_interactions = interactions[val_indices].tolist()
        test_interactions = interactions[test_indices].tolist()
    else:
        # Split by users
        user_ids = list(set([user_id for user_id, _, _ in interactions]))
        np.random.shuffle(user_ids)
        
        test_size = int(len(user_ids) * test_ratio)
        val_size = int(len(user_ids) * val_ratio)
        
        test_users = set(user_ids[:test_size])
        val_users = set(user_ids[test_size:test_size + val_size])
        
        train_interactions = []
        val_interactions = []
        test_interactions = []
        
        for user_id, place_id, label in interactions:
            if user_id in test_users:
                test_interactions.append((user_id, place_id, label))
            elif user_id in val_users:
                val_interactions.append((user_id, place_id, label))
            else:
                train_interactions.append((user_id, place_id, label))
    
    return train_interactions, val_interactions, test_interactions


def create_context_features(interactions, context_dim=8):
    """
    Create random context features for testing.
    In a real implementation, these would come from actual contextual data.
    
    Args:
        interactions: List of (user_id, place_id, label) tuples
        context_dim: Dimension of context features
    
    Returns:
        Dictionary mapping (user_id, place_id) to context features
    """
    context_features = {}
    
    for user_id, place_id, _ in interactions:
        # Create random context vector
        context = torch.randn(context_dim)
        context_features[(user_id, place_id)] = context
    
    return context_features