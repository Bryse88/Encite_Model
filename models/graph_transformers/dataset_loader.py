"""
Dataset loader for the Heterogeneous Graph Transformer model.
Loads data from Firestore and builds a heterogeneous graph.
"""

import torch
import numpy as np
from torch_geometric.data import HeteroData
from google.cloud import firestore
from typing import Dict, List, Tuple, Optional, Set
import datetime


class FirestoreGraphLoader:
    """
    Loads data from Firestore and builds a heterogeneous graph for PyTorch Geometric.
    
    This loader creates a graph with nodes for users, places, and events, and edges
    for various relationships (visited, liked, friends_with, etc.).
    """
    
    def __init__(self, db=None, cache_path=None):
        """
        Initialize the loader.
        
        Args:
            db: Firestore client instance
            cache_path: Optional path to cache the graph data
        """
        self.db = db or firestore.Client()
        self.cache_path = cache_path
        
        # Mappings from Firestore IDs to node indices
        self.user_mapping = {}
        self.place_mapping = {}
        self.event_mapping = {}
        self.group_mapping = {}
        
        # Node features
        self.user_features = []
        self.place_features = []
        self.event_features = []
        self.group_features = []
        
        # Category encoding
        self.category_encoding = {}
        
        # Edge data
        self.visited_edges = []
        self.liked_edges = []
        self.friends_edges = []
        self.member_of_edges = []
        self.interested_in_edges = []
        
        # Edge timestamps (for temporal encoding)
        self.visited_times = []
        self.liked_times = []
        self.friends_times = []
        
        # Statistics
        self.stats = {
            'num_users': 0,
            'num_places': 0,
            'num_events': 0,
            'num_groups': 0,
            'num_visited_edges': 0,
            'num_liked_edges': 0,
            'num_friends_edges': 0,
            'num_member_of_edges': 0,
            'num_interested_in_edges': 0
        }
    
    def load_all_data(self, limit=None):
        """
        Load all data from Firestore and build the graph.
        
        Args:
            limit: Optional limit on the number of documents to load
        
        Returns:
            HeteroData: PyTorch Geometric heterogeneous graph
        """
        print("Loading data from Firestore...")
        self._load_category_encoding()
        self._load_users(limit)
        self._load_places(limit)
        self._load_events(limit)
        self._load_groups(limit)
        self._load_interactions(limit)
        
        print("Building graph...")
        graph = self._build_graph()
        
        print("Graph statistics:")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
        
        if self.cache_path:
            print(f"Saving graph to {self.cache_path}...")
            torch.save(graph, self.cache_path)
        
        return graph
    
    def load_from_cache(self):
        """Load graph from cache file."""
        if not self.cache_path:
            raise ValueError("Cache path not specified")
        
        print(f"Loading graph from {self.cache_path}...")
        return torch.load(self.cache_path)
    
    def _load_category_encoding(self):
        """Load place categories and create one-hot encoding."""
        categories_ref = self.db.collection('metadata').document('place_categories')
        categories_doc = categories_ref.get()
        
        if categories_doc.exists:
            categories = categories_doc.to_dict().get('categories', [])
        else:
            # Extract categories from places collection
            categories_set = set()
            places_ref = self.db.collection('places').limit(1000)
            for place in places_ref.stream():
                place_data = place.to_dict()
                if 'categories' in place_data:
                    categories_set.update(place_data['categories'])
            
            categories = list(categories_set)
            
            # Store categories in metadata
            self.db.collection('metadata').document('place_categories').set({
                'categories': categories,
                'updated_at': firestore.SERVER_TIMESTAMP
            })
        
        # Create category encoding
        for i, category in enumerate(categories):
            self.category_encoding[category] = i
    
    def _load_users(self, limit=None):
        """Load user data from Firestore."""
        query = self.db.collection('users')
        if limit:
            query = query.limit(limit)
        
        for i, user in enumerate(query.stream()):
            user_id = user.id
            user_data = user.to_dict()
            
            self.user_mapping[user_id] = i
            
            # Extract user features
            features = [
                user_data.get('age', 30) / 100,  # Normalize age
                user_data.get('budget_preference', 3) / 5,  # Normalize budget (1-5)
                user_data.get('active_level', 3) / 5,  # Normalize activity level
            ]
            
            # Add preference features if available
            preferences = user_data.get('preferences', {})
            for category in self.category_encoding:
                features.append(preferences.get(category, 0.5))
            
            self.user_features.append(features)
        
        self.stats['num_users'] = len(self.user_features)
        
        
        '''
        This needs to be pulled from supabase
        '''
    
    def _load_places(self, limit=None):
        """Load place data from Firestore."""
        query = self.db.collection('places')
        if limit:
            query = query.limit(limit)
        
        for i, place in enumerate(query.stream()):
            place_id = place.id
            place_data = place.to_dict()
            
            self.place_mapping[place_id] = i
            
            # Extract basic features
            features = [
                place_data.get('price_level', 2) / 5,  # Normalize price (0-5)
                place_data.get('rating', 3) / 5,  # Normalize rating
                place_data.get('popularity', 50) / 100,  # Normalize popularity
                float(place_data.get('indoor', True)),  # Indoor/outdoor
            ]
            
            # One-hot encode categories
            categories = place_data.get('categories', [])
            category_features = np.zeros(len(self.category_encoding))
            for category in categories:
                if category in self.category_encoding:
                    category_features[self.category_encoding[category]] = 1.0
            
            features.extend(category_features)
            self.place_features.append(features)
        
        self.stats['num_places'] = len(self.place_features)
    
    def _load_events(self, limit=None):
        """Load event data from Firestore."""
        query = self.db.collection('events')
        if limit:
            query = query.limit(limit)
        
        now = datetime.datetime.now()
        
        for i, event in enumerate(query.stream()):
            event_id = event.id
            event_data = event.to_dict()
            
            self.event_mapping[event_id] = i
            
            # Extract basic features
            start_time = event_data.get('start_time', now)
            if isinstance(start_time, str):
                start_time = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            
            # Time until event (normalized, negative for past events)
            time_delta = (start_time - now).total_seconds() / (86400 * 30)  # Normalize to months
            time_delta = max(-1.0, min(1.0, time_delta))  # Clamp to [-1, 1]
            
            features = [
                event_data.get('price', 30) / 100,  # Normalize price
                event_data.get('popularity', 50) / 100,  # Normalized popularity
                time_delta,  # Time until event (normalized)
                float(event_data.get('indoor', True)),  # Indoor/outdoor
            ]
            
            # One-hot encode categories (using same encoding as places)
            categories = event_data.get('categories', [])
            category_features = np.zeros(len(self.category_encoding))
            for category in categories:
                if category in self.category_encoding:
                    category_features[self.category_encoding[category]] = 1.0
            
            features.extend(category_features)
            self.event_features.append(features)
        
        self.stats['num_events'] = len(self.event_features)
    
    def _load_groups(self, limit=None):
        """Load group data from Firestore."""
        query = self.db.collection('groups')
        if limit:
            query = query.limit(limit)
        
        for i, group in enumerate(query.stream()):
            group_id = group.id
            group_data = group.to_dict()
            
            self.group_mapping[group_id] = i
            
            # Extract basic features
            features = [
                len(group_data.get('members', [])) / 50,  # Normalize size
                group_data.get('activity_level', 3) / 5,  # Normalize activity
            ]
            
            # One-hot encode interests (using same encoding as places)
            interests = group_data.get('interests', [])
            interest_features = np.zeros(len(self.category_encoding))
            for interest in interests:
                if interest in self.category_encoding:
                    interest_features[self.category_encoding[interest]] = 1.0
            
            features.extend(interest_features)
            self.group_features.append(features)
        
        self.stats['num_groups'] = len(self.group_features)
    
    def _load_interactions(self, limit=None):
        """Load interaction data from Firestore."""
        query = self.db.collection('interactions')
        if limit:
            query = query.limit(limit)
        
        now = datetime.datetime.now()
        
        for interaction in query.stream():
            interaction_data = interaction.to_dict()
            interaction_type = interaction_data.get('type')
            
            if interaction_type == 'visit':
                self._process_visit(interaction_data, now)
            elif interaction_type == 'like':
                self._process_like(interaction_data, now)
            elif interaction_type == 'friendship':
                self._process_friendship(interaction_data, now)
            elif interaction_type == 'group_membership':
                self._process_group_membership(interaction_data)
            elif interaction_type == 'interest':
                self._process_interest(interaction_data)
    
    def _process_visit(self, interaction_data, now):
        """Process a 'visit' interaction."""
        user_id = interaction_data.get('user_id')
        place_id = interaction_data.get('place_id')
        
        if user_id in self.user_mapping and place_id in self.place_mapping:
            user_idx = self.user_mapping[user_id]
            place_idx = self.place_mapping[place_id]
            
            self.visited_edges.append((user_idx, place_idx))
            
            # Process timestamp for temporal encoding
            timestamp = interaction_data.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                # Compute days since interaction
                days_ago = (now - timestamp).total_seconds() / 86400
                self.visited_times.append(days_ago)
            else:
                self.visited_times.append(30)  # Default to 30 days ago
            
            self.stats['num_visited_edges'] += 1
    
    def _process_like(self, interaction_data, now):
        """Process a 'like' interaction."""
        user_id = interaction_data.get('user_id')
        place_id = interaction_data.get('place_id')
        
        if user_id in self.user_mapping and place_id in self.place_mapping:
            user_idx = self.user_mapping[user_id]
            place_idx = self.place_mapping[place_id]
            
            self.liked_edges.append((user_idx, place_idx))
            
            # Process timestamp for temporal encoding
            timestamp = interaction_data.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                # Compute days since interaction
                days_ago = (now - timestamp).total_seconds() / 86400
                self.liked_times.append(days_ago)
            else:
                self.liked_times.append(30)  # Default to 30 days ago
            
            self.stats['num_liked_edges'] += 1
    
    def _process_friendship(self, interaction_data, now):
        """Process a 'friendship' interaction."""
        user1_id = interaction_data.get('user1_id')
        user2_id = interaction_data.get('user2_id')
        
        if user1_id in self.user_mapping and user2_id in self.user_mapping:
            user1_idx = self.user_mapping[user1_id]
            user2_idx = self.user_mapping[user2_id]
            
            # Add edges in both directions (undirected)
            self.friends_edges.append((user1_idx, user2_idx))
            self.friends_edges.append((user2_idx, user1_idx))
            
            # Process timestamp for temporal encoding
            timestamp = interaction_data.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                # Compute days since interaction
                days_ago = (now - timestamp).total_seconds() / 86400
                self.friends_times.append(days_ago)
                self.friends_times.append(days_ago)  # Add twice for both directions
            else:
                self.friends_times.append(90)  # Default to 90 days ago
                self.friends_times.append(90)
            
            self.stats['num_friends_edges'] += 2
    
    def _process_group_membership(self, interaction_data):
        """Process a 'group_membership' interaction."""
        user_id = interaction_data.get('user_id')
        group_id = interaction_data.get('group_id')
        
        if user_id in self.user_mapping and group_id in self.group_mapping:
            user_idx = self.user_mapping[user_id]
            group_idx = self.group_mapping[group_id]
            
            self.member_of_edges.append((user_idx, group_idx))
            self.stats['num_member_of_edges'] += 1
    
    def _process_interest(self, interaction_data):
        """Process an 'interest' interaction."""
        user_id = interaction_data.get('user_id')
        place_id = interaction_data.get('place_id')
        
        if user_id in self.user_mapping and place_id in self.place_mapping:
            user_idx = self.user_mapping[user_id]
            place_idx = self.place_mapping[place_id]
            
            self.interested_in_edges.append((user_idx, place_idx))
            self.stats['num_interested_in_edges'] += 1
    
    def _build_graph(self):
        """Build the heterogeneous graph from loaded data."""
        graph = HeteroData()
        
        # Add node features
        if self.user_features:
            graph['user'].x = torch.tensor(self.user_features, dtype=torch.float)
        
        if self.place_features:
            graph['place'].x = torch.tensor(self.place_features, dtype=torch.float)
        
        if self.event_features:
            graph['event'].x = torch.tensor(self.event_features, dtype=torch.float)
        
        if self.group_features:
            graph['group'].x = torch.tensor(self.group_features, dtype=torch.float)
        
        # Add edge indices
        if self.visited_edges:
            graph['user', 'visited', 'place'].edge_index = torch.tensor(
                self.visited_edges, dtype=torch.long).t().contiguous()
            
            if self.visited_times:
                graph['user', 'visited', 'place'].edge_attr = torch.tensor(
                    self.visited_times, dtype=torch.float).unsqueeze(1)
        
        if self.liked_edges:
            graph['user', 'liked', 'place'].edge_index = torch.tensor(
                self.liked_edges, dtype=torch.long).t().contiguous()
            
            if self.liked_times:
                graph['user', 'liked', 'place'].edge_attr = torch.tensor(
                    self.liked_times, dtype=torch.float).unsqueeze(1)
        
        if self.friends_edges:
            graph['user', 'friends_with', 'user'].edge_index = torch.tensor(
                self.friends_edges, dtype=torch.long).t().contiguous()
            
            if self.friends_times:
                graph['user', 'friends_with', 'user'].edge_attr = torch.tensor(
                    self.friends_times, dtype=torch.float).unsqueeze(1)
        
        if self.member_of_edges:
            graph['user', 'member_of', 'group'].edge_index = torch.tensor(
                self.member_of_edges, dtype=torch.long).t().contiguous()
        
        if self.interested_in_edges:
            graph['user', 'interested_in', 'place'].edge_index = torch.tensor(
                self.interested_in_edges, dtype=torch.long).t().contiguous()
        
        # Add reverse edges for bidirectional message passing
        if 'visited' in graph.edge_types:
            graph['place', 'visited_by', 'user'].edge_index = graph['user', 'visited', 'place'].edge_index.flip(0)
            if hasattr(graph['user', 'visited', 'place'], 'edge_attr'):
                graph['place', 'visited_by', 'user'].edge_attr = graph['user', 'visited', 'place'].edge_attr
        
        if 'liked' in graph.edge_types:
            graph['place', 'liked_by', 'user'].edge_index = graph['user', 'liked', 'place'].edge_index.flip(0)
            if hasattr(graph['user', 'liked', 'place'], 'edge_attr'):
                graph['place', 'liked_by', 'user'].edge_attr = graph['user', 'liked', 'place'].edge_attr
        
        if 'member_of' in graph.edge_types:
            graph['group', 'has_member', 'user'].edge_index = graph['user', 'member_of', 'group'].edge_index.flip(0)
        
        if 'interested_in' in graph.edge_types:
            graph['place', 'interest_of', 'user'].edge_index = graph['user', 'interested_in', 'place'].edge_index.flip(0)
        
        return graph


def load_graph(config):
    """
    Load graph data according to configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        graph: PyTorch Geometric heterogeneous graph
        metadata: Graph metadata
    """
    db = firestore.Client(project=config.get('project_id'))
    
    # Try loading from cache first
    cache_path = config.get('cache_path')
    loader = FirestoreGraphLoader(db, cache_path)
    
    if cache_path and os.path.exists(cache_path) and not config.get('force_reload', False):
        try:
            graph = loader.load_from_cache()
            return graph, graph.metadata()
        except Exception as e:
            print(f"Error loading from cache: {e}")
            print("Loading from Firestore instead...")
    
    # Load from Firestore
    limit = config.get('limit')
    graph = loader.load_all_data(limit)
    
    return graph, graph.metadata()