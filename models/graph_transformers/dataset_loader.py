# Sample code for graph construction from Firestore
import torch
from torch_geometric.data import HeteroData
import firebase_admin
from firebase_admin import firestore

# Initialize graph
graph = HeteroData()

# Fetch nodes from Firestore
db = firestore.client()
users = db.collection('users').stream()
places = db.collection('places').stream()
events = db.collection('events').stream()

# Add nodes to graph
user_mapping = {}  # Map Firestore IDs to node indices
place_mapping = {}
event_mapping = {}

# Add user nodes with features
user_features = []
for i, user in enumerate(users):
    user_data = user.to_dict()
    user_mapping[user.id] = i
    user_features.append([
        user_data.get('age', 0),
        user_data.get('budget_preference', 0),
        # Other user features
    ])
graph['user'].x = torch.tensor(user_features, dtype=torch.float)

# Similar code for places and events...

# Add edges
visits = db.collection('interactions').where('type', '==', 'visit').stream()
visit_edges = []
for visit in visits:
    visit_data = visit.to_dict()
    user_idx = user_mapping.get(visit_data['user_id'])
    place_idx = place_mapping.get(visit_data['place_id'])
    if user_idx is not None and place_idx is not None:
        visit_edges.append([user_idx, place_idx])

graph['user', 'visits', 'place'].edge_index = torch.tensor(visit_edges, dtype=torch.long).t()

# Add other edge types (friendships, interests, etc.)