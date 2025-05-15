import pandas as pd
import numpy as np
import torch
import os
import ast
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear, to_hetero
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine

from models.transformer.embedding_io import save_and_upload

# 1. Load and preprocess the data
def load_data():
    # Load the CSVs
    places_df = pd.read_csv('data/places_rows.csv')
    events_df = pd.read_csv('data/events_rows.csv')
    items_df = pd.read_csv('data/items_rows.csv')
    users_df = pd.read_csv('data/encite_onboarding_data_randomized_density.csv')
    
    # Preprocessing for users
    users_df['experience_vibes'] = users_df['experience_vibes'].apply(ast.literal_eval)
    users_df['activities'] = users_df['activities'].apply(ast.literal_eval)
    users_df['location_priorities'] = users_df['location_priorities'].apply(ast.literal_eval)
    
    # Convert categorical string values to numerical ids
    # Each node type will have its own id space
    users_df['user_id'] = users_df.index
    places_df['place_idx'] = places_df.index
    events_df['event_idx'] = events_df.index
    items_df['item_idx'] = items_df.index
    
    return users_df, places_df, events_df, items_df

# 2. Create feature vectors for each node type
def create_node_features(users_df, places_df, events_df, items_df):
    # User features
    user_features = []
    for _, user in users_df.iterrows():
        # Age (normalized)
        age = user['age'] / 100
        
        # Schedule density and planning style (already numeric)
        density = user['schedule_density'] / 5  # Assuming max is 5
        planning = user['planning_style'] / 5   # Assuming max is 5
        
        # One-hot encode experience vibes
        vibe_categories = ['Chill & Laid-back', 'Social & Outgoing', 'Loud & Energetic', 'Intimate']
        vibes = user['experience_vibes']
        vibe_features = [1 if vibe in vibes else 0 for vibe in vibe_categories]
        
        # One-hot encode activities
        activity_categories = [
            'Try new restaurants / cafes', 
            'Go to live events', 
            'Explore parks or nature', 
            'Hosting or joining game/movie nights'
        ]
        activities = user['activities']
        activity_features = [1 if activity in activities else 0 for activity in activity_categories]
        
        # One-hot encode dietary preferences
        diet_categories = ['No restrictions', 'Vegetarian', 'Vegan', 'Halal', 'Kosher', 'Gluten-free']
        diet = user['dietary_preference']
        diet_features = [1 if diet == category else 0 for category in diet_categories]
        
        # Location priorities (already a dictionary of numerical values)
        location_priorities = list(user['location_priorities'].values())
        
        # One-hot encode travel willingness
        travel_categories = ['Low', 'Medium', 'High']
        travel = user['travel_willingness']
        travel_features = [1 if travel == category else 0 for category in travel_categories]
        
        # Combine all features
        features = [age, density, planning] + vibe_features + activity_features + diet_features + location_priorities + travel_features
        user_features.append(features)
    
    # Convert to torch tensor
    user_features = torch.tensor(user_features, dtype=torch.float)
    
    # Place features
    place_features = []
    for _, place in places_df.iterrows():
        # Price level and rating (normalized)
        price = place['price_level'] / 5 if not pd.isna(place['price_level']) else 0
        rating = place['rating'] / 5 if not pd.isna(place['rating']) else 0
        popularity = place['popularity_score'] / 10 if not pd.isna(place['popularity_score']) else 0
        
        # One-hot encode category
        category = place['category']
        # Get unique categories from the dataset
        unique_categories = places_df['category'].unique().tolist()
        category_features = [1 if category == cat else 0 for cat in unique_categories]
        
        # Combine all features
        features = [price, rating, popularity] + category_features
        place_features.append(features)
    
    # Convert to torch tensor
    place_features = torch.tensor(place_features, dtype=torch.float)
    
    # Event features
    event_features = []
    for _, event in events_df.iterrows():
        # Price (normalized)
        price = float(event['price']) / 100 if not pd.isna(event['price']) else 0
        
        # One-hot encode category
        category = event['category']
        unique_categories = events_df['category'].unique().tolist()
        category_features = [1 if category == cat else 0 for cat in unique_categories]
        
        # Combine all features
        features = [price] + category_features
        event_features.append(features)
    
    # Convert to torch tensor
    event_features = torch.tensor(event_features, dtype=torch.float)
    
    # Item features
    item_features = []
    for _, item in items_df.iterrows():
        # Price and popularity (normalized)
        price = float(item['price']) / 100 if not pd.isna(item['price']) else 0
        popularity = item['popularity_score'] / 10 if not pd.isna(item['popularity_score']) else 0
        
        # One-hot encode category
        category = item['category']
        unique_categories = items_df['category'].unique().tolist()
        category_features = [1 if category == cat else 0 for cat in unique_categories]
        
        # Combine all features
        features = [price, popularity] + category_features
        item_features.append(features)
    
    # Convert to torch tensor
    item_features = torch.tensor(item_features, dtype=torch.float)
    
    return user_features, place_features, event_features, item_features

# 3. Create synthetic groups and generate interactions
def create_synthetic_groups(users_df, places_df, events_df, items_df, num_groups=50):
    np.random.seed(42)
    
    # Create groups with 2-6 users in each
    groups = []
    group_members = []
    
    for i in range(num_groups):
        group_size = np.random.randint(2, 7)
        # Sample users randomly
        member_indices = np.random.choice(len(users_df), size=group_size, replace=False)
        
        # Create group entry
        group = {
            'group_id': i,
            'name': f'Group {i}',
            'size': group_size,
            # Aggregate user features for group features
            'avg_age': users_df.iloc[member_indices]['age'].mean(),
            'avg_density': users_df.iloc[member_indices]['schedule_density'].mean(),
            'avg_planning': users_df.iloc[member_indices]['planning_style'].mean()
        }
        groups.append(group)
        
        # Create membership relations
        for user_idx in member_indices:
            group_members.append({
                'group_id': i,
                'user_id': user_idx
            })
    
    groups_df = pd.DataFrame(groups)
    group_members_df = pd.DataFrame(group_members)
    
    # Create synthetic interactions for groups
    
    # Group-Place interactions
    group_place_edges = []
    edge_types = ['interested_in', 'scheduled', 'disliked', 'visited']
    
    for group_id in range(num_groups):
        # Each group interacts with 3-10 places
        num_interactions = np.random.randint(3, 11)
        place_indices = np.random.choice(len(places_df), size=num_interactions, replace=False)
        
        for place_idx in place_indices:
            edge_type = np.random.choice(edge_types, p=[0.4, 0.3, 0.1, 0.2])  # Weighted probabilities
            group_place_edges.append({
                'group_id': group_id,
                'place_idx': place_idx,
                'edge_type': edge_type
            })
    
    # Group-Event interactions
    group_event_edges = []
    
    for group_id in range(num_groups):
        # Each group interacts with 2-8 events
        num_interactions = np.random.randint(2, 9)
        event_indices = np.random.choice(len(events_df), size=num_interactions, replace=False)
        
        for event_idx in event_indices:
            edge_type = np.random.choice(edge_types, p=[0.4, 0.3, 0.1, 0.2])
            group_event_edges.append({
                'group_id': group_id,
                'event_idx': event_idx,
                'edge_type': edge_type
            })
    
    # Group-Item interactions
    group_item_edges = []
    item_edge_types = ['interested_in', 'disliked', 'ordered']
    
    for group_id in range(num_groups):
        # Each group interacts with 1-5 items
        num_interactions = np.random.randint(1, 6)
        item_indices = np.random.choice(len(items_df), size=num_interactions, replace=False)
        
        for item_idx in item_indices:
            edge_type = np.random.choice(item_edge_types, p=[0.5, 0.2, 0.3])
            group_item_edges.append({
                'group_id': group_id,
                'item_idx': item_idx,
                'edge_type': edge_type
            })
    
    # User-Place interactions
    user_place_edges = []
    user_edge_types = ['likes', 'disliked', 'visited', 'scheduled']
    
    for user_id in range(len(users_df)):
        # Each user interacts with 3-15 places
        num_interactions = np.random.randint(3, 16)
        place_indices = np.random.choice(len(places_df), size=num_interactions, replace=False)
        
        for place_idx in place_indices:
            edge_type = np.random.choice(user_edge_types, p=[0.4, 0.1, 0.3, 0.2])
            user_place_edges.append({
                'user_id': user_id,
                'place_idx': place_idx,
                'edge_type': edge_type
            })
    
    # User-Event interactions
    user_event_edges = []
    
    for user_id in range(len(users_df)):
        # Each user interacts with 2-10 events
        num_interactions = np.random.randint(2, 11)
        event_indices = np.random.choice(len(events_df), size=num_interactions, replace=False)
        
        for event_idx in event_indices:
            edge_type = np.random.choice(['interested_in', 'disliked', 'visited', 'scheduled'], p=[0.4, 0.1, 0.3, 0.2])
            user_event_edges.append({
                'user_id': user_id,
                'event_idx': event_idx,
                'edge_type': edge_type
            })
    
    # User-Item interactions
    user_item_edges = []
    
    for user_id in range(len(users_df)):
        # Each user interacts with 1-8 items
        num_interactions = np.random.randint(1, 9)
        item_indices = np.random.choice(len(items_df), size=num_interactions, replace=False)
        
        for item_idx in item_indices:
            edge_type = np.random.choice(['interested_in', 'disliked', 'ordered'], p=[0.5, 0.2, 0.3])
            user_item_edges.append({
                'user_id': user_id,
                'item_idx': item_idx,
                'edge_type': edge_type
            })
    
    # User-User interactions
    user_user_edges = []
    
    # Use group relationships to create user-user connections (friends_with)
    for group_id, group_df in group_members_df.groupby('group_id'):
        users_in_group = group_df['user_id'].values
        for i in range(len(users_in_group)):
            for j in range(i+1, len(users_in_group)):
                # 70% chance of being friends if in same group
                if np.random.random() < 0.7:
                    user_user_edges.append({
                        'user_id_1': users_in_group[i],
                        'user_id_2': users_in_group[j],
                        'edge_type': 'friends_with'
                    })
    
    # Convert all to DataFrames
    group_place_df = pd.DataFrame(group_place_edges)
    group_event_df = pd.DataFrame(group_event_edges)
    group_item_df = pd.DataFrame(group_item_edges)
    user_place_df = pd.DataFrame(user_place_edges)
    user_event_df = pd.DataFrame(user_event_edges)
    user_item_df = pd.DataFrame(user_item_edges)
    user_user_df = pd.DataFrame(user_user_edges)
    
    return (groups_df, group_members_df, group_place_df, group_event_df, group_item_df, 
            user_place_df, user_event_df, user_item_df, user_user_df)

# 4. Generate synthetic group features
def create_group_features(groups_df, group_members_df, users_df):
    group_features = []
    
    for _, group in groups_df.iterrows():
        group_id = group['group_id']
        member_ids = group_members_df[group_members_df['group_id'] == group_id]['user_id'].values
        
        # Aggregate member features
        member_rows = users_df.loc[member_ids]
        
        # Average numerical features
        avg_age = member_rows['age'].mean() / 100
        avg_density = member_rows['schedule_density'].mean() / 5
        avg_planning = member_rows['planning_style'].mean() / 5
        
        # Combine categorical features - count frequency of each value
        all_vibes = []
        for vibes in member_rows['experience_vibes']:
            all_vibes.extend(vibes)
        
        vibe_categories = ['Chill & Laid-back', 'Social & Outgoing', 'Loud & Energetic', 'Intimate']
        vibe_counts = [all_vibes.count(vibe) / len(member_ids) for vibe in vibe_categories]
        
        all_activities = []
        for activities in member_rows['activities']:
            all_activities.extend(activities)
        
        activity_categories = [
            'Try new restaurants / cafes', 
            'Go to live events', 
            'Explore parks or nature', 
            'Hosting or joining game/movie nights'
        ]
        activity_counts = [all_activities.count(activity) / len(member_ids) for activity in activity_categories]
        
        # Dietary restrictions - take the most restrictive option
        diet_categories = [
            'No allergies or restrictions',
            'Vegetarian/Vegan',
            'Gluten-free',
            'Nut allergies',
            'Dairy-free'
            ]

        normalized_categories = [d.lower().strip() for d in diet_categories]
        diet_features = [0] * len(diet_categories)

        for diet in member_rows['dietary_preference']:
            diet_lower = str(diet).lower().strip()
            if diet_lower in normalized_categories:
                diet_idx = normalized_categories.index(diet_lower)
                diet_features[diet_idx] = 1

        
        # Location priorities - average the values
        location_priorities = [0] * 6  # There are 6 priority categories
        for _, user in member_rows.iterrows():
            priorities = list(user['location_priorities'].values())
            location_priorities = [location_priorities[i] + priorities[i] for i in range(len(priorities))]
        
        location_priorities = [p / len(member_ids) for p in location_priorities]
        
        # Travel willingness - take the minimum
        travel_categories = ['Low', 'Medium', 'High']
        travel_values = {'Low': 0, 'Medium': 1, 'High': 2}
        min_travel = min(travel_values[travel] for travel in member_rows['travel_willingness'])
        travel_features = [1 if i == min_travel else 0 for i in range(3)]
        
        # Combine all features
        features = [avg_age, avg_density, avg_planning, group['size'] / 10] + vibe_counts + activity_counts + diet_features + location_priorities + travel_features
        group_features.append(features)
    
    # Convert to torch tensor
    group_features = torch.tensor(group_features, dtype=torch.float)
    
    return group_features

# 5. Build the heterogeneous graph
def build_hetero_graph(
    user_features, place_features, event_features, item_features, group_features,
    group_members_df, group_place_df, group_event_df, group_item_df,
    user_place_df, user_event_df, user_item_df, user_user_df
):
    # Create an empty heterogeneous graph
    data = HeteroData()
    
    # Add node features
    data['user'].x = user_features
    data['place'].x = place_features
    data['event'].x = event_features
    data['item'].x = item_features
    data['group'].x = group_features
    
    # Add edges

    # Group-User edges
    src = torch.tensor(group_members_df['group_id'].values, dtype=torch.long)
    dst = torch.tensor(group_members_df['user_id'].values, dtype=torch.long)
    data['group', 'has_member', 'user'].edge_index = torch.stack([src, dst])
    
    # User-Group edges (reverse of above)
    data['user', 'member_of', 'group'].edge_index = torch.stack([dst, src])  # dst,src because it's reversed
    
    # Group-Place edges
    for edge_type in ['interested_in', 'scheduled', 'disliked', 'visited']:
        filtered_df = group_place_df[group_place_df['edge_type'] == edge_type]
        if not filtered_df.empty:
            src = torch.tensor(filtered_df['group_id'].values, dtype=torch.long)
            dst = torch.tensor(filtered_df['place_idx'].values, dtype=torch.long)
            data['group', edge_type, 'place'].edge_index = torch.stack([src, dst])
    
    # Group-Event edges
    for edge_type in ['interested_in', 'scheduled', 'disliked', 'visited']:
        filtered_df = group_event_df[group_event_df['edge_type'] == edge_type]
        if not filtered_df.empty:
            src = torch.tensor(filtered_df['group_id'].values, dtype=torch.long)
            dst = torch.tensor(filtered_df['event_idx'].values, dtype=torch.long)
            data['group', edge_type, 'event'].edge_index = torch.stack([src, dst])
    
    # Group-Item edges
    for edge_type in ['interested_in', 'disliked', 'ordered']:
        filtered_df = group_item_df[group_item_df['edge_type'] == edge_type]
        if not filtered_df.empty:
            src = torch.tensor(filtered_df['group_id'].values, dtype=torch.long)
            dst = torch.tensor(filtered_df['item_idx'].values, dtype=torch.long)
            data['group', edge_type, 'item'].edge_index = torch.stack([src, dst])
    
    # User-Place edges
    for edge_type in ['likes', 'disliked', 'visited', 'scheduled']:
        filtered_df = user_place_df[user_place_df['edge_type'] == edge_type]
        if not filtered_df.empty:
            src = torch.tensor(filtered_df['user_id'].values, dtype=torch.long)
            dst = torch.tensor(filtered_df['place_idx'].values, dtype=torch.long)
            data['user', edge_type, 'place'].edge_index = torch.stack([src, dst])
    
    # User-Event edges
    for edge_type in ['interested_in', 'disliked', 'visited', 'scheduled']:
        filtered_df = user_event_df[user_event_df['edge_type'] == edge_type]
        if not filtered_df.empty:
            src = torch.tensor(filtered_df['user_id'].values, dtype=torch.long)
            dst = torch.tensor(filtered_df['event_idx'].values, dtype=torch.long)
            data['user', edge_type, 'event'].edge_index = torch.stack([src, dst])
    
    # User-Item edges
    for edge_type in ['interested_in', 'disliked', 'ordered']:
        filtered_df = user_item_df[user_item_df['edge_type'] == edge_type]
        if not filtered_df.empty:
            src = torch.tensor(filtered_df['user_id'].values, dtype=torch.long)
            dst = torch.tensor(filtered_df['item_idx'].values, dtype=torch.long)
            data['user', edge_type, 'item'].edge_index = torch.stack([src, dst])
    
    # User-User edges
    filtered_df = user_user_df[user_user_df['edge_type'] == 'friends_with']
    if not filtered_df.empty:
        src = torch.tensor(filtered_df['user_id_1'].values, dtype=torch.long)
        dst = torch.tensor(filtered_df['user_id_2'].values, dtype=torch.long)
        # Add edge in both directions as friendship is bidirectional
        data['user', 'friends_with', 'user'].edge_index = torch.stack([
            torch.cat([src, dst]),  # A is friends with B
            torch.cat([dst, src])   # B is friends with A
        ])
    
    # Add grouped_with relations from being in same group
    user_grouped_with = []
    
    for _, group_df in group_members_df.groupby('group_id'):
        users_in_group = group_df['user_id'].values
        for i in range(len(users_in_group)):
            for j in range(len(users_in_group)):
                if i != j:  # Don't connect users to themselves
                    user_grouped_with.append((users_in_group[i], users_in_group[j]))
    
    if user_grouped_with:
        grouped_with_edges = torch.tensor(user_grouped_with, dtype=torch.long).t()
        data['user', 'grouped_with', 'user'].edge_index = grouped_with_edges
    
    return data

# 6. Define the HGT model
class HGTModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super().__init__()
        
        # Get the node types and edge types from the data
        self.node_types = list(data.node_types)
        self.edge_types = list(data.edge_types)
        
        # Store the size of the input node features
        self.node_feature_dims = {node_type: data[node_type].x.size(1) for node_type in self.node_types}
        
        # Input projection layers
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in self.node_types:
            self.lin_dict[node_type] = Linear(self.node_feature_dims[node_type], hidden_channels)
        
        # HGT Convolution layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads)
            self.convs.append(conv)
        
        # Output layers for entity types of interest
        self.user_out = Linear(hidden_channels, out_channels)
        self.group_out = Linear(hidden_channels, out_channels)
        self.place_out = Linear(hidden_channels, out_channels)
        self.event_out = Linear(hidden_channels, out_channels)
        self.item_out = Linear(hidden_channels, out_channels)
        
    def forward(self, x_dict, edge_index_dict):
        # Initial input projection
        h_dict = {node_type: self.lin_dict[node_type](x) for node_type, x in x_dict.items()}
        
        # Apply graph convolutions
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            # Apply ReLU activation
            h_dict = {node_type: F.relu(h) for node_type, h in h_dict.items()}
        
        # Project to output embeddings
        out_dict = {
            'user': self.user_out(h_dict['user']),
            'group': self.group_out(h_dict['group']),
            'place': self.place_out(h_dict['place']),
            'event': self.event_out(h_dict['event']),
            'item': self.item_out(h_dict['item'])
        }
        
        return out_dict

def create_training_data(data, negative_sample_ratio=5):
    """
    Prepare positive and negative samples for contrastive learning.
    """
    # Collect all positive edges
    positive_samples = []
    
    # Group-Place positive interactions
    for edge_type in ['interested_in', 'scheduled', 'visited']:
        if ('group', edge_type, 'place') in data.edge_types:
            edge_index = data['group', edge_type, 'place'].edge_index
            for i in range(edge_index.size(1)):
                positive_samples.append(('group', edge_index[0, i].item(), 'place', edge_index[1, i].item()))
    
    # Group-Event positive interactions
    for edge_type in ['interested_in', 'scheduled', 'visited']:
        if ('group', edge_type, 'event') in data.edge_types:
            edge_index = data['group', edge_type, 'event'].edge_index
            for i in range(edge_index.size(1)):
                positive_samples.append(('group', edge_index[0, i].item(), 'event', edge_index[1, i].item()))
    
    # Group-Item positive interactions
    for edge_type in ['interested_in', 'ordered']:
        if ('group', edge_type, 'item') in data.edge_types:
            edge_index = data['group', edge_type, 'item'].edge_index
            for i in range(edge_index.size(1)):
                positive_samples.append(('group', edge_index[0, i].item(), 'item', edge_index[1, i].item()))
    
    # User-Place positive interactions
    for edge_type in ['likes', 'visited', 'scheduled']:
        if ('user', edge_type, 'place') in data.edge_types:
            edge_index = data['user', edge_type, 'place'].edge_index
            for i in range(edge_index.size(1)):
                positive_samples.append(('user', edge_index[0, i].item(), 'place', edge_index[1, i].item()))
    
    # User-Event positive interactions
    for edge_type in ['interested_in', 'visited', 'scheduled']:
        if ('user', edge_type, 'event') in data.edge_types:
            edge_index = data['user', edge_type, 'event'].edge_index
            for i in range(edge_index.size(1)):
                positive_samples.append(('user', edge_index[0, i].item(), 'event', edge_index[1, i].item()))
    
    # User-Item positive interactions
    for edge_type in ['interested_in', 'ordered']:
        if ('user', edge_type, 'item') in data.edge_types:
            edge_index = data['user', edge_type, 'item'].edge_index
            for i in range(edge_index.size(1)):
                positive_samples.append(('user', edge_index[0, i].item(), 'item', edge_index[1, i].item()))
    
    # Collect all negative edges
    negative_samples = []
    
    # Create negative samples - entities that have 'disliked' relation
    # Group-Place negative interactions
    if ('group', 'disliked', 'place') in data.edge_types:
        edge_index = data['group', 'disliked', 'place'].edge_index
        for i in range(edge_index.size(1)):
            negative_samples.append(('group', edge_index[0, i].item(), 'place', edge_index[1, i].item()))
    
    # Group-Event negative interactions
    if ('group', 'disliked', 'event') in data.edge_types:
        edge_index = data['group', 'disliked', 'event'].edge_index
        for i in range(edge_index.size(1)):
            negative_samples.append(('group', edge_index[0, i].item(), 'event', edge_index[1, i].item()))
    
    # Group-Item negative interactions
    if ('group', 'disliked', 'item') in data.edge_types:
        edge_index = data['group', 'disliked', 'item'].edge_index
        for i in range(edge_index.size(1)):
            negative_samples.append(('group', edge_index[0, i].item(), 'item', edge_index[1, i].item()))
    
    # User-Place negative interactions
    if ('user', 'disliked', 'place') in data.edge_types:
        edge_index = data['user', 'disliked', 'place'].edge_index
        for i in range(edge_index.size(1)):
            negative_samples.append(('user', edge_index[0, i].item(), 'place', edge_index[1, i].item()))
    
    # User-Event negative interactions
    if ('user', 'disliked', 'event') in data.edge_types:
        edge_index = data['user', 'disliked', 'event'].edge_index
        for i in range(edge_index.size(1)):
            negative_samples.append(('user', edge_index[0, i].item(), 'event', edge_index[1, i].item()))
    
    # User-Item negative interactions
    if ('user', 'disliked', 'item') in data.edge_types:
        edge_index = data['user', 'disliked', 'item'].edge_index
        for i in range(edge_index.size(1)):
            negative_samples.append(('user', edge_index[0, i].item(), 'item', edge_index[1, i].item()))
    
    # If we don't have enough negative samples, generate additional ones
    num_required_negatives = len(positive_samples) * negative_sample_ratio
    
    # Generate additional negative samples if needed
    if len(negative_samples) < num_required_negatives:
        additional_negatives = []
        num_groups = data['group'].x.size(0)
        num_users = data['user'].x.size(0)
        num_places = data['place'].x.size(0)
        num_events = data['event'].x.size(0)
        num_items = data['item'].x.size(0)
        
        # Collect existing positive and negative pairs for easy lookup
        existing_edges = set()
        for sample in positive_samples + negative_samples:
            existing_edges.add((sample[0], sample[1], sample[2], sample[3]))
        
        # Generate random negative samples, prioritizing groups
        while len(additional_negatives) < (num_required_negatives - len(negative_samples)):
            source_type = np.random.choice(['group', 'user'], p=[0.7, 0.3])  # Prioritize group negatives
            
            if source_type == 'group':
                source_id = np.random.randint(0, num_groups)
            else:  # user
                source_id = np.random.randint(0, num_users)
                
            target_type = np.random.choice(['place', 'event', 'item'])
            
            if target_type == 'place':
                target_id = np.random.randint(0, num_places)
            elif target_type == 'event':
                target_id = np.random.randint(0, num_events)
            else:  # item
                target_id = np.random.randint(0, num_items)
            
            # Check if this is not already a positive or negative edge
            if (source_type, source_id, target_type, target_id) not in existing_edges:
                additional_negatives.append((source_type, source_id, target_type, target_id))
                existing_edges.add((source_type, source_id, target_type, target_id))
        
        negative_samples.extend(additional_negatives)
    
    # Split into train/validation/test sets
    positive_train, positive_valtest = train_test_split(positive_samples, test_size=0.3, random_state=42)
    positive_val, positive_test = train_test_split(positive_valtest, test_size=0.5, random_state=42)
    
    negative_train, negative_valtest = train_test_split(negative_samples, test_size=0.3, random_state=42)
    negative_val, negative_test = train_test_split(negative_valtest, test_size=0.5, random_state=42)
    
    return {
        'train': (positive_train, negative_train),
        'val': (positive_val, negative_val),
        'test': (positive_test, negative_test)
    }

# 8. Define the contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embeddings_dict, positive_samples, negative_samples, group_weight=2.0):
        """
        Compute the contrastive loss with higher weight for group recommendations.
        
        Args:
            embeddings_dict: Dictionary of embeddings for each node type
            positive_samples: List of tuples (source_type, source_id, target_type, target_id)
            negative_samples: List of tuples (source_type, source_id, target_type, target_id)
            group_weight: Weight to apply to group-related loss
        """
        loss = 0.0
        num_pos = len(positive_samples)
        num_neg = len(negative_samples)
        
        # Process positive samples
        for sample in positive_samples:
            source_type, source_id, target_type, target_id = sample
            source_emb = embeddings_dict[source_type][source_id]
            target_emb = embeddings_dict[target_type][target_id]
            
            # Compute similarity (cosine distance)
            sim = F.cosine_similarity(source_emb.unsqueeze(0), target_emb.unsqueeze(0))
            pos_loss = 1.0 - sim  # For positive samples, we want embeddings to be similar
            
            # Apply weight for group recommendations
            if source_type == 'group':
                pos_loss *= group_weight
                
            loss += pos_loss
        
        # Process negative samples
        for sample in negative_samples:
            source_type, source_id, target_type, target_id = sample
            source_emb = embeddings_dict[source_type][source_id]
            target_emb = embeddings_dict[target_type][target_id]
            
            # Compute similarity
            sim = F.cosine_similarity(source_emb.unsqueeze(0), target_emb.unsqueeze(0))
            neg_loss = torch.max(torch.tensor(0.0, device=sim.device), sim - self.margin)  # Hinge loss
            
            # Apply weight for group recommendations
            if source_type == 'group':
                neg_loss *= group_weight
                
            loss += neg_loss
        
        # Normalize by the number of samples
        return loss / (num_pos + num_neg)

# 9. Training function
def train_hgt_model(data, train_data, val_data, hidden_channels=64, out_channels=32, 
                    num_heads=8, num_layers=2, epochs=100, lr=0.001, group_weight=2.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HGTModel(hidden_channels, out_channels, num_heads, num_layers, data).to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = ContrastiveLoss(margin=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out_dict = model(data.x_dict, data.edge_index_dict)
        
        # Compute loss
        pos_samples, neg_samples = train_data
        loss = criterion(out_dict, pos_samples, neg_samples, group_weight)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data.x_dict, data.edge_index_dict)
            val_pos, val_neg = val_data
            val_loss = criterion(val_out, val_pos, val_neg, group_weight)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = {
                'model_state_dict': model.state_dict().copy(),
                'epoch': epoch,
                'val_loss': val_loss.item(),
                'config': {
                    'hidden_channels': hidden_channels,
                    'out_channels': out_channels,
                    'num_heads': num_heads,
                    'num_layers': num_layers
                }
            }
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    # Load best model
    model.load_state_dict(best_model['model_state_dict'])
    
    return model, best_model
 
# 12. Main function to run the entire pipeline
def main():
    # Load data
    users_df, places_df, events_df, items_df = load_data()
    
    # Create node features
    user_features, place_features, event_features, item_features = create_node_features(users_df, places_df, events_df, items_df)
    
    # Create synthetic groups and interactions
    (groups_df, group_members_df, group_place_df, group_event_df, group_item_df,
     user_place_df, user_event_df, user_item_df, user_user_df) = create_synthetic_groups(users_df, places_df, events_df, items_df)
    
    # Create group features
    group_features = create_group_features(groups_df, group_members_df, users_df)
    
    # Build heterogeneous graph
    data = build_hetero_graph(
        user_features, place_features, event_features, item_features, group_features,
        group_members_df, group_place_df, group_event_df, group_item_df,
        user_place_df, user_event_df, user_item_df, user_user_df
    )
    
    # Create training data
    split_data = create_training_data(data)
    
    # Train model
    model, best_model = train_hgt_model(
        data, 
        split_data['train'], 
        split_data['val'],
        hidden_channels=64,
        out_channels=32,
        num_heads=8,
        num_layers=2,
        epochs=100,
        lr=0.001,
        group_weight=2.0  # Higher weight for group-related loss
    )
    
    
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('embeddings', exist_ok=True)
    
    # Save the model checkpoint locally
    checkpoint_path = 'checkpoints/hgt_best_model.pt'
    torch.save(best_model, checkpoint_path)
    print(f"Saved model checkpoint to {checkpoint_path}")
    
    # Generate final embeddings
    model.eval()
    with torch.no_grad():
        final_embeddings = model(data.x_dict, data.edge_index_dict)
    
    # Move embeddings to CPU
    final_embeddings = {
        node_type: embedding.cpu() for node_type, embedding in final_embeddings.items()
    }
    
    # Save embeddings locally
    embedding_path = 'embeddings/hgt_embeddings.pt'
    
    # Upload to GCS
    bucket_name = 'encite_user_embeddings'
    
    # Save and upload model checkpoint
    save_and_upload(
        best_model, 
        checkpoint_path, 
        bucket_name, 
        'checkpoints/hgt_best_model.pt'
    )
    
    # Save and upload embeddings
    save_and_upload(
        final_embeddings, 
        embedding_path,
        bucket_name, 
        'embeddings/hgt_embeddings.pt'
    )
    
    print("Model training and embedding export complete!")


if __name__ == "__main__":
    main()