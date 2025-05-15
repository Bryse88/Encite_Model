#!/usr/bin/env python3
# PyTorch Geometric HeteroData Graph Builder for Social Planning App

"""
Graph Builder Module for Social Planning App
-------------------------------------------
This module handles the construction of a heterogeneous graph data structure
for a social planning application using PyTorch Geometric.

The graph represents relationships between users, places, events, and items,
enabling recommendation and planning features.

Dependencies:
- PyTorch Geometric: For heterogeneous graph data structures
- Pandas: For data manipulation and analysis
- NumPy: For numerical operations
- scikit-learn: For data preprocessing
- tqdm: For progress tracking
"""

# Standard library imports
import os  # For file and directory operations
import ast  # For safely evaluating string literals
import json  # For JSON file handling
import re  # For regular expressions
import random  # For random operations

# Third-party imports
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
from tqdm import tqdm  # For progress bars
import torch  # PyTorch for tensor operations
from torch_geometric.data import HeteroData  # For heterogeneous graph data structure
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer  # For data preprocessing

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

def load_datasets():
    """Load all required datasets"""
    print("Loading datasets...")
    places_df = pd.read_csv('data/places_rows.csv')
    events_df = pd.read_csv('data/events_rows.csv')
    items_df = pd.read_csv('data/items_rows.csv')
    users_df = pd.read_csv('data/encite_onboarding_data_randomized_density.csv')

    
    # Clean and preprocess users data
    users_df['experience_vibes'] = users_df['experience_vibes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    users_df['activities'] = users_df['activities'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    users_df['location_priorities'] = users_df['location_priorities'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    users_df['user_id'] = users_df.index.astype(str)  # Create user_id column
    
    # Clean and preprocess places data
    places_df['tags'] = places_df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else [])
    
    # Clean and preprocess events data
    events_df['tags'] = events_df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else [])
    
    return places_df, events_df, items_df, users_df

def create_node_mappings(places_df, events_df, items_df, users_df):
    """Create mappings from IDs to indices for all node types"""
    print("Creating node mappings...")
    
    # Create mappings
    user_mapping = {user_id: idx for idx, user_id in enumerate(users_df['user_id'])}
    place_mapping = {place_id: idx for idx, place_id in enumerate(places_df['id'])}
    event_mapping = {event_id: idx for idx, event_id in enumerate(events_df['id'])}
    item_mapping = {item_id: idx for idx, item_id in enumerate(items_df['id'])}
    
    # Save mappings
    mappings = {
        'user_mapping': user_mapping,
        'place_mapping': place_mapping,
        'event_mapping': event_mapping,
        'item_mapping': item_mapping
    }
    
    with open('data/node_mappings.json', 'w') as f:
        json.dump(mappings, f)
    
    return user_mapping, place_mapping, event_mapping, item_mapping

def extract_all_categories_and_tags(places_df, events_df, items_df):
    """Extract unique categories and tags from places, events, and items"""
    print("Extracting categories and tags...")
    
    # Extract categories
    place_categories = set(places_df['category'].dropna().unique())
    event_categories = set(events_df['category'].dropna().unique())
    item_categories = set(items_df['category'].dropna().unique())
    
    # Extract tags
    place_tags = set()
    for tags_list in places_df['tags'].dropna():
        if isinstance(tags_list, list):
            place_tags.update(tags_list)
    
    event_tags = set()
    for tags_list in events_df['tags'].dropna():
        if isinstance(tags_list, list):
            event_tags.update(tags_list)
    
    # Combine all categories and tags
    all_categories_tags = list(place_categories | event_categories | item_categories | place_tags | event_tags)
    
    # Create dictionary mapping from category/tag to index
    category_tag_mapping = {cat_tag: idx for idx, cat_tag in enumerate(all_categories_tags)}
    
    with open('data/category_tag_mapping.json', 'w') as f:
        json.dump(category_tag_mapping, f)
    
    return category_tag_mapping, all_categories_tags

def extract_user_preferences(users_df):
    """Extract user preferences from experience_vibes and activities"""
    print("Extracting user preferences...")
    
    # Extract unique vibes and activities
    all_vibes = set()
    all_activities = set()
    
    for vibes in users_df['experience_vibes']:
        if isinstance(vibes, list):
            all_vibes.update(vibes)
    
    for activities in users_df['activities']:
        if isinstance(activities, list):
            all_activities.update(activities)
    
    vibe_mapping = {vibe: idx for idx, vibe in enumerate(all_vibes)}
    activity_mapping = {activity: idx for idx, activity in enumerate(all_activities)}
    
    # Save mappings
    with open('data/vibe_mapping.json', 'w') as f:
        json.dump(vibe_mapping, f)
    
    with open('data/activity_mapping.json', 'w') as f:
        json.dump(activity_mapping, f)
    
    return vibe_mapping, activity_mapping

def create_vibe_to_category_mapping():
    """Create a mapping from vibes to relevant categories/tags"""
    print("Creating vibe to category mapping...")
    
    vibe_to_category = {
        'Chill & Laid-back': ['cafe', 'park', 'library', 'bookstore', 'book_store', 'spa', 'museum'],
        'Social & Outgoing': ['restaurant', 'bar', 'night_club', 'stadium', 'event', 'shopping_mall'],
        'Loud & Energetic': ['night_club', 'bar', 'stadium', 'amusement_park', 'bowling_alley'],
        'Intimate': ['cafe', 'restaurant', 'spa', 'art_gallery'],
        'Educational & Cultural': ['museum', 'art_gallery', 'library', 'university', 'tourist_attraction'],
        'Adventurous & Active': ['park', 'gym', 'amusement_park', 'zoo', 'bowling_alley', 'stadium']
    }
    
    activity_to_category = {
        'Try new restaurants / cafes': ['restaurant', 'cafe', 'bakery', 'meal_takeaway', 'meal_delivery'],
        'Go to live events': ['stadium', 'night_club', 'bar', 'movie_theater', 'tourist_attraction'],
        'Explore parks or nature': ['park', 'zoo', 'tourist_attraction'],
        'Hosting or joining game/movie nights': ['movie_theater', 'bowling_alley', 'night_club'],
        'Shopping': ['shopping_mall', 'store', 'department_store', 'book_store', 'shoe_store'],
        'Fitness activities': ['gym', 'park', 'health'],
        'Seeking unique local experiences': ['tourist_attraction', 'museum', 'art_gallery']
    }
    
    with open('data/vibe_to_category.json', 'w') as f:
        json.dump(vibe_to_category, f)
    
    with open('data/activity_to_category.json', 'w') as f:
        json.dump(activity_to_category, f)
    
    return vibe_to_category, activity_to_category

def create_user_node_features(users_df, vibe_mapping, activity_mapping):
    """Create user node features"""
    print("Creating user node features...")
    
    num_users = len(users_df)
    num_vibes = len(vibe_mapping)
    num_activities = len(activity_mapping)
    
    # Initialize feature arrays
    budget_features = np.zeros(num_users)  # Assuming user budget is related to their location_priorities['Cost / Budget']
    density_features = np.zeros(num_users)
    vibe_features = np.zeros((num_users, num_vibes))
    activity_features = np.zeros((num_users, num_activities))
    
    for idx, user in users_df.iterrows():
        # Budget feature - extract from location_priorities if available
        if isinstance(user['location_priorities'], dict) and 'Cost / Budget' in user['location_priorities']:
            budget_features[idx] = 1.0 - user['location_priorities']['Cost / Budget']  # Inverse because lower priority means higher budget
        
        # Density feature
        density_features[idx] = user['schedule_density'] if not pd.isna(user['schedule_density']) else 3  # Default to middle value
        
        # Vibe features
        if isinstance(user['experience_vibes'], list):
            for vibe in user['experience_vibes']:
                if vibe in vibe_mapping:
                    vibe_features[idx, vibe_mapping[vibe]] = 1.0
        
        # Activity features
        if isinstance(user['activities'], list):
            for activity in user['activities']:
                if activity in activity_mapping:
                    activity_features[idx, activity_mapping[activity]] = 1.0
    
    # Normalize features
    budget_scaler = StandardScaler()
    density_scaler = StandardScaler()
    
    normalized_budget = budget_scaler.fit_transform(budget_features.reshape(-1, 1)).flatten()
    normalized_density = density_scaler.fit_transform(density_features.reshape(-1, 1)).flatten()
    
    # Combine all features
    user_features = np.column_stack([
        normalized_budget, 
        normalized_density,
        vibe_features,
        activity_features
    ])
    
    return torch.tensor(user_features, dtype=torch.float)

def create_place_node_features(places_df, category_tag_mapping):
    """Create place node features"""
    print("Creating place node features...")
    
    num_places = len(places_df)
    num_categories_tags = len(category_tag_mapping)
    
    # Initialize feature arrays
    popularity_features = np.zeros(num_places)
    price_features = np.zeros(num_places)
    category_tag_features = np.zeros((num_places, num_categories_tags))
    
    for idx, place in places_df.iterrows():
        # Popularity feature
        popularity_features[idx] = float(place['popularity_score']) if not pd.isna(place['popularity_score']) else 5.0
        
        # Price feature
        price_features[idx] = float(place['price_level']) if not pd.isna(place['price_level']) else 2.0
        
        # Category and tags features
        if not pd.isna(place['category']):
            category = place['category']
            if category in category_tag_mapping:
                category_tag_features[idx, category_tag_mapping[category]] = 1.0
        
        if isinstance(place['tags'], list):
            for tag in place['tags']:
                if tag in category_tag_mapping:
                    category_tag_features[idx, category_tag_mapping[tag]] = 1.0
    
    # Normalize features
    popularity_scaler = StandardScaler()
    price_scaler = StandardScaler()
    
    normalized_popularity = popularity_scaler.fit_transform(popularity_features.reshape(-1, 1)).flatten()
    normalized_price = price_scaler.fit_transform(price_features.reshape(-1, 1)).flatten()
    
    # Combine all features
    place_features = np.column_stack([
        normalized_popularity,
        normalized_price,
        category_tag_features
    ])
    
    return torch.tensor(place_features, dtype=torch.float)

def create_event_node_features(events_df, category_tag_mapping):
    """Create event node features"""
    print("Creating event node features...")
    
    num_events = len(events_df)
    num_categories_tags = len(category_tag_mapping)
    
    # Initialize feature arrays
    price_features = np.zeros(num_events)
    category_tag_features = np.zeros((num_events, num_categories_tags))
    
    for idx, event in events_df.iterrows():
        # Price feature
        price_features[idx] = float(event['price']) if not pd.isna(event['price']) else 15.0
        
        # Category and tags features
        if not pd.isna(event['category']):
            category = event['category']
            if category in category_tag_mapping:
                category_tag_features[idx, category_tag_mapping[category]] = 1.0
        
        if isinstance(event['tags'], list):
            for tag in event['tags']:
                if tag in category_tag_mapping:
                    category_tag_features[idx, category_tag_mapping[tag]] = 1.0
    
    # Normalize features
    price_scaler = StandardScaler()
    normalized_price = price_scaler.fit_transform(price_features.reshape(-1, 1)).flatten()
    
    # Combine all features
    event_features = np.column_stack([
        normalized_price,
        category_tag_features
    ])
    
    return torch.tensor(event_features, dtype=torch.float)

def create_item_node_features(items_df, category_tag_mapping):
    """Create item node features"""
    print("Creating item node features...")
    
    num_items = len(items_df)
    num_categories_tags = len(category_tag_mapping)
    
    # Initialize feature arrays
    popularity_features = np.zeros(num_items)
    price_features = np.zeros(num_items)
    category_features = np.zeros((num_items, num_categories_tags))
    
    for idx, item in items_df.iterrows():
        # Popularity feature
        popularity_features[idx] = float(item['popularity_score']) if not pd.isna(item['popularity_score']) else 5.0
        
        # Price feature
        price_features[idx] = float(item['price']) if not pd.isna(item['price']) else 10.0
        
        # Category feature
        if not pd.isna(item['category']):
            category = item['category'].lower()
            # Map item categories to general categories if needed
            category_mapping = {
                'beverage': 'restaurant',
                'food': 'restaurant',
                'beer': 'bar',
                'cake': 'bakery',
                'pastry': 'bakery',
                'spirit': 'bar',
                'main course': 'restaurant',
                'dessert': 'restaurant',
                'wine': 'bar',
                'appetizer': 'restaurant',
                'cocktail': 'bar',
                'bread': 'bakery'
            }
            
            mapped_category = category_mapping.get(category, category)
            if mapped_category in category_tag_mapping:
                category_features[idx, category_tag_mapping[mapped_category]] = 1.0
    
    # Normalize features
    popularity_scaler = StandardScaler()
    price_scaler = StandardScaler()
    
    normalized_popularity = popularity_scaler.fit_transform(popularity_features.reshape(-1, 1)).flatten()
    normalized_price = price_scaler.fit_transform(price_features.reshape(-1, 1)).flatten()
    
    # Combine all features
    item_features = np.column_stack([
        normalized_popularity,
        normalized_price,
        category_features
    ])
    
    return torch.tensor(item_features, dtype=torch.float)

def create_user_likes_edges(users_df, places_df, events_df, items_df, 
                           user_mapping, place_mapping, event_mapping, item_mapping,
                           vibe_to_category, activity_to_category, category_tag_mapping):
    """Create edges for user likes relationships"""
    print("Creating user-likes-business edges...")
    
    user_likes_place_src = []
    user_likes_place_dst = []
    
    user_likes_event_src = []
    user_likes_event_dst = []
    
    user_likes_item_src = []
    user_likes_item_dst = []
    
    for user_idx, user in tqdm(users_df.iterrows(), total=len(users_df)):
        user_id = user['user_id']
        
        # Get user preferences
        user_vibes = user['experience_vibes'] if isinstance(user['experience_vibes'], list) else []
        user_activities = user['activities'] if isinstance(user['activities'], list) else []
        
        # Collect all relevant categories for this user
        relevant_categories = set()
        for vibe in user_vibes:
            if vibe in vibe_to_category:
                relevant_categories.update(vibe_to_category[vibe])
        
        for activity in user_activities:
            if activity in activity_to_category:
                relevant_categories.update(activity_to_category[activity])
        
        # Match places based on categories
        for place_idx, place in places_df.iterrows():
            place_id = place['id']
            place_category = place['category']
            place_tags = place['tags'] if isinstance(place['tags'], list) else []
            
            # Check if place category or tags match user preferences
            if place_category in relevant_categories or any(tag in relevant_categories for tag in place_tags):
                user_likes_place_src.append(user_mapping[user_id])
                user_likes_place_dst.append(place_mapping[place_id])
        
        # Match events based on categories
        for event_idx, event in events_df.iterrows():
            event_id = event['id']
            event_category = event['category']
            event_tags = event['tags'] if isinstance(event['tags'], list) else []
            
            # Check if event category or tags match user preferences
            if event_category in relevant_categories or any(tag in relevant_categories for tag in event_tags):
                user_likes_event_src.append(user_mapping[user_id])
                user_likes_event_dst.append(event_mapping[event_id])
        
        # Match items based on categories
        for item_idx, item in items_df.iterrows():
            item_id = item['id']
            item_category = item['category'].lower() if not pd.isna(item['category']) else ""
            
            # Map item categories to general categories
            category_mapping = {
                'beverage': 'restaurant',
                'food': 'restaurant',
                'beer': 'bar',
                'cake': 'bakery',
                'pastry': 'bakery',
                'spirit': 'bar',
                'main course': 'restaurant',
                'dessert': 'restaurant',
                'wine': 'bar',
                'appetizer': 'restaurant',
                'cocktail': 'bar',
                'bread': 'bakery'
            }
            
            mapped_category = category_mapping.get(item_category, item_category)
            
            # Check if mapped category matches user preferences
            if mapped_category in relevant_categories:
                user_likes_item_src.append(user_mapping[user_id])
                user_likes_item_dst.append(item_mapping[item_id])
    
    # Create edge index tensors
    user_likes_place_edge_index = torch.tensor([user_likes_place_src, user_likes_place_dst], dtype=torch.long)
    user_likes_event_edge_index = torch.tensor([user_likes_event_src, user_likes_event_dst], dtype=torch.long)
    user_likes_item_edge_index = torch.tensor([user_likes_item_src, user_likes_item_dst], dtype=torch.long)
    
    return user_likes_place_edge_index, user_likes_event_edge_index, user_likes_item_edge_index

def build_hetero_data(user_features, place_features, event_features, item_features,
                     user_likes_place_edge_index, user_likes_event_edge_index, user_likes_item_edge_index):
    """Build the final heterogeneous graph"""
    print("Building heterogeneous graph...")
    
    data = HeteroData()
    
    # Add node features
    data['user'].x = user_features
    data['place'].x = place_features
    data['event'].x = event_features
    data['item'].x = item_features
    
    # Add edge indices
    data['user', 'likes', 'place'].edge_index = user_likes_place_edge_index
    data['user', 'likes', 'event'].edge_index = user_likes_event_edge_index
    data['user', 'likes', 'item'].edge_index = user_likes_item_edge_index
    
    return data

def main():
    """Main function to build the heterogeneous graph"""
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load datasets
    places_df, events_df, items_df, users_df = load_datasets()
    
    # Create node mappings
    user_mapping, place_mapping, event_mapping, item_mapping = create_node_mappings(
        places_df, events_df, items_df, users_df
    )
    
    # Extract categories and tags
    category_tag_mapping, all_categories_tags = extract_all_categories_and_tags(
        places_df, events_df, items_df
    )
    
    # Extract user preferences
    vibe_mapping, activity_mapping = extract_user_preferences(users_df)
    
    # Create vibe to category mapping
    vibe_to_category, activity_to_category = create_vibe_to_category_mapping()
    
    # Create node features
    user_features = create_user_node_features(users_df, vibe_mapping, activity_mapping)
    place_features = create_place_node_features(places_df, category_tag_mapping)
    event_features = create_event_node_features(events_df, category_tag_mapping)
    item_features = create_item_node_features(items_df, category_tag_mapping)
    
    # Create edges
    user_likes_place_edge_index, user_likes_event_edge_index, user_likes_item_edge_index = create_user_likes_edges(
        users_df, places_df, events_df, items_df, 
        user_mapping, place_mapping, event_mapping, item_mapping,
        vibe_to_category, activity_to_category, category_tag_mapping
    )
    
    # Build the heterogeneous graph
    data = build_hetero_data(
        user_features, place_features, event_features, item_features,
        user_likes_place_edge_index, user_likes_event_edge_index, user_likes_item_edge_index
    )
    
    # Save the graph
    print("Saving graph to data/graph_cache.pt")
    torch.save(data, 'data/graph_cache.pt')
    
    # Print graph summary
    print("\nGraph summary:")
    print(f"Number of users: {data['user'].x.size(0)}")
    print(f"Number of places: {data['place'].x.size(0)}")
    print(f"Number of events: {data['event'].x.size(0)}")
    print(f"Number of items: {data['item'].x.size(0)}")
    print(f"Number of user-likes-place edges: {data['user', 'likes', 'place'].edge_index.size(1)}")
    print(f"Number of user-likes-event edges: {data['user', 'likes', 'event'].edge_index.size(1)}")
    print(f"Number of user-likes-item edges: {data['user', 'likes', 'item'].edge_index.size(1)}")
    print("\nGraph construction completed successfully!")

if __name__ == "__main__":
    main()