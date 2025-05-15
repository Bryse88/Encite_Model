"""
Synthetic Interaction Data Generator for Encite

This script generates synthetic user-item interaction data for bootstrapping
the Two-Tower recommendation model training, using:
- User onboarding preferences
- Available places, events, and items

The generated data contains positive interactions based on preference matching
and negative interactions for contrast, all saved in a compatible format.
"""

import pandas as pd
import numpy as np
import json
import random
import ast
from typing import List, Dict, Any, Tuple, Set
import argparse
from pathlib import Path


def parse_list_or_dict(value: str):
    """
    Parse a string representation of a list or dictionary.
    """
    if pd.isna(value) or value == '':
        return []
    
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def read_user_data(filepath: str) -> pd.DataFrame:
    """
    Read and preprocess user onboarding data.
    """
    print(f"Reading user data from {filepath}")
    
    df = pd.read_csv(filepath)
    
     # ✅ Automatically add user_id if missing
    if 'user_id' not in df.columns:
        df.insert(0, 'user_id', range(len(df)))
    
    # For columns that might contain lists or dictionaries as strings
    list_columns = ['preferred_activities', 'preferred_vibes', 'dietary_restrictions']
    
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_or_dict)
    
    return df


def read_entity_data(places_path: str, events_path: str, items_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read and preprocess entity data (places, events, items).
    """
    print(f"Reading entity data from {places_path}, {events_path}, and {items_path}")
    
    places_df = pd.read_csv('data/places_rows.csv')
    events_df = pd.read_csv('data/events_rows.csv')
    items_df = pd.read_csv('data/items_rows.csv')
    
    # For columns that might contain lists or dictionaries as strings
    list_columns = ['categories', 'tags', 'features']
    
    # Process places
    for col in list_columns:
        if col in places_df.columns:
            places_df[col] = places_df[col].apply(parse_list_or_dict)
    
    # Process events
    for col in list_columns:
        if col in events_df.columns:
            events_df[col] = events_df[col].apply(parse_list_or_dict)
    
    # Process items
    for col in list_columns:
        if col in items_df.columns:
            items_df[col] = items_df[col].apply(parse_list_or_dict)
    
    return places_df, events_df, items_df


def create_activity_to_category_mapping() -> Dict[str, List[str]]:
    """
    Create a mapping from user activity preferences to entity categories.
    This mapping connects user preferences to relevant entity categories.
    """
    return {
        "Try new restaurants": ["restaurant", "food", "dining"],
        "Bar hopping": ["bar", "pub", "nightlife", "drinks"],
        "Coffee shops": ["cafe", "coffee", "bakery"],
        "Museums": ["museum", "art", "culture", "history"],
        "Live music": ["music", "concert", "live performance", "venue"],
        "Theater": ["theater", "performance", "arts"],
        "Art galleries": ["gallery", "art", "exhibition"],
        "Shopping": ["shopping", "retail", "store", "mall"],
        "Parks & outdoors": ["park", "outdoor", "nature", "hiking", "garden"],
        "Sporting events": ["sports", "game", "stadium", "arena"],
        "Fitness classes": ["fitness", "gym", "yoga", "workout"],
        "Group tours": ["tour", "sightseeing", "guided"],
        "Food festivals": ["festival", "food", "culinary", "market"],
        "Wine tasting": ["wine", "winery", "vineyard", "tasting"],
        "Brewery tours": ["brewery", "beer", "craft beer", "distillery"],
        "Dance clubs": ["club", "dance", "nightlife", "party"],
        "Comedy shows": ["comedy", "stand-up", "show"],
        "Cooking classes": ["cooking", "class", "culinary", "workshop"],
        "Board game cafes": ["board game", "game", "cafe"],
        "Movie theaters": ["cinema", "movie", "theater", "film"]
    }


def create_vibe_to_category_mapping() -> Dict[str, List[str]]:
    """
    Create a mapping from user vibe preferences to entity attributes.
    """
    return {
        "Chill": ["relaxed", "casual", "calm", "lounge", "cozy"],
        "Upscale": ["luxury", "fine dining", "upscale", "elegant", "exclusive"],
        "Adventurous": ["adventure", "outdoor", "exciting", "thrill", "extreme"],
        "Romantic": ["romantic", "intimate", "date night", "cozy", "charming"],
        "Family-friendly": ["family", "kid-friendly", "all-ages", "child"],
        "Intellectual": ["educational", "intellectual", "academic", "learning", "cultural"],
        "Social": ["social", "group", "community", "interactive", "networking"],
        "Artsy": ["art", "creative", "artistic", "bohemian", "gallery"],
        "Energetic": ["energetic", "lively", "busy", "active", "upbeat"],
        "Foodie": ["foodie", "culinary", "gourmet", "food", "dining"],
        "Historic": ["historic", "heritage", "traditional", "classic", "landmark"],
        "Modern": ["modern", "contemporary", "trendy", "innovative", "tech"],
        "Nature-focused": ["nature", "outdoor", "garden", "park", "scenic"],
        "Quirky": ["quirky", "unique", "unusual", "offbeat", "eccentric"],
        "Wellness": ["wellness", "health", "spa", "relaxation", "mindful"]
    }


def entity_matches_user_preferences(
    user_row: pd.Series, 
    entity_row: pd.Series, 
    entity_type: str,
    activity_map: Dict[str, List[str]],
    vibe_map: Dict[str, List[str]]
) -> float:
    """
    Calculate how well an entity matches a user's preferences.
    Returns a score between 0 and 1.
    """
    match_score = 0.0
    matches_found = 0
    
    # Get user preferences
    user_activities = user_row.get('preferred_activities', [])
    user_vibes = user_row.get('preferred_vibes', [])
    dietary_restrictions = user_row.get('dietary_restrictions', [])
    
    # Get entity attributes
    entity_categories = []
    entity_tags = []
    
    if 'categories' in entity_row and not pd.isna(entity_row['categories']):
        entity_categories = entity_row['categories'] if isinstance(entity_row['categories'], list) else [entity_row['categories']]
    
    # ✅ FIXED VERSION
    if 'tags' in entity_row and entity_row['tags'] is not None and not isinstance(entity_row['tags'], float):
        entity_tags = entity_row['tags'] if isinstance(entity_row['tags'], list) else [entity_row['tags']]

    
    # Flatten categories and tags for easier matching
    entity_attributes = [attr.lower() for attr in entity_categories + entity_tags if isinstance(attr, str)]
    
    # Match activities to categories
    for activity in user_activities:
        if activity in activity_map:
            relevant_categories = activity_map[activity]
            
            # Check if any relevant category is in entity attributes
            for category in relevant_categories:
                if any(category.lower() in attr or attr in category.lower() for attr in entity_attributes):
                    match_score += 1.0
                    matches_found += 1
                    break
    
    # Match vibes to entity attributes
    for vibe in user_vibes:
        if vibe in vibe_map:
            relevant_attributes = vibe_map[vibe]
            
            # Check if any relevant attribute is in entity attributes
            for attr in relevant_attributes:
                if any(attr.lower() in entity_attr or entity_attr in attr.lower() for entity_attr in entity_attributes):
                    match_score += 1.0
                    matches_found += 1
                    break
    
    # Handle dietary restrictions for food-related entities
    if entity_type in ['place', 'item'] and len(dietary_restrictions) > 0:
        # Check if entity has dietary information
        dietary_compatible = True
        
        # In real data you would check specific dietary fields
        # For synthetic data, we'll just assume some places don't match dietary needs
        if random.random() < 0.2 and entity_type == 'place':
            dietary_compatible = False
        
        if dietary_compatible:
            match_score += 1.0
            matches_found += 1
    
    # Calculate final normalized score
    if matches_found == 0:
        return 0.0
    
    return match_score / matches_found


def generate_synthetic_interactions(
    users_df: pd.DataFrame,
    places_df: pd.DataFrame,
    events_df: pd.DataFrame,
    items_df: pd.DataFrame,
    min_pos: int = 3,
    max_pos: int = 10,
    neg_samples: int = 4
) -> List[Dict[str, Any]]:
    """
    Generate synthetic interactions between users and entities.
    
    Args:
        users_df: DataFrame with user data
        places_df: DataFrame with place data
        events_df: DataFrame with event data
        items_df: DataFrame with item data
        min_pos: Minimum number of positive interactions per user
        max_pos: Maximum number of positive interactions per user
        neg_samples: Number of negative interactions per user
    
    Returns:
        List of interaction dictionaries
    """
    print("Generating synthetic interactions")
    
    interactions = []
    
    # Create category mappings
    activity_map = create_activity_to_category_mapping()
    vibe_map = create_vibe_to_category_mapping()
    
    # Process each user
    for _, user in users_df.iterrows():
        user_id = user['user_id']
        
        # Track entities that have been interacted with
        interacted_entities = set()
        
        # Generate positive interactions
        num_pos_interactions = random.randint(min_pos, max_pos)
        
        # Pool of potential entities to match with user
        place_scores = []
        event_scores = []
        item_scores = []
        
        # Score match potential for places
        for _, place in places_df.iterrows():
            score = entity_matches_user_preferences(user, place, 'place', activity_map, vibe_map)
            if score > 0:
                place_scores.append(('place', place['id'], score))
        
        # Score match potential for events
        for _, event in events_df.iterrows():
            score = entity_matches_user_preferences(user, event, 'event', activity_map, vibe_map)
            if score > 0:
                event_scores.append(('event', event['id'], score))
        
        # Score match potential for items
        for _, item in items_df.iterrows():
            score = entity_matches_user_preferences(user, item, 'item', activity_map, vibe_map)
            if score > 0:
                item_scores.append(('item', item['id'], score))
        
        # Combine all scores and take the highest
        all_scores = place_scores + event_scores + item_scores
        
        # Sort by score (highest first)
        all_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Take top matches as positive interactions
        pos_count = 0
        for entity_type, entity_id, _ in all_scores:
            if pos_count >= num_pos_interactions:
                break
                
            # Create positive interaction
            interaction = {
                "user_id": user_id,
                "entity_id": entity_id,
                "entity_type": entity_type,
                "label": 1  # Positive interaction
            }
            
            interactions.append(interaction)
            interacted_entities.add((entity_type, entity_id))
            pos_count += 1
        
        # If we don't have enough positive interactions, add some random ones
        while pos_count < num_pos_interactions:
            entity_type = random.choice(['place', 'event', 'item'])
            
            if entity_type == 'place' and len(places_df) > 0:
                entity_id = random.choice(places_df['id'].tolist())
            elif entity_type == 'event' and len(events_df) > 0:
                entity_id = random.choice(events_df['id'].tolist())
            elif entity_type == 'item' and len(items_df) > 0:
                entity_id = random.choice(items_df['id'].tolist())
            else:
                continue
                
            if (entity_type, entity_id) not in interacted_entities:
                # Create positive interaction
                interaction = {
                    "user_id": user_id,
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "label": 1  # Positive interaction
                }
                
                interactions.append(interaction)
                interacted_entities.add((entity_type, entity_id))
                pos_count += 1
        
        # Generate negative interactions (mismatches)
        neg_count = 0
        
        # Get low-scoring entities first
        low_scores = sorted(all_scores, key=lambda x: x[2])[:20]  # Take 20 worst matches
        
        # Sample from low scores first
        while neg_count < neg_samples and len(low_scores) > 0:
            entity_type, entity_id, _ = low_scores.pop(0)
            
            if (entity_type, entity_id) not in interacted_entities:
                # Create negative interaction
                interaction = {
                    "user_id": user_id,
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "label": 0  # Negative interaction
                }
                
                interactions.append(interaction)
                interacted_entities.add((entity_type, entity_id))
                neg_count += 1
        
        # If we still need more negative samples, choose random entities
        while neg_count < neg_samples:
            entity_type = random.choice(['place', 'event', 'item'])
            
            if entity_type == 'place' and len(places_df) > 0:
                entity_id = random.choice(places_df['id'].tolist())
            elif entity_type == 'event' and len(events_df) > 0:
                entity_id = random.choice(events_df['id'].tolist())
            elif entity_type == 'item' and len(items_df) > 0:
                entity_id = random.choice(items_df['id'].tolist())
            else:
                continue
                
            if (entity_type, entity_id) not in interacted_entities:
                # Create negative interaction
                interaction = {
                    "user_id": user_id,
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "label": 0  # Negative interaction
                }
                
                interactions.append(interaction)
                interacted_entities.add((entity_type, entity_id))
                neg_count += 1
    
    return interactions


def analyze_interactions(interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the generated interactions to verify balance and distribution.
    """
    total = len(interactions)
    positive = sum(1 for i in interactions if i['label'] == 1)
    negative = total - positive
    
    entity_types = {
        'place': sum(1 for i in interactions if i['entity_type'] == 'place'),
        'event': sum(1 for i in interactions if i['entity_type'] == 'event'),
        'item': sum(1 for i in interactions if i['entity_type'] == 'item')
    }
    
    unique_users = len(set(i['user_id'] for i in interactions))
    unique_entities = len(set((i['entity_type'], i['entity_id']) for i in interactions))
    
    return {
        'total_interactions': total,
        'positive_interactions': positive,
        'negative_interactions': negative,
        'positive_pct': round(positive / total * 100, 2),
        'negative_pct': round(negative / total * 100, 2),
        'entity_type_distribution': {
            'place': round(entity_types['place'] / total * 100, 2),
            'event': round(entity_types['event'] / total * 100, 2),
            'item': round(entity_types['item'] / total * 100, 2)
        },
        'unique_users': unique_users,
        'unique_entities': unique_entities,
        'interactions_per_user': round(total / unique_users, 2)
    }


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic user-item interaction data for Encite')
    
    parser.add_argument('--users', type=str, default='data/encite_onboarding_data_randomized_density.csv',
                        help='Path to user onboarding CSV file')
    parser.add_argument('--places', type=str, default='data/places_rows.csv',
                        help='Path to places CSV file')
    parser.add_argument('--events', type=str, default='data/events_rows.csv',
                        help='Path to events CSV file')
    parser.add_argument('--items', type=str, default='data/items_rows.csv',
                        help='Path to items CSV file')
    parser.add_argument('--output', type=str, default='synthetic_interactions.json',
                        help='Path to output JSON file')
    parser.add_argument('--min-pos', type=int, default=3,
                        help='Minimum number of positive interactions per user')
    parser.add_argument('--max-pos', type=int, default=10,
                        help='Maximum number of positive interactions per user')
    parser.add_argument('--neg-samples', type=int, default=4,
                        help='Number of negative interactions per user')
    
    args = parser.parse_args()
    
    # Read data
    users_df = read_user_data(args.users)
    places_df, events_df, items_df = read_entity_data(args.places, args.events, args.items)
    
    # Print basic statistics
    print(f"Loaded {len(users_df)} users")
    print(f"Loaded {len(places_df)} places")
    print(f"Loaded {len(events_df)} events")
    print(f"Loaded {len(items_df)} items")
    
    # Generate interactions
    interactions = generate_synthetic_interactions(
        users_df, 
        places_df, 
        events_df, 
        items_df,
        min_pos=args.min_pos,
        max_pos=args.max_pos,
        neg_samples=args.neg_samples
    )
    
    # Analyze and print statistics
    stats = analyze_interactions(interactions)
    print("\nInteraction Statistics:")
    print(f"Total interactions: {stats['total_interactions']}")
    print(f"Positive interactions: {stats['positive_interactions']} ({stats['positive_pct']}%)")
    print(f"Negative interactions: {stats['negative_interactions']} ({stats['negative_pct']}%)")
    print("\nEntity type distribution:")
    for entity_type, pct in stats['entity_type_distribution'].items():
        print(f"  - {entity_type}: {pct}%")
    print(f"\nUnique users: {stats['unique_users']}")
    print(f"Unique entities: {stats['unique_entities']}")
    print(f"Average interactions per user: {stats['interactions_per_user']}")
    
    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(interactions, f, indent=2)
    
    print(f"\nSaved {len(interactions)} synthetic interactions to {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()