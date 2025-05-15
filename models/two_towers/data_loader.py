import json
import torch
import pandas as pd
from pathlib import Path
from dataset import FirestoreDataLoader
from dataset import create_context_features
from typing import Tuple, Dict

def load_synthetic_dataset_or_firestore(config, device) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], list]:
    """
    Load user features, place features, and interactions from either synthetic JSON or Firestore,
    depending on what's defined in the config.
    
    Args:
        config (dict): Configuration dictionary
        device (torch.device): Torch device to load tensors onto
    
    Returns:
        Tuple of (user_features, place_features, interactions)
    """
    synthetic_path = config.get("data", {}).get("synthetic_path", None)

    if synthetic_path and Path(synthetic_path).exists():
        print(f"🔁 Loading synthetic interactions from {synthetic_path}")
        # Load interaction list
        with open(synthetic_path, 'r') as f:
            interactions = json.load(f)

        # Extract IDs
        user_ids = sorted(list(set([x["user_id"] for x in interactions])))
        place_ids = sorted(list(set([x["entity_id"] for x in interactions if x["entity_type"] == "place"])))

        # Load user features
        user_df = pd.read_csv("data/encite_onboarding_data_randomized_density.csv")
        user_features = {}
        for _, row in user_df.iterrows():
            # user_id = row["user_id"] if "user_id" in row else str(row["user"])
            user_id = row["user_id"]

            feats = []
            if "preferred_activities" in row:
                feats += [len(eval(row["preferred_activities"]))]  # crude feature
            if "preferred_vibes" in row:
                feats += [len(eval(row["preferred_vibes"]))]
            feats += [row.get("schedule_density", 3), row.get("max_distance", 1000)]
            user_features[user_id] = torch.tensor(feats, dtype=torch.float).to(device)

        # Load place features
        place_df = pd.read_csv("data/places_rows.csv")
        place_features = {}
        for _, row in place_df.iterrows():
            place_id = row["id"] if "id" in row else row["place_id"]
            feats = [row.get("price_level", 2), row.get("rating", 3), row.get("popularity_score", 0.5)]
            place_features[place_id] = torch.tensor(feats, dtype=torch.float).to(device)

        # Format interactions
        # formatted_interactions = [
        #     (i["user_id"], i["entity_id"], float(i["label"]))
        #     for i in interactions if i["entity_type"] == "place"
        # ]
        # Filter interactions to only those with valid user & place features
        formatted_interactions = [
            (i["user_id"], i["entity_id"], float(i["label"]))
            for i in interactions
            if i["entity_type"] == "place"
            and i["user_id"] in user_features
            and i["entity_id"] in place_features
        ]


        return user_features, place_features, formatted_interactions

    else:
        print("🌐 Falling back to Firestore loader...")
        data_loader = FirestoreDataLoader(cache_path=config.get("data", {}).get("cache_path"))
        return data_loader.load_all_data(limit=config.get("data", {}).get("limit"))
