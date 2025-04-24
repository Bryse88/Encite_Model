"""
Export embeddings from the trained Two-Tower model to Firestore or vector database.
"""

import os
import argparse
import yaml
import torch
import time
from tqdm import tqdm
from google.cloud import firestore
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

from model import create_model
from dataset import FirestoreDataLoader
from utils import get_device


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('export_embeddings')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export embeddings from trained Two-Tower model')
    parser.add_argument('--config', type=str, default='configs/two_tower_config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--export_to', type=str, choices=['firestore', 'pinecone', 'local'],
                        default='firestore', help='Where to export embeddings')
    parser.add_argument('--entity_type', type=str, choices=['user', 'place', 'both'],
                        default='both', help='Entity type to export')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for local exports')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def export_to_firestore(embeddings_dict, batch_size=500, collection_prefix=''):
    """
    Export embeddings to Firestore.
    
    Args:
        embeddings_dict: Dictionary of entity_id -> embedding
        batch_size: Number of documents to update in each batch
        collection_prefix: Prefix for collection names (for testing)
    """
    db = firestore.Client()
    
    # Process user and place embeddings separately
    for entity_type, embeddings in embeddings_dict.items():
        logger.info(f"Exporting {len(embeddings)} {entity_type} embeddings to Firestore...")
        
        # Create collection name
        collection = f"{collection_prefix}{entity_type}s"  # e.g., 'users', 'places'
        
        # Prepare for batch updates
        batch = db.batch()
        count = 0
        total = 0
        
        # Update each document
        for entity_id, embedding in tqdm(embeddings.items(), desc=f"Updating {entity_type} documents"):
            doc_ref = db.collection(collection).document(entity_id)
            
            # Convert embedding to list and store
            embedding_list = embedding.tolist()
            batch.update(doc_ref, {
                'embedding': embedding_list,
                'embedding_updated_at': firestore.SERVER_TIMESTAMP
            })
            
            count += 1
            total += 1
            
            # Commit batch when it reaches the batch size
            if count >= batch_size:
                batch.commit()
                batch = db.batch()
                count = 0
        
        # Commit any remaining updates
        if count > 0:
            batch.commit()
        
        logger.info(f"Saved {total} {entity_type} embeddings to Firestore")


def export_to_pinecone(embeddings_dict, api_key, index_name):
    """
    Export embeddings to Pinecone vector database.
    
    Args:
        embeddings_dict: Dictionary of entity_type -> {entity_id -> embedding}
        api_key: Pinecone API key
        index_name: Name of Pinecone index
    """
    import pinecone
    
    # Initialize Pinecone
    pinecone.init(api_key=api_key, environment="us-west1-gcp")  # Update with your environment
    
    # Check if index exists, create if not
    if index_name not in pinecone.list_indexes():
        # Get embedding dimension from first embedding
        first_type = next(iter(embeddings_dict))
        first_id = next(iter(embeddings_dict[first_type]))
        dim = len(embeddings_dict[first_type][first_id])
        
        # Create index
        pinecone.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine"
        )
    
    # Connect to index
    index = pinecone.Index(index_name)
    
    # Process user and place embeddings separately
    for entity_type, embeddings in embeddings_dict.items():
        logger.info(f"Exporting {len(embeddings)} {entity_type} embeddings to Pinecone...")
        
        # Prepare vectors for upsert
        vectors = []
        batch_size = 100  # Pinecone recommended batch size
        
        # Convert embeddings to vectors
        for entity_id, embedding in tqdm(embeddings.items(), desc=f"Preparing {entity_type} vectors"):
            vector_id = f"{entity_type}:{entity_id}"
            
            # Convert embedding to list
            vector = embedding.tolist()
            
            # Add metadata
            metadata = {
                "type": entity_type,
                "id": entity_id
            }
            
            vectors.append((vector_id, vector, metadata))
            
            # Upsert in batches
            if len(vectors) >= batch_size:
                index.upsert(vectors=[(id, vec, meta) for id, vec, meta in vectors])
                vectors = []
        
        # Upsert any remaining vectors
        if vectors:
            index.upsert(vectors=[(id, vec, meta) for id, vec, meta in vectors])
        
        logger.info(f"Saved {len(embeddings)} {entity_type} embeddings to Pinecone")


def export_to_local(embeddings_dict, output_dir):
    """
    Export embeddings to local files.
    
    Args:
        embeddings_dict: Dictionary of entity_type -> {entity_id -> embedding}
        output_dir: Directory to save embeddings
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Process user and place embeddings separately
    for entity_type, embeddings in embeddings_dict.items():
        logger.info(f"Exporting {len(embeddings)} {entity_type} embeddings to local files...")
        
        # Convert to tensor form for easier loading
        ids = list(embeddings.keys())
        embedding_tensors = torch.stack([embeddings[eid] for eid in ids])
        
        # Save embeddings
        embeddings_path = os.path.join(output_dir, f"{entity_type}_embeddings.pt")
        ids_path = os.path.join(output_dir, f"{entity_type}_ids.pt")
        
        torch.save(embedding_tensors, embeddings_path)
        torch.save(ids, ids_path)
        
        logger.info(f"Saved {entity_type} embeddings to {embeddings_path}")
        logger.info(f"Saved {entity_type} IDs to {ids_path}")


def generate_embeddings(model, features, device, batch_size=128, is_place=False):
    """
    Generate embeddings for entities using the appropriate tower.
    
    Args:
        model: Two-Tower model
        features: Dictionary of entity_id -> feature tensor
        device: Device to use
        batch_size: Batch size for processing
        is_place: Whether to use place tower (False for user tower)
    
    Returns:
        Dictionary of entity_id -> embedding tensor
    """
    model.eval()
    embeddings = {}
    
    # Process in batches to avoid OOM
    entity_ids = list(features.keys())
    
    for i in range(0, len(entity_ids), batch_size):
        batch_ids = entity_ids[i:i + batch_size]
        batch_features = torch.stack([features[eid] for eid in batch_ids]).to(device)
        
        with torch.no_grad():
            if is_place:
                batch_embeddings = model.get_place_embedding(batch_features)
            else:
                batch_embeddings = model.get_user_embedding(batch_features)
        
        # Store embeddings
        for j, eid in enumerate(batch_ids):
            embeddings[eid] = batch_embeddings[j].cpu()
    
    return embeddings


def main():
    """Main function."""
    args = parse_args()
    config = load_config(args.config)
    
    # Setup device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    data_loader = FirestoreDataLoader(cache_path=config.get('data', {}).get('cache_path'))
    user_features, place_features, _ = data_loader.load_all_data(
        limit=config.get('data', {}).get('limit')
    )
    
    # Create model
    logger.info("Creating model...")
    model_config = config.get('model', {})
    model_config['user_input_dim'] = data_loader.user_dim
    model_config['place_input_dim'] = data_loader.place_dim
    
    model = create_model(model_config)
    
    # Load trained model
    logger.info(f"Loading trained model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Generate embeddings
    embeddings_dict = {}
    
    if args.entity_type in ['user', 'both']:
        logger.info("Generating user embeddings...")
        user_embeddings = generate_embeddings(model, user_features, device, is_place=False)
        embeddings_dict['user'] = user_embeddings
    
    if args.entity_type in ['place', 'both']:
        logger.info("Generating place embeddings...")
        place_embeddings = generate_embeddings(model, place_features, device, is_place=True)
        embeddings_dict['place'] = place_embeddings
    
    # Export embeddings
    if args.export_to == 'firestore':
        export_to_firestore(
            embeddings_dict,
            batch_size=config.get('export', {}).get('batch_size', 500),
            collection_prefix=config.get('export', {}).get('collection_prefix', '')
        )
    elif args.export_to == 'pinecone':
        api_key = os.environ.get('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        index_name = config.get('export', {}).get('index_name', 'encite-embeddings')
        export_to_pinecone(embeddings_dict, api_key, index_name)
    elif args.export_to == 'local':
        output_dir = args.output_dir
        if not output_dir:
            output_dir = args.output_dir
        if not output_dir:
            output_dir = os.path.join('outputs', 'embeddings', time.strftime("%Y%m%d-%H%M%S"))
        
        export_to_local(embeddings_dict, output_dir)
    
    logger.info("Embedding export complete!")


if __name__ == '__main__':
    main()