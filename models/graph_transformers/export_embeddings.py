"""
Export embeddings from a trained HGT model to Firestore or vector database.
"""

import os
import argparse
import yaml
import torch
import time
from tqdm import tqdm
from google.cloud import firestore
import pinecone

from model import create_model
from dataset_loader import load_graph
from utils import get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export embeddings from trained HGT model')
    parser.add_argument('--config', type=str, default='configs/hgt_config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for embeddings (if saving to local files)')
    parser.add_argument('--export_to', type=str, choices=['firestore', 'pinecone', 'local'],
                        default='firestore', help='Where to export embeddings')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def export_to_firestore(embeddings_dict, mapping_dict, batch_size=500):
    """
    Export embeddings to Firestore.
    
    Args:
        embeddings_dict: Dictionary of embeddings for each node type
        mapping_dict: Dictionary of node mappings (index to Firestore ID)
        batch_size: Number of documents to update in each batch
    """
    db = firestore.Client()
    
    for node_type, embeddings in embeddings_dict.items():
        print(f"Exporting {node_type} embeddings to Firestore...")
        
        # Get mapping for this node type
        node_mapping = mapping_dict.get(node_type, {})
        if not node_mapping:
            print(f"Warning: No mapping found for {node_type}")
            continue
        
        # Create inverse mapping
        inv_mapping = {v: k for k, v in node_mapping.items()}
        
        # Prepare for batch updates
        batch = db.batch()
        count = 0
        total = 0
        
        # Create collection name based on node type
        collection = node_type + 's'  # e.g., 'user' -> 'users'
        
        # Update each document
        for idx in tqdm(range(embeddings.shape[0]), desc=f"Updating {node_type} documents"):
            if idx in inv_mapping:
                doc_id = inv_mapping[idx]
                doc_ref = db.collection(collection).document(doc_id)
                
                # Convert embedding to list and store
                embedding_list = embeddings[idx].tolist()
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
        
        print(f"Saved {total} {node_type} embeddings to Firestore")


def export_to_pinecone(embeddings_dict, mapping_dict, index_name, api_key):
    """
    Export embeddings to Pinecone vector database.
    
    Args:
        embeddings_dict: Dictionary of embeddings for each node type
        mapping_dict: Dictionary of node mappings (index to Firestore ID)
        index_name: Name of Pinecone index
        api_key: Pinecone API key
    """
    # Initialize Pinecone
    pinecone.init(api_key=api_key, environment="us-west1-gcp")  # Update with your environment
    
    # Check if index exists, create if not
    if index_name not in pinecone.list_indexes():
        # Get embedding dimension from first embedding
        first_type = next(iter(embeddings_dict))
        dim = embeddings_dict[first_type].shape[1]
        
        # Create index
        pinecone.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine"
        )
    
    # Connect to index
    index = pinecone.Index(index_name)
    
    for node_type, embeddings in embeddings_dict.items():
        print(f"Exporting {node_type} embeddings to Pinecone...")
        
        # Get mapping for this node type
        node_mapping = mapping_dict.get(node_type, {})
        if not node_mapping:
            print(f"Warning: No mapping found for {node_type}")
            continue
        
        # Create inverse mapping
        inv_mapping = {v: k for k, v in node_mapping.items()}
        
        # Prepare vectors for upsert
        vectors = []
        batch_size = 100  # Pinecone recommended batch size
        
        # Convert embeddings to vectors
        for idx in tqdm(range(embeddings.shape[0]), desc=f"Preparing {node_type} vectors"):
            if idx in inv_mapping:
                doc_id = inv_mapping[idx]
                vector_id = f"{node_type}:{doc_id}"
                
                # Convert embedding to list
                vector = embeddings[idx].tolist()
                
                # Add metadata
                metadata = {
                    "type": node_type,
                    "id": doc_id
                }
                
                vectors.append((vector_id, vector, metadata))
                
                # Upsert in batches
                if len(vectors) >= batch_size:
                    index.upsert(vectors=[(id, vec, meta) for id, vec, meta in vectors])
                    vectors = []
        
        # Upsert any remaining vectors
        if vectors:
            index.upsert(vectors=[(id, vec, meta) for id, vec, meta in vectors])
        
        print(f"Saved {len(inv_mapping)} {node_type} embeddings to Pinecone")


def export_to_local(embeddings_dict, mapping_dict, output_dir):
    """
    Export embeddings to local files.
    
    Args:
        embeddings_dict: Dictionary of embeddings for each node type
        mapping_dict: Dictionary of node mappings (index to Firestore ID)
        output_dir: Directory to save embeddings
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for node_type, embeddings in embeddings_dict.items():
        print(f"Exporting {node_type} embeddings to local files...")
        
        # Get mapping for this node type
        node_mapping = mapping_dict.get(node_type, {})
        if not node_mapping:
            print(f"Warning: No mapping found for {node_type}")
            continue
        
        # Create inverse mapping
        inv_mapping = {v: k for k, v in node_mapping.items()}
        
        # Save embeddings
        embeddings_path = os.path.join(output_dir, f"{node_type}_embeddings.pt")
        torch.save(embeddings, embeddings_path)
        
        # Save mapping
        mapping_path = os.path.join(output_dir, f"{node_type}_mapping.pt")
        torch.save(inv_mapping, mapping_path)
        
        print(f"Saved {node_type} embeddings to {embeddings_path}")
        print(f"Saved {node_type} mapping to {mapping_path}")


def main():
    """Main function."""
    args = parse_args()
    config = load_config(args.config)
    
    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load graph data
    print("Loading graph data...")
    graph, metadata = load_graph(config['data'])
    
    # Extract node mappings from graph loader
    # Assuming the graph loader has stored these mappings
    # In a production environment, you might want to access these directly
    node_mappings = {}
    loader = graph.loader if hasattr(graph, 'loader') else None
    
    if loader:
        node_mappings = {
            'user': loader.user_mapping,
            'place': loader.place_mapping,
            'event': loader.event_mapping if hasattr(loader, 'event_mapping') else {},
            'group': loader.group_mapping if hasattr(loader, 'group_mapping') else {}
        }
    else:
        print("Warning: Graph loader not available, node mappings may be incomplete")
        # Try to load mappings from cache or Firestore directly
        # This is a placeholder - implement based on your data structure
    
    # Create model
    in_channels_dict = {
        'user': graph['user'].x.size(1),
        'place': graph['place'].x.size(1),
        'event': graph['event'].x.size(1) if 'event' in graph.node_types else 0,
        'group': graph['group'].x.size(1) if 'group' in graph.node_types else 0
    }
    
    print("Creating model...")
    model = create_model(config['model'], metadata, in_channels_dict)
    
    # Load trained model
    print(f"Loading trained model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Generate embeddings
    print("Generating embeddings...")
    graph = graph.to(device)
    
    with torch.no_grad():
        out_dict = model(graph)
    
    # Move embeddings to CPU
    embeddings_dict = {
        node_type: embedding.cpu() for node_type, embedding in out_dict.items()
    }
    
    # Export embeddings
    if args.export_to == 'firestore':
        export_to_firestore(embeddings_dict, node_mappings)
    elif args.export_to == 'pinecone':
        api_key = os.environ.get('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        index_name = config.get('pinecone', {}).get('index_name', 'encite-embeddings')
        export_to_pinecone(embeddings_dict, node_mappings, index_name, api_key)
    elif args.export_to == 'local':
        output_dir = args.output_dir
        if not output_dir:
            output_dir = os.path.join('outputs', 'embeddings', time.strftime("%Y%m%d-%H%M%S"))
        
        export_to_local(embeddings_dict, node_mappings, output_dir)
    
    print("Embedding export complete!")


if __name__ == '__main__':
    main()