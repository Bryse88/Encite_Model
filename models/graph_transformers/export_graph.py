"""
Export graph data from Firestore to a local cache file.
This is useful for offline development and testing.
"""

import os
import argparse
import torch
from google.cloud import firestore
from dataset_loader import FirestoreGraphLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export graph data from Firestore')
    parser.add_argument('--project_id', type=str, default=None,
                        help='Google Cloud project ID')
    parser.add_argument('--output_path', type=str, default='data/graph_cache.pt',
                        help='Output path for graph cache')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of documents to load per collection')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Initialize Firestore client
    if args.project_id:
        db = firestore.Client(project=args.project_id)
    else:
        db = firestore.Client()
    
    print(f"Initializing graph loader...")
    loader = FirestoreGraphLoader(db=db, cache_path=args.output_path)
    
    print(f"Loading graph data from Firestore (limit={args.limit})...")
    graph = loader.load_all_data(limit=args.limit)
    
    print(f"Graph loaded and saved to {args.output_path}")
    print(f"Graph statistics:")
    print(f"  Number of users: {graph['user'].num_nodes}")
    print(f"  Number of places: {graph['place'].num_nodes}")
    
    if 'event' in graph.node_types:
        print(f"  Number of events: {graph['event'].num_nodes}")
    
    if 'group' in graph.node_types:
        print(f"  Number of groups: {graph['group'].num_nodes}")
    
    print(f"Edge types: {graph.edge_types}")
    for edge_type in graph.edge_types:
        print(f"  {edge_type}: {graph[edge_type].edge_index.size(1)} edges")


if __name__ == '__main__':
    main()