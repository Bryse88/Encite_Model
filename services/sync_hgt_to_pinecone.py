#!/usr/bin/env python3
"""
sync_hgt_to_pinecone.py

Syncs place, item, and event embeddings from Supabase to Pinecone using a trained HGT model.
Handles data fetching, graph construction, embedding generation, and vector upserts.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch_geometric.data import HeteroData
from supabase import create_client, Client
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sync_hgt_to_pinecone.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
SPECIALTY_KEYWORDS = {
    "happy hour": "Happy Hour",
    "live music": "Live Music",
    "special dinner": "Special Dinner",
    "tasting": "Tasting",
    "gallery talk": "Art Talk",
    "comedy": "Comedy Night",
    "tour": "Guided Tour"
}



class HGTEmbeddingSync:
    """Main class for syncing HGT embeddings to Pinecone"""
    
    def __init__(self):
        self.supabase: Client = None
        self.pinecone: Pinecone = None
        self.model = None
        self.embedding_dim = 64  # Based on your HGT architecture
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize connections
        self._init_supabase()
        self._init_pinecone()
        
    def _init_supabase(self):
        """Initialize Supabase client"""
        try:
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_ANON_KEY')
            
            if not supabase_url or not supabase_key:
                raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment")
                
            self.supabase = create_client(supabase_url, supabase_key)
            logger.info("✅ Supabase client initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase: {e}")
            raise
            
    def _init_pinecone(self):
        """Initialize Pinecone client and ensure index exists"""
        try:
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise ValueError("PINECONE_API_KEY must be set in environment")
                
            self.pinecone = Pinecone(api_key=api_key)
            
            # Ensure index exists
            index_name = os.getenv('PINECONE_INDEX_NAME', 'Encite')
            
            existing_indexes = [idx.name for idx in self.pinecone.list_indexes()]
            if index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {index_name}")
                self.pinecone.create_index(
                    name=index_name,
                    dimension=self.embedding_dim,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=os.getenv('PINECONE_REGION', 'us-east-1')
                    )
                )
                
            self.index = self.pinecone.Index(index_name)
            logger.info(f"✅ Pinecone client initialized with index: {index_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Pinecone: {e}")
            raise
    
    def load_hgt_model(self, checkpoint_path: str = 'checkpoints/hgt_model.pt'):
        """Load trained HGT model from checkpoint"""
        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
                
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Assuming the model class is saved with the checkpoint
            # You may need to adjust this based on your actual model structure
            self.model = checkpoint.get('model') or checkpoint
            self.model.eval()
            self.model.to(self.device)
            
            logger.info(f"✅ HGT model loaded from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load HGT model: {e}")
            raise
    
    def fetch_supabase_data(self) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """Fetch places, events, and items from Supabase"""
        try:
            # Fetch places
            places_response = self.supabase.table('places').select('*').execute()
            places = places_response.data
            logger.info(f"📥 Fetched {len(places)} places from Supabase")
            
            # Fetch events (filter out expired ones)
            now = datetime.now(timezone.utc).isoformat()
            events_response = self.supabase.table('events').select('*').gte('end_time', now).execute()
            events = events_response.data
            logger.info(f"📥 Fetched {len(events)} active events from Supabase")
            
            # Fetch items
            items_response = self.supabase.table('items').select('*').execute()
            items = items_response.data
            logger.info(f"📥 Fetched {len(items)} items from Supabase")
            
            return places, events, items
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch data from Supabase: {e}")
            raise
    
    def build_hetero_graph(self, users: List[Dict], places: List[Dict], 
                          events: List[Dict], items: List[Dict]) -> HeteroData:
        """Build HeteroData graph from Supabase data"""
        try:
            graph = HeteroData()
            
            # Create node mappings
            user_id_to_idx = {user['id']: idx for idx, user in enumerate(users)}
            place_id_to_idx = {place['id']: idx for idx, place in enumerate(places)}
            event_id_to_idx = {event['id']: idx for idx, event in enumerate(events)}
            item_id_to_idx = {item['id']: idx for idx, item in enumerate(items)}
            
            # Add node features (you may need to adjust based on your actual schema)
            graph['user'].x = torch.randn(len(users), 32)  # Placeholder features
            graph['place'].x = torch.randn(len(places), 32)
            graph['event'].x = torch.randn(len(events), 32)
            graph['item'].x = torch.randn(len(items), 32)
            
            # Store original data for metadata
            graph['user'].original_data = users
            graph['place'].original_data = places
            graph['event'].original_data = events
            graph['item'].original_data = items
            
            # Store ID mappings
            graph.user_id_to_idx = user_id_to_idx
            graph.place_id_to_idx = place_id_to_idx
            graph.event_id_to_idx = event_id_to_idx
            graph.item_id_to_idx = item_id_to_idx
            
            # Add edges (you'll need to fetch these from your interaction tables)
            self._add_edges_to_graph(graph, user_id_to_idx, place_id_to_idx, 
                                   event_id_to_idx, item_id_to_idx)
            
            logger.info(f"✅ Built heterogeneous graph with {len(users)} users, "
                       f"{len(places)} places, {len(events)} events, {len(items)} items")
            
            return graph
            
        except Exception as e:
            logger.error(f"❌ Failed to build hetero graph: {e}")
            raise
    
    def _add_edges_to_graph(self, graph: HeteroData, user_id_to_idx: Dict, 
                           place_id_to_idx: Dict, event_id_to_idx: Dict, 
                           item_id_to_idx: Dict):
        """Add edges to the graph from interaction tables"""
        try:
            # Fetch user-place interactions
            user_place_response = self.supabase.table('user_place_interactions').select('*').execute()
            user_place_edges = []
            for interaction in user_place_response.data:
                if (interaction['user_id'] in user_id_to_idx and 
                    interaction['place_id'] in place_id_to_idx):
                    user_place_edges.append([
                        user_id_to_idx[interaction['user_id']],
                        place_id_to_idx[interaction['place_id']]
                    ])
            
            if user_place_edges:
                graph['user', 'likes', 'place'].edge_index = torch.tensor(user_place_edges).t()
            
            # Add similar logic for other edge types
            # user-event, user-item, user-group, etc.
            # You'll need to adapt this based on your actual database schema
            
            logger.info("✅ Added edges to heterogeneous graph")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not add all edges to graph: {e}")
            # Continue with empty edges if interaction data is not available
            graph['user', 'likes', 'place'].edge_index = torch.empty((2, 0), dtype=torch.long)
            graph['user', 'likes', 'event'].edge_index = torch.empty((2, 0), dtype=torch.long)
            graph['user', 'likes', 'item'].edge_index = torch.empty((2, 0), dtype=torch.long)
    
    def generate_embeddings(self, graph: HeteroData) -> Dict[str, np.ndarray]:
        """Generate embeddings using the HGT model"""
        try:
            graph = graph.to(self.device)
            
            with torch.no_grad():
                # Get embeddings from HGT model
                embeddings = self.model.encode(graph)
                
                # Convert to numpy and move to CPU
                result = {}
                for node_type in ['place', 'event', 'item']:
                    if node_type in embeddings:
                        result[node_type] = embeddings[node_type].cpu().numpy()
                        logger.info(f"✅ Generated {len(result[node_type])} {node_type} embeddings")
                
                return result
                
        except Exception as e:
            logger.error(f"❌ Failed to generate embeddings: {e}")
            raise
    
    def format_vectors_for_pinecone(self, embeddings: Dict[str, np.ndarray], 
                                   graph: HeteroData) -> Dict[str, List[Dict]]:
        """Format embeddings with metadata for Pinecone upsert"""
        
        def extract_specialties(text: str) -> List[str]:
            specialties = []
            lower_text = text.lower()
            for keyword, label in SPECIALTY_KEYWORDS.items():
                if keyword in lower_text:
                    specialties.append(label)
            return specialties

        try:
            formatted_vectors = {}
            
            for node_type in ['place', 'event', 'item']:
                if node_type not in embeddings:
                    continue
                    
                vectors = []
                original_data = graph[node_type].original_data
                node_embeddings = embeddings[node_type]
                
                for idx, (data, embedding) in enumerate(zip(original_data, node_embeddings)):
                    vector_id = f"{node_type}_{data['id']}"
                    
                    # Prepare metadata based on node type
                    metadata = {
                        'id': str(data['id']),
                        'type': node_type,
                        'name': data.get('name', ''),
                    }
                    
                    # Add type-specific metadata
                    if node_type == 'place':
                        metadata.update({
                            'category': data.get('category', ''),
                            'lat': data.get('latitude', 0.0),
                            'lng': data.get('longitude', 0.0),
                            'address': data.get('address', ''),
                            'rating': data.get('rating', 0.0)
                        })
                    elif node_type == 'event':
                        metadata.update({
                            'category': data.get('category', ''),
                            'start_time': data.get('start_time', ''),
                            'end_time': data.get('end_time', ''),
                            'location': data.get('location', ''),
                            'price': data.get('price', 0.0),
                            'specialties': extract_specialties(data.get('name', '') + " " + data.get('description', ''))
                        })
                    elif node_type == 'item':
                        metadata.update({
                            'category': data.get('category', ''),
                            'price': data.get('price', 0.0),
                            'brand': data.get('brand', ''),
                            'description': data.get('description', '')
                        })
                    
                    vectors.append({
                        'id': vector_id,
                        'values': embedding.tolist(),
                        'metadata': metadata
                    })
                
                formatted_vectors[node_type] = vectors
                logger.info(f"✅ Formatted {len(vectors)} {node_type} vectors for Pinecone")
            
            return formatted_vectors
            
        except Exception as e:
            logger.error(f"❌ Failed to format vectors: {e}")
            raise
    
    def upsert_to_pinecone(self, formatted_vectors: Dict[str, List[Dict]], 
                          batch_size: int = 100):
        """Upsert vectors to Pinecone with appropriate namespaces"""
        try:
            for node_type, vectors in formatted_vectors.items():
                if not vectors:
                    continue
                    
                namespace = f"{node_type}s"  # places, events, items
                
                # Batch upsert
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i + batch_size]
                    
                    self.index.upsert(
                        vectors=batch,
                        namespace=namespace
                    )
                    
                    logger.info(f"📤 Upserted batch {i//batch_size + 1} "
                               f"({len(batch)} vectors) to namespace '{namespace}'")
                
                logger.info(f"✅ Successfully upserted {len(vectors)} {node_type} vectors")
                
        except Exception as e:
            logger.error(f"❌ Failed to upsert vectors to Pinecone: {e}")
            raise
    
    def delete_expired_events(self):
        """Delete expired events from Pinecone"""
        try:
            # Fetch expired events from Supabase
            now = datetime.now(timezone.utc).isoformat()
            expired_response = self.supabase.table('events').select('id').lt('end_time', now).execute()
            expired_events = expired_response.data
            
            if not expired_events:
                logger.info("No expired events to delete")
                return
            
            # Delete from Pinecone
            expired_ids = [f"event_{event['id']}" for event in expired_events]
            
            # Delete in batches
            batch_size = 1000
            for i in range(0, len(expired_ids), batch_size):
                batch_ids = expired_ids[i:i + batch_size]
                self.index.delete(ids=batch_ids, namespace='events')
                
            logger.info(f"🗑️ Deleted {len(expired_ids)} expired events from Pinecone")
            
        except Exception as e:
            logger.error(f"❌ Failed to delete expired events: {e}")
            # Don't raise - this is cleanup, shouldn't fail the main sync
    
    def sync_embeddings(self, checkpoint_path: str = 'checkpoints/hgt_model.pt'):
        """Main sync function"""
        try:
            logger.info("🚀 Starting HGT to Pinecone sync process")
            
            # Load model
            self.load_hgt_model(checkpoint_path)
            
            # Fetch data
            places, events, items = self.fetch_supabase_data()
            
            # Build graph
            graph = self.build_hetero_graph(places, events, items)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(graph)
            
            # Format for Pinecone
            formatted_vectors = self.format_vectors_for_pinecone(embeddings, graph)
            
            # Upsert to Pinecone
            self.upsert_to_pinecone(formatted_vectors)
            
            # Cleanup expired events
            self.delete_expired_events()
            
            logger.info("✅ HGT to Pinecone sync completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Sync process failed: {e}")
            raise


def main():
    """Main entry point"""
    try:
        syncer = HGTEmbeddingSync()
        checkpoint_path = os.getenv('HGT_CHECKPOINT_PATH', 'checkpoints/hgt_best_model.pt')
        syncer.sync_embeddings(checkpoint_path)
        
    except Exception as e:
        logger.error(f"❌ Application failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()