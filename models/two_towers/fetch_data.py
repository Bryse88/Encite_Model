"""
Unified Fetcher Module for Encite
---------------------------------
This module provides a comprehensive solution for fetching location-based data
from multiple sources with a priority on Supabase as the source of truth and
Foursquare as a fallback for real-time data.

The module normalizes data from different sources into a consistent format
that aligns with the feature structure used in the Heterogeneous Graph Transformer
(HGT) training pipeline.

Dependencies:
- supabase-py
- httpx
- pandas
- dotenv (for configuration)
"""

import os
import json
import logging
from typing import Dict, List, Union, Optional, Tuple
from datetime import datetime, timedelta
import time
import math

import pandas as pd
import httpx
from supabase import create_client, Client
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("encite_fetcher")

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Foursquare configuration
FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY")
FOURSQUARE_BASE_URL = "https://api.foursquare.com/v3"

# Default settings
DEFAULT_RADIUS = 1000  # meters
DEFAULT_LIMIT = 50     # maximum number of results

class FetcherError(Exception):
    """Custom exception class for fetcher-related errors."""
    pass


class EntityFetcher:
    """
    Unified fetcher class that handles fetching entities (places, events, items)
    from Supabase and external APIs like Foursquare.
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None, 
                 foursquare_api_key: str = None):
        """
        Initialize the fetcher with API credentials.
        
        Args:
            supabase_url (str, optional): Supabase URL. Defaults to env variable.
            supabase_key (str, optional): Supabase API key. Defaults to env variable.
            foursquare_api_key (str, optional): Foursquare API key. Defaults to env variable.
        
        Raises:
            FetcherError: If required credentials are missing.
        """
        # Initialize Supabase client
        self.supabase_url = supabase_url or SUPABASE_URL
        self.supabase_key = supabase_key or SUPABASE_KEY
        
        if not self.supabase_url or not self.supabase_key:
            logger.warning("Supabase credentials not provided. Supabase features disabled.")
            self.supabase = None
        else:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
                logger.info("Supabase client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {str(e)}")
                self.supabase = None
        
        # Initialize Foursquare API key
        self.foursquare_api_key = foursquare_api_key or FOURSQUARE_API_KEY
        if not self.foursquare_api_key:
            logger.warning("Foursquare API key not provided. Foursquare features disabled.")
        
        # Initialize Foursquare API client
        self.http_client = httpx.Client(
            headers={
                "Accept": "application/json",
                "Authorization": f"fsq_auth {self.foursquare_api_key}" if self.foursquare_api_key else ""
            },
            timeout=30.0
        )
        
        # Cache for API responses to reduce duplicate calls
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        if hasattr(self, 'http_client') and self.http_client:
            self.http_client.close()
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the Haversine distance between two points in kilometers.
        
        Args:
            lat1 (float): Latitude of point 1
            lon1 (float): Longitude of point 1
            lat2 (float): Latitude of point 2
            lon2 (float): Longitude of point 2
            
        Returns:
            float: Distance in kilometers
        """
        # Radius of the Earth in kilometers
        R = 6371.0
        
        # Convert degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine formula
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        
        return distance * 1000  # Convert to meters
    
    def _get_cache_key(self, prefix: str, **kwargs) -> str:
        """
        Generate a cache key from the function prefix and kwargs.
        
        Args:
            prefix (str): Function prefix
            **kwargs: Keyword arguments to the function
            
        Returns:
            str: Cache key
        """
        # Sort kwargs for consistent keys
        sorted_kwargs = sorted(kwargs.items())
        
        # Convert to a string
        kwargs_str = json.dumps(sorted_kwargs)
        
        return f"{prefix}:{kwargs_str}"
    
    def _get_from_cache(self, key: str) -> Optional[Tuple[datetime, any]]:
        """
        Get a value from the cache if it exists and is not expired.
        
        Args:
            key (str): Cache key
            
        Returns:
            Optional[Tuple[datetime, any]]: Cached value and timestamp if found, None otherwise
        """
        if key in self.cache:
            timestamp, value = self.cache[key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                return value
        return None
    
    def _set_in_cache(self, key: str, value: any) -> None:
        """
        Set a value in the cache.
        
        Args:
            key (str): Cache key
            value (any): Value to cache
        """
        self.cache[key] = (datetime.now(), value)
    
    def _clean_cache(self) -> None:
        """Remove expired entries from the cache."""
        now = datetime.now()
        expired_keys = []
        
        for key, (timestamp, _) in self.cache.items():
            if (now - timestamp).total_seconds() >= self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
    
    def _fetch_from_supabase(self, table: str, lat: float, lon: float, 
                            radius: float = DEFAULT_RADIUS, 
                            limit: int = DEFAULT_LIMIT,
                            category: Optional[str] = None) -> List[Dict]:
        """
        Fetch entities from Supabase within a specified radius of the given coordinates.
        
        Args:
            table (str): Supabase table name to query (places, events, items)
            lat (float): Latitude coordinate
            lon (float): Longitude coordinate
            radius (float, optional): Search radius in meters. Defaults to DEFAULT_RADIUS.
            limit (int, optional): Maximum number of results. Defaults to DEFAULT_LIMIT.
            category (str, optional): Category filter. Defaults to None.
            
        Returns:
            List[Dict]: List of entities matching the criteria
            
        Raises:
            FetcherError: If Supabase client is not available or query fails
        """
        if not self.supabase:
            logger.warning("Supabase client not available")
            return []
        
        try:
            # Cache key for this query
            cache_key = self._get_cache_key("supabase", table=table, lat=lat, lon=lon, 
                                           radius=radius, limit=limit, category=category)
            
            # Check cache first
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for {cache_key}")
                return cached_result
            
            # Using PostGIS ST_DWithin to find entities within radius
            # Note: This assumes PostGIS is enabled on your Supabase instance
            # and the table has a geography column named 'location'
            
            # Build the query
            query = self.supabase.table(table) \
                .select('*') \
                .filter('location', 'not.is', 'null')
            
            # Add category filter if provided
            if category:
                query = query.filter('category', 'eq', category)
            
            # Get results
            response = query.execute()
            
            if hasattr(response, 'data'):
                # Filter results by distance manually if PostGIS is not available
                results = []
                for item in response.data:
                    if 'latitude' in item and 'longitude' in item:
                        distance = self._calculate_distance(
                            lat, lon, 
                            item['latitude'], item['longitude']
                        )
                        if distance <= radius:
                            item['distance_meters'] = distance
                            results.append(item)
                    
                # Sort by distance and apply limit
                results = sorted(results, key=lambda x: x.get('distance_meters', float('inf')))[:limit]
                
                # Cache results
                self._set_in_cache(cache_key, results)
                
                return results
            return []
        
        except Exception as e:
            logger.error(f"Error fetching from Supabase {table}: {str(e)}")
            raise FetcherError(f"Supabase query failed: {str(e)}")
    
    async def _fetch_from_foursquare(self, lat: float, lon: float, 
                                   radius: float = DEFAULT_RADIUS, 
                                   limit: int = DEFAULT_LIMIT,
                                   category: Optional[str] = None) -> List[Dict]:
        """
        Fetch places from Foursquare API within a specified radius.
        
        Args:
            lat (float): Latitude coordinate
            lon (float): Longitude coordinate
            radius (float, optional): Search radius in meters. Defaults to DEFAULT_RADIUS.
            limit (int, optional): Maximum number of results. Defaults to DEFAULT_LIMIT.
            category (str, optional): Category filter. Defaults to None.
            
        Returns:
            List[Dict]: List of places from Foursquare
            
        Raises:
            FetcherError: If API request fails
        """
        if not self.foursquare_api_key:
            logger.warning("Foursquare API key not available")
            return []
        
        try:
            # Cache key for this query
            cache_key = self._get_cache_key("foursquare", lat=lat, lon=lon, 
                                           radius=radius, limit=limit, category=category)
            
            # Check cache first
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for {cache_key}")
                return cached_result
            
            # Build parameters
            params = {
                'll': f"{lat},{lon}",
                'radius': radius,
                'limit': limit,
                'sort': 'RELEVANCE'
            }
            
            # Add category if provided
            if category:
                # Map our category to Foursquare category IDs
                # This is a simplified example - you'll need a proper mapping
                category_mappings = {
                    'food': '13000',  # Food category in Foursquare
                    'drinks': '13003',  # Nightlife category in Foursquare
                    'arts': '10000',   # Arts & Entertainment
                    'outdoors': '16000',  # Outdoors & Recreation
                    # Add more mappings as needed
                }
                
                if category in category_mappings:
                    params['categories'] = category_mappings[category]
                else:
                    # Use the query parameter for text search if not a known category
                    params['query'] = category
            
            # Make the API request
            url = f"{FOURSQUARE_BASE_URL}/places/search"
            
            response = self.http_client.get(url, params=params)
            response.raise_for_status()
            
            places = response.json().get('results', [])
            
            # Cache results
            self._set_in_cache(cache_key, places)
            
            return places
            
        except httpx.HTTPError as e:
            logger.error(f"Foursquare API error: {str(e)}")
            raise FetcherError(f"Foursquare API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Foursquare response: {str(e)}")
            raise FetcherError(f"Failed to parse Foursquare response: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in Foursquare fetch: {str(e)}")
            raise FetcherError(f"Unexpected error in Foursquare fetch: {str(e)}")
    
    def _normalize_place_data(self, place: Dict, source: str) -> Dict:
        """
        Normalize place data from different sources to a standard format.
        
        Args:
            place (Dict): Raw place data
            source (str): Source of the data ('supabase' or 'foursquare')
            
        Returns:
            Dict: Normalized place data
        """
        try:
            if source == 'supabase':
                # Supabase data is already in our expected format
                normalized = {
                    'id': place.get('id', ''),
                    'name': place.get('name', ''),
                    'latitude': place.get('latitude', 0.0),
                    'longitude': place.get('longitude', 0.0),
                    'category': place.get('category', ''),
                    'rating': place.get('rating', 0.0),
                    'price_level': place.get('price_level', 0),
                    'popularity_score': place.get('popularity_score', 0.0),
                    'distance_meters': place.get('distance_meters', 0),
                    'address': place.get('address', ''),
                    'image_url': place.get('image_url', ''),
                    'website': place.get('website', ''),
                    'phone': place.get('phone', ''),
                    'hours': place.get('hours', {}),
                    'features': place.get('features', []),
                    'source': 'supabase',
                    'source_id': place.get('id', ''),
                    'raw_data': place
                }
                return normalized
            
            elif source == 'foursquare':
                # Convert Foursquare format to our standard format
                location = place.get('location', {})
                categories = place.get('categories', [])
                
                # Extract category from Foursquare categories
                category = ''
                if categories and len(categories) > 0:
                    category = categories[0].get('name', '')
                
                # Extract price level
                price_level = 0
                if 'price' in place:
                    price_level = len(place['price'])  # Foursquare uses $ symbols
                
                # Extract rating - Foursquare uses 0-10 scale, normalize to 0-5
                rating = 0.0
                if 'rating' in place:
                    rating = place['rating'] / 2.0
                
                # Create normalized place object
                normalized = {
                    'id': f"fsq_{place.get('fsq_id', '')}",
                    'name': place.get('name', ''),
                    'latitude': location.get('latitude', 0.0),
                    'longitude': location.get('longitude', 0.0),
                    'category': category,
                    'rating': rating,
                    'price_level': price_level,
                    'popularity_score': place.get('popularity', 0.0),
                    'distance_meters': place.get('distance', 0),
                    'address': ', '.join(location.get('formatted_address', [])),
                    'image_url': self._extract_foursquare_photo(place),
                    'website': place.get('website', ''),
                    'phone': place.get('tel', ''),
                    'hours': self._extract_foursquare_hours(place),
                    'features': self._extract_foursquare_features(place),
                    'source': 'foursquare',
                    'source_id': place.get('fsq_id', ''),
                    'raw_data': place
                }
                return normalized
            
            else:
                logger.warning(f"Unknown source: {source}")
                return {
                    'source': source,
                    'raw_data': place
                }
                
        except Exception as e:
            logger.error(f"Error normalizing place data: {str(e)}")
            # Return a minimal normalized object with the raw data
            return {
                'name': place.get('name', 'Unknown Place'),
                'source': source,
                'error': str(e),
                'raw_data': place
            }
    
    def _extract_foursquare_photo(self, place: Dict) -> str:
        """
        Extract photo URL from Foursquare place data.
        
        Args:
            place (Dict): Foursquare place data
            
        Returns:
            str: Photo URL or empty string
        """
        try:
            photos = place.get('photos', [])
            if photos and len(photos) > 0:
                photo = photos[0]
                prefix = photo.get('prefix', '')
                suffix = photo.get('suffix', '')
                if prefix and suffix:
                    return f"{prefix}original{suffix}"
            return ''
        except Exception as e:
            logger.error(f"Error extracting Foursquare photo: {str(e)}")
            return ''
    
    def _extract_foursquare_hours(self, place: Dict) -> Dict:
        """
        Extract hours from Foursquare place data.
        
        Args:
            place (Dict): Foursquare place data
            
        Returns:
            Dict: Structured hours data
        """
        try:
            hours = {}
            if 'hours' in place:
                fs_hours = place['hours']
                if 'display' in fs_hours:
                    hours['display'] = fs_hours['display']
                if 'regular' in fs_hours:
                    hours['regular'] = fs_hours['regular']
            return hours
        except Exception as e:
            logger.error(f"Error extracting Foursquare hours: {str(e)}")
            return {}
    
    def _extract_foursquare_features(self, place: Dict) -> List[str]:
        """
        Extract features from Foursquare place data.
        
        Args:
            place (Dict): Foursquare place data
            
        Returns:
            List[str]: List of features
        """
        features = []
        try:
            # Extract attributes
            if 'attributes' in place:
                for attr in place['attributes']:
                    features.append(attr)
            
            # Extract categories beyond the primary one
            if 'categories' in place and len(place['categories']) > 1:
                for category in place['categories'][1:]:
                    features.append(category.get('name', ''))
            
            return [f for f in features if f]  # Filter out empty strings
        except Exception as e:
            logger.error(f"Error extracting Foursquare features: {str(e)}")
            return []
    
    def _normalize_event_data(self, event: Dict, source: str) -> Dict:
        """
        Normalize event data from different sources to a standard format.
        
        Args:
            event (Dict): Raw event data
            source (str): Source of the data
            
        Returns:
            Dict: Normalized event data
        """
        try:
            if source == 'supabase':
                # Supabase data is already in our expected format
                normalized = {
                    'id': event.get('id', ''),
                    'name': event.get('name', ''),
                    'latitude': event.get('latitude', 0.0),
                    'longitude': event.get('longitude', 0.0),
                    'category': event.get('category', ''),
                    'start_time': event.get('start_time', ''),
                    'end_time': event.get('end_time', ''),
                    'price': event.get('price', 0.0),
                    'popularity_score': event.get('popularity_score', 0.0),
                    'distance_meters': event.get('distance_meters', 0),
                    'address': event.get('address', ''),
                    'image_url': event.get('image_url', ''),
                    'website': event.get('website', ''),
                    'description': event.get('description', ''),
                    'tags': event.get('tags', []),
                    'source': 'supabase',
                    'source_id': event.get('id', ''),
                    'raw_data': event
                }
                return normalized
            
            # TODO: Add more sources (Eventbrite, etc.)
            
            else:
                logger.warning(f"Unknown source for event: {source}")
                return {
                    'source': source,
                    'raw_data': event
                }
                
        except Exception as e:
            logger.error(f"Error normalizing event data: {str(e)}")
            # Return a minimal normalized object with the raw data
            return {
                'name': event.get('name', 'Unknown Event'),
                'source': source,
                'error': str(e),
                'raw_data': event
            }
    
    def _normalize_item_data(self, item: Dict, source: str) -> Dict:
        """
        Normalize item data from different sources to a standard format.
        
        Args:
            item (Dict): Raw item data
            source (str): Source of the data
            
        Returns:
            Dict: Normalized item data
        """
        try:
            if source == 'supabase':
                # Supabase data is already in our expected format
                normalized = {
                    'id': item.get('id', ''),
                    'name': item.get('name', ''),
                    'category': item.get('category', ''),
                    'rating': item.get('rating', 0.0),
                    'price': item.get('price', 0.0),
                    'popularity_score': item.get('popularity_score', 0.0),
                    'image_url': item.get('image_url', ''),
                    'description': item.get('description', ''),
                    'tags': item.get('tags', []),
                    'source': 'supabase',
                    'source_id': item.get('id', ''),
                    'raw_data': item
                }
                return normalized
            
            else:
                logger.warning(f"Unknown source for item: {source}")
                return {
                    'source': source,
                    'raw_data': item
                }
                
        except Exception as e:
            logger.error(f"Error normalizing item data: {str(e)}")
            # Return a minimal normalized object with the raw data
            return {
                'name': item.get('name', 'Unknown Item'),
                'source': source,
                'error': str(e),
                'raw_data': item
            }
    
    def fetch_places(self, lat: float, lon: float, radius: float = DEFAULT_RADIUS, 
                    limit: int = DEFAULT_LIMIT, category: Optional[str] = None, 
                    use_fallback: bool = True) -> Union[List[Dict], pd.DataFrame]:
        """
        Fetch places near the specified coordinates.
        
        Args:
            lat (float): Latitude coordinate
            lon (float): Longitude coordinate
            radius (float, optional): Search radius in meters. Defaults to DEFAULT_RADIUS.
            limit (int, optional): Maximum number of results. Defaults to DEFAULT_LIMIT.
            category (str, optional): Category filter. Defaults to None.
            use_fallback (bool, optional): Whether to use Foursquare as fallback. Defaults to True.
            
        Returns:
            Union[List[Dict], pd.DataFrame]: Normalized place data
            
        Raises:
            FetcherError: If all data sources fail
        """
        results = []
        
        # Clean expired cache entries
        self._clean_cache()
        
        # Try Supabase first
        try:
            supabase_places = self._fetch_from_supabase(
                table='places',
                lat=lat,
                lon=lon,
                radius=radius,
                limit=limit,
                category=category
            )
            
            # Normalize Supabase results
            for place in supabase_places:
                normalized = self._normalize_place_data(place, 'supabase')
                results.append(normalized)
                
            logger.info(f"Fetched {len(results)} places from Supabase")
        except Exception as e:
            logger.error(f"Error fetching places from Supabase: {str(e)}")
            # Continue with fallback
        
        # If no results from Supabase, try Foursquare
        if len(results) == 0 and use_fallback and self.foursquare_api_key:
            try:
                import asyncio
                
                # Run the async function in a synchronous context
                foursquare_places = asyncio.run(self._fetch_from_foursquare(
                    lat=lat,
                    lon=lon,
                    radius=radius,
                    limit=limit,
                    category=category
                ))
                
                # Normalize Foursquare results
                for place in foursquare_places:
                    normalized = self._normalize_place_data(place, 'foursquare')
                    results.append(normalized)
                
                logger.info(f"Fetched {len(results)} places from Foursquare fallback")
            except Exception as e:
                logger.error(f"Error fetching places from Foursquare: {str(e)}")
        
        # Convert to DataFrame if needed
        if isinstance(results, list) and len(results) > 0:
            # Create a pandas DataFrame
            df = pd.DataFrame(results)
            # Clean up column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            return df
        
        return results
    
    def fetch_events(self, lat: float, lon: float, radius: float = DEFAULT_RADIUS, 
                    limit: int = DEFAULT_LIMIT, category: Optional[str] = None) -> Union[List[Dict], pd.DataFrame]:
        """
        Fetch events near the specified coordinates.
        
        Args:
            lat (float): Latitude coordinate
            lon (float): Longitude coordinate
            radius (float, optional): Search radius in meters. Defaults to DEFAULT_RADIUS.
            limit (int, optional): Maximum number of results. Defaults to DEFAULT_LIMIT.
            category (str, optional): Category filter. Defaults to None.
            
        Returns:
            Union[List[Dict], pd.DataFrame]: Normalized event data
            
        Raises:
            FetcherError: If all data sources fail
        """
        results = []
        
        # Clean expired cache entries
        self._clean_cache()
        
        # Try Supabase first
        try:
            supabase_events = self._fetch_from_supabase(
                table='events',
                lat=lat,
                lon=lon,
                radius=radius,
                limit=limit,
                category=category
            )
            
            # Normalize Supabase results
            for event in supabase_events:
                normalized = self._normalize_event_data(event, 'supabase')
                results.append(normalized)
                
            logger.info(f"Fetched {len(results)} events from Supabase")
        except Exception as e:
            logger.error(f"Error fetching events from Supabase: {str(e)}")
            # Continue with fallback
        
        # TODO: Add Eventbrite or other event API fallback if needed
        
        # Convert to DataFrame if needed
        if isinstance(results, list) and len(results) > 0:
            # Create a pandas DataFrame
            df = pd.DataFrame(results)
            # Clean up column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            return df
        
        return results
    
    def fetch_items(self, lat: float, lon: float, radius: float = DEFAULT_RADIUS, 
                   limit: int = DEFAULT_LIMIT, category: Optional[str] = None) -> Union[List[Dict], pd.DataFrame]:
        """
        Fetch items near the specified coordinates.
        
        Args:
            lat (float): Latitude coordinate
            lon (float): Longitude coordinate
            radius (float, optional): Search radius in meters. Defaults to DEFAULT_RADIUS.
            limit (int, optional): Maximum number of results. Defaults to DEFAULT_LIMIT.
            category (str, optional): Category filter. Defaults to None.
            
        Returns:
            Union[List[Dict], pd.DataFrame]: Normalized item data
            
        Raises:
            FetcherError: If all data sources fail
        """
        results = []
        
        # Clean expired cache entries
        self._clean_cache()
        
        # Try Supabase first
        try:
            supabase_items = self._fetch_from_supabase(
                table='items',
                lat=lat,
                lon=lon,
                radius=radius,
                limit=limit,
                category=category
            )
            
            # Normalize Supabase results
            for item in supabase_items:
                normalized = self._normalize_item_data(item, 'supabase')
                results.append(normalized)
                
            logger.info(f"Fetched {len(results)} items from Supabase")
        except Exception as e:
            logger.error(f"Error fetching items from Supabase: {str(e)}")
            # Continue with fallback
        
        # No fallback for items currently implemented
        
        # Convert to DataFrame if needed
        if isinstance(results, list) and len(results) > 0:
            # Create a pandas DataFrame
            df = pd.DataFrame(results)
            # Clean up column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            return df
        
        return results
    
    def fetch_all(self, lat: float, lon: float, radius: float = DEFAULT_RADIUS, 
                 limit: int = DEFAULT_LIMIT, category: Optional[str] = None,
                 use_fallback: bool = True) -> Dict[str, Union[List[Dict], pd.DataFrame]]:
        """
        Fetch all entity types (places, events, items) near the specified coordinates.
        
        Args:
            lat (float): Latitude coordinate
            lon (float): Longitude coordinate
            radius (float, optional): Search radius in meters. Defaults to DEFAULT_RADIUS.
            limit (int, optional): Maximum number of results per entity type. Defaults to DEFAULT_LIMIT.
            category (str, optional): Category filter. Defaults to None.
            use_fallback (bool, optional): Whether to use fallback services. Defaults to True.
            
        Returns:
            Dict[str, Union[List[Dict], pd.DataFrame]]: Dictionary with keys 'places', 'events', 'items'
                                                      and their respective data
        """
        results = {
            'places': [],
            'events': [],
            'items': []
        }
        
        try:
            # Fetch places
            places = self.fetch_places(
                lat=lat,
                lon=lon,
                radius=radius,
                limit=limit,
                category=category,
                use_fallback=use_fallback
            )
            results['places'] = places
            
            # Fetch events
            events = self.fetch_events(
                lat=lat,
                lon=lon,
                radius=radius,
                limit=limit,
                category=category
            )
            results['events'] = events
            
            # Fetch items
            items = self.fetch_items(
                lat=lat,
                lon=lon,
                radius=radius,
                limit=limit,
                category=category
            )
            results['items'] = items
            
            logger.info(f"Fetched all entity types successfully")
            
        except Exception as e:
            logger.error(f"Error in fetch_all: {str(e)}")
            # Return whatever data we have
        
        return results
    
    def fetch_for_hgt(self, lat: float, lon: float, radius: float = DEFAULT_RADIUS,
                    limit: int = DEFAULT_LIMIT, category: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch and prepare data specifically formatted for HGT model input.
        
        Args:
            lat (float): Latitude coordinate
            lon (float): Longitude coordinate
            radius (float, optional): Search radius in meters. Defaults to DEFAULT_RADIUS.
            limit (int, optional): Maximum number of results per entity type. Defaults to DEFAULT_LIMIT.
            category (str, optional): Category filter. Defaults to None.
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with standardized DataFrames ready for HGT input
        """
        # Fetch all entity types
        raw_data = self.fetch_all(lat, lon, radius, limit, category)
        
        # Initialize result dictionary
        hgt_data = {}
        
        # Process places for HGT
        if isinstance(raw_data['places'], pd.DataFrame) and not raw_data['places'].empty:
            places_df = raw_data['places'].copy()
            
            # Select and rename columns for HGT model
            selected_cols = [
                'id', 'name', 'category', 'rating', 'price_level', 
                'popularity_score', 'latitude', 'longitude', 'distance_meters'
            ]
            
            # Ensure all columns exist, fill with defaults if not
            for col in selected_cols:
                if col not in places_df.columns:
                    if col in ['rating', 'popularity_score', 'latitude', 'longitude', 'distance_meters']:
                        places_df[col] = 0.0
                    elif col == 'price_level':
                        places_df[col] = 0
                    else:
                        places_df[col] = ''
            
            # Select only necessary columns and drop any rows with NaN in critical columns
            places_hgt = places_df[selected_cols]
            places_hgt = places_hgt.dropna(subset=['id', 'name'])
            
            # Add entity type column
            places_hgt['entity_type'] = 'place'
            
            hgt_data['places'] = places_hgt
        else:
            hgt_data['places'] = pd.DataFrame()
        
        # Process events for HGT
        if isinstance(raw_data['events'], pd.DataFrame) and not raw_data['events'].empty:
            events_df = raw_data['events'].copy()
            
            # Select and rename columns for HGT model
            selected_cols = [
                'id', 'name', 'category', 'price', 'popularity_score',
                'latitude', 'longitude', 'distance_meters'
            ]
            
            # Ensure all columns exist, fill with defaults if not
            for col in selected_cols:
                if col not in events_df.columns:
                    if col in ['price', 'popularity_score', 'latitude', 'longitude', 'distance_meters']:
                        events_df[col] = 0.0
                    else:
                        events_df[col] = ''
            
            # Convert price to price_level for consistency with places
            if 'price' in events_df.columns and 'price_level' not in events_df.columns:
                # Map price ranges to levels (adjust as needed)
                events_df['price_level'] = events_df['price'].apply(
                    lambda x: 0 if x == 0.0 else
                              1 if x < 20.0 else
                              2 if x < 50.0 else
                              3 if x < 100.0 else 4
                )
            
            # Select only necessary columns and drop any rows with NaN in critical columns
            events_hgt = events_df[selected_cols + ['price_level'] if 'price_level' in events_df.columns else selected_cols]
            events_hgt = events_hgt.dropna(subset=['id', 'name'])
            
            # Add entity type column
            events_hgt['entity_type'] = 'event'
            
            hgt_data['events'] = events_hgt
        else:
            hgt_data['events'] = pd.DataFrame()
        
        # Process items for HGT
        if isinstance(raw_data['items'], pd.DataFrame) and not raw_data['items'].empty:
            items_df = raw_data['items'].copy()
            
            # Select and rename columns for HGT model
            selected_cols = [
                'id', 'name', 'category', 'rating', 'price', 
                'popularity_score'
            ]
            
            # Ensure all columns exist, fill with defaults if not
            for col in selected_cols:
                if col not in items_df.columns:
                    if col in ['rating', 'price', 'popularity_score']:
                        items_df[col] = 0.0
                    else:
                        items_df[col] = ''
            
            # Convert price to price_level for consistency
            if 'price' in items_df.columns and 'price_level' not in items_df.columns:
                # Map price ranges to levels (adjust as needed)
                items_df['price_level'] = items_df['price'].apply(
                    lambda x: 0 if x == 0.0 else
                              1 if x < 20.0 else
                              2 if x < 50.0 else
                              3 if x < 100.0 else 4
                )
            
            # Select only necessary columns and drop any rows with NaN in critical columns
            items_hgt = items_df[selected_cols + ['price_level'] if 'price_level' in items_df.columns else selected_cols]
            items_hgt = items_hgt.dropna(subset=['id', 'name'])
            
            # Add entity type column and default coordinates (not location-based)
            items_hgt['entity_type'] = 'item'
            items_hgt['latitude'] = 0.0
            items_hgt['longitude'] = 0.0
            items_hgt['distance_meters'] = 0.0
            
            hgt_data['items'] = items_hgt
        else:
            hgt_data['items'] = pd.DataFrame()
        
        # Combine all entities into a single DataFrame if needed
        all_entities = pd.concat([
            hgt_data['places'], 
            hgt_data['events'], 
            hgt_data['items']
        ], ignore_index=True) if not all(df.empty for df in hgt_data.values()) else pd.DataFrame()
        
        if not all_entities.empty:
            hgt_data['all'] = all_entities
        
        return hgt_data


def create_fetcher(supabase_url: str = None, supabase_key: str = None, 
                  foursquare_api_key: str = None) -> EntityFetcher:
    """
    Factory function to create an EntityFetcher instance.
    
    Args:
        supabase_url (str, optional): Supabase URL. Defaults to env variable.
        supabase_key (str, optional): Supabase API key. Defaults to env variable.
        foursquare_api_key (str, optional): Foursquare API key. Defaults to env variable.
        
    Returns:
        EntityFetcher: A configured EntityFetcher instance
    """
    return EntityFetcher(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        foursquare_api_key=foursquare_api_key
    )


# Example usage:
if __name__ == "__main__":
    # Create fetcher instance
    fetcher = create_fetcher()
    
    # Example coordinates (Chicago, IL)
    latitude = 41.8781
    longitude = -87.6298
    
    # Fetch places
    places = fetcher.fetch_places(
        lat=latitude,
        lon=longitude,
        radius=1000,  # 1km radius
        limit=10,
        category="food"
    )
    
    if isinstance(places, pd.DataFrame) and not places.empty:
        print(f"Found {len(places)} places:")
        print(places[['name', 'category', 'rating', 'distance_meters']].head())
    else:
        print("No places found.")
    
    # Fetch all entity types and prepare for HGT
    hgt_data = fetcher.fetch_for_hgt(
        lat=latitude,
        lon=longitude,
        radius=1500,  # 1.5km radius
        limit=20
    )
    
    if 'all' in hgt_data and not hgt_data['all'].empty:
        print(f"\nData ready for HGT model:")
        print(f"Total entities: {len(hgt_data['all'])}")
        print(f"Entity types: {hgt_data['all']['entity_type'].value_counts().to_dict()}")
    else:
        print("No data found for HGT model.")