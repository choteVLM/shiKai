"""
Database utilities.

This module contains utilities for working with vector databases.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

def create_vector_db(
    host: str = "localhost", 
    port: int = 6333, 
    collection_name: str = "frameRAG",
    vector_size: int = 512,
    distance: Distance = Distance.COSINE,
    recreate: bool = True
) -> QdrantClient:
    """
    Create or connect to a vector database.
    
    Args:
        host: Database host
        port: Database port
        collection_name: Name of the collection to create
        vector_size: Size of the vectors to store
        distance: Distance metric to use
        recreate: Whether to recreate the collection if it exists
        
    Returns:
        QdrantClient instance
        
    Raises:
        ConnectionError: If connection to the database fails
    """
    try:
        client = QdrantClient(host=host, port=port)
        
        # Check if collection exists and recreate if needed
        if client.collection_exists(collection_name=collection_name):
            if recreate:
                logger.info(f"Deleting existing collection: {collection_name}")
                client.delete_collection(collection_name=collection_name)
            else:
                logger.info(f"Using existing collection: {collection_name}")
                return client
        
        # Create the collection
        logger.info(f"Creating collection: {collection_name} with vector size {vector_size}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance
            )
        )
        
        return client
    except Exception as e:
        logger.error(f"Error creating vector database: {e}")
        raise ConnectionError(f"Failed to connect to vector database: {e}")

def store_embeddings(
    client: QdrantClient,
    collection_name: str,
    embeddings: List[List[float]],
    ids: Optional[List[Union[str, int]]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """
    Store embeddings in the vector database.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to store in
        embeddings: List of embeddings to store
        ids: Optional list of IDs for the embeddings
        metadata: Optional list of metadata for the embeddings
        
    Returns:
        True if successful, False otherwise
        
    Raises:
        ValueError: If inputs have inconsistent lengths
    """
    try:
        # Generate sequential IDs if not provided
        if ids is None:
            ids = list(range(len(embeddings)))
        
        # Validate input lengths
        if len(ids) != len(embeddings):
            raise ValueError(f"Length mismatch: {len(ids)} IDs vs {len(embeddings)} embeddings")
        
        if metadata is not None and len(metadata) != len(embeddings):
            raise ValueError(f"Length mismatch: {len(metadata)} metadata vs {len(embeddings)} embeddings")
        
        # Create points
        points = []
        for i, embedding in enumerate(embeddings):
            point = {
                "id": ids[i],
                "vector": embedding
            }
            
            if metadata is not None:
                point["payload"] = metadata[i]
                
            points.append(PointStruct(**point))
        
        # Upsert points
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        logger.info(f"Stored {len(embeddings)} embeddings in collection {collection_name}")
        return True
    
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")
        return False

def search_similar(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int = 5,
    filter_conditions: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for similar vectors in the database.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to search in
        query_vector: Vector to search for
        limit: Maximum number of results to return
        filter_conditions: Optional filter conditions
        
    Returns:
        List of results with id, score and payload
    """
    try:
        # Create filter if conditions provided
        search_filter = None
        if filter_conditions:
            conditions = []
            for field, value in filter_conditions.items():
                conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
            search_filter = Filter(must=conditions)
        
        # Perform search
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            filter=search_filter
        )
        
        # Format results
        formatted_results = []
        for res in results:
            formatted_results.append({
                "id": res.id,
                "score": res.score,
                "payload": res.payload
            })
            
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error searching for similar vectors: {e}")
        return [] 