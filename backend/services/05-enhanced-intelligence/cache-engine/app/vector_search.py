# backend/services/05-enhanced-intelligence/cache-engine/app/vector_search.py
"""
Vector Search Engine
Advanced vector similarity search for semantic caching
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple
import logging
import time
import math
import numpy as np
from datetime import datetime
from collections import defaultdict
import hashlib
import json

logger = logging.getLogger(__name__)

# Create router
vector_router = APIRouter()

# Models
class VectorSearchRequest(BaseModel):
    query_vector: List[float] = Field(..., description="Query vector for similarity search")
    top_k: int = Field(default=5, description="Number of top similar results to return")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity score")
    filter_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    include_scores: bool = Field(default=True, description="Include similarity scores")
    search_method: str = Field(default="cosine", description="cosine, euclidean, dot_product")

class VectorIndexRequest(BaseModel):
    key: str = Field(..., description="Unique key for the vector")
    vector: List[float] = Field(..., description="Vector to index")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Associated metadata")
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags for filtering")

class VectorSearchResponse(BaseModel):
    query_vector_dim: int
    results: List[Dict[str, Any]]
    total_found: int
    search_method: str
    similarity_threshold: float
    processing_time_seconds: float
    timestamp: str

class VectorStatsResponse(BaseModel):
    total_vectors: int
    vector_dimensions: int
    index_size_mb: float
    search_performance_ms: float
    similarity_methods: List[str]
    memory_usage_mb: float

class VectorSearchEngine:
    """Advanced Vector Search Engine for Semantic Similarity"""
    
    def __init__(self):
        # Vector storage
        self.vector_index = {}  # key -> vector data
        self.vectors_matrix = None  # Numpy array for efficient computation
        self.key_to_index = {}  # key -> matrix index
        self.index_to_key = {}  # matrix index -> key
        
        # Configuration
        self.max_vectors = 10000
        self.vector_dimension = None
        
        # Search methods
        self.search_methods = {
            "cosine": self._cosine_similarity_search,
            "euclidean": self._euclidean_similarity_search,
            "dot_product": self._dot_product_search,
            "manhattan": self._manhattan_similarity_search
        }
        
        # Statistics
        self.stats = {
            "vectors_indexed": 0,
            "searches_performed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_search_time": 0.0,
            "index_rebuilds": 0
        }
        
        # Search cache
        self.search_cache = {}
        self.cache_max_size = 500
        
        # Performance monitoring
        self.search_times = []
        self.max_search_times = 100
    
    async def index_vector(self, request: VectorIndexRequest) -> Dict[str, Any]:
        """Index a vector for similarity search"""
        
        try:
            # Validate vector
            if not request.vector:
                return {"success": False, "error": "Empty vector"}
            
            vector = np.array(request.vector, dtype=np.float32)
            
            # Set or validate dimensions
            if self.vector_dimension is None:
                self.vector_dimension = len(vector)
            elif len(vector) != self.vector_dimension:
                return {
                    "success": False, 
                    "error": f"Vector dimension mismatch. Expected {self.vector_dimension}, got {len(vector)}"
                }
            
            # Normalize vector
            normalized_vector = self._normalize_vector(vector)
            
            # Create vector entry
            vector_entry = {
                "key": request.key,
                "vector": normalized_vector,
                "original_vector": vector,
                "metadata": request.metadata or {},
                "tags": request.tags or [],
                "indexed_at": datetime.now(),
                "dimension": len(vector),
                "norm": np.linalg.norm(vector)
            }
            
            # Store in index
            self.vector_index[request.key] = vector_entry
            
            # Rebuild matrix if needed
            await self._rebuild_vectors_matrix()
            
            self.stats["vectors_indexed"] += 1
            
            logger.info(f"🔍 Vector indexed: {request.key} (dim: {len(vector)})")
            
            return {
                "success": True,
                "key": request.key,
                "vector_dimension": len(vector),
                "total_vectors": len(self.vector_index)
            }
            
        except Exception as e:
            logger.error(f"❌ Vector indexing error for {request.key}: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_vectors(self, request: VectorSearchRequest) -> Dict[str, Any]:
        """Search for similar vectors"""
        
        start_time = time.time()
        
        try:
            # Validate query vector
            if not request.query_vector:
                return {"results": [], "total_found": 0, "error": "Empty query vector"}
            
            query_vector = np.array(request.query_vector, dtype=np.float32)
            
            # Validate dimensions
            if self.vector_dimension and len(query_vector) != self.vector_dimension:
                return {
                    "results": [], 
                    "total_found": 0, 
                    "error": f"Query vector dimension mismatch. Expected {self.vector_dimension}, got {len(query_vector)}"
                }
            
            # Check cache
            cache_key = self._get_search_cache_key(request)
            if cache_key in self.search_cache:
                self.stats["cache_hits"] += 1
                cached_result = self.search_cache[cache_key]
                cached_result["from_cache"] = True
                return cached_result
            
            self.stats["cache_misses"] += 1
            
            # Normalize query vector
            normalized_query = self._normalize_vector(query_vector)
            
            # Perform search
            search_method = self.search_methods.get(request.search_method, self._cosine_similarity_search)
            similarities = search_method(normalized_query, request.filter_metadata)
            
            # Filter by threshold
            filtered_similarities = [
                sim for sim in similarities 
                if sim["similarity_score"] >= request.similarity_threshold
            ]
            
            # Sort by similarity (highest first)
            filtered_similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Limit results
            top_results = filtered_similarities[:request.top_k]
            
            # Format results
            formatted_results = []
            for result in top_results:
                formatted_result = {
                    "key": result["key"],
                    "metadata": result.get("metadata", {}),
                    "tags": result.get("tags", [])
                }
                
                if request.include_scores:
                    formatted_result["similarity_score"] = round(result["similarity_score"], 6)
                    formatted_result["distance"] = round(result.get("distance", 0), 6)
                
                formatted_results.append(formatted_result)
            
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            result = {
                "query_vector_dim": len(query_vector),
                "results": formatted_results,
                "total_found": len(filtered_similarities),
                "search_method": request.search_method,
                "similarity_threshold": request.similarity_threshold,
                "processing_time": processing_time
            }
            
            # Cache result
            self._cache_search_result(cache_key, result)
            
            logger.info(
                f"🔍 Vector search: {len(formatted_results)}/{len(filtered_similarities)} results "
                f"({request.search_method}, {processing_time:.4f}s)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Vector search error: {e}")
            return {
                "results": [],
                "total_found": 0,
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def remove_vector(self, key: str) -> Dict[str, Any]:
        """Remove vector from index"""
        
        try:
            if key not in self.vector_index:
                return {"success": False, "error": "Key not found"}
            
            # Remove from index
            del self.vector_index[key]
            
            # Rebuild matrix
            await self._rebuild_vectors_matrix()
            
            # Clear search cache
            self.search_cache.clear()
            
            logger.info(f"🗑️ Vector removed: {key}")
            
            return {
                "success": True,
                "key": key,
                "remaining_vectors": len(self.vector_index)
            }
            
        except Exception as e:
            logger.error(f"❌ Vector removal error for {key}: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_vector_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get information about indexed vector"""
        
        try:
            if key not in self.vector_index:
                return None
            
            entry = self.vector_index[key]
            
            return {
                "key": key,
                "vector_dimension": entry["dimension"],
                "vector_norm": float(entry["norm"]),
                "metadata": entry["metadata"],
                "tags": entry["tags"],
                "indexed_at": entry["indexed_at"].isoformat(),
                "vector_preview": entry["original_vector"][:10].tolist() if len(entry["original_vector"]) > 10 else entry["original_vector"].tolist()
            }
            
        except Exception as e:
            logger.error(f"❌ Get vector info error for {key}: {e}")
            return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector search engine statistics"""
        
        try:
            # Calculate memory usage
            vectors_memory = 0
            if self.vectors_matrix is not None:
                vectors_memory = self.vectors_matrix.nbytes
            
            index_memory = sum(
                len(str(entry)) for entry in self.vector_index.values()
            )
            
            total_memory_mb = (vectors_memory + index_memory) / (1024 * 1024)
            
            # Calculate average search time
            avg_search_time = (
                sum(self.search_times) / len(self.search_times) 
                if self.search_times else 0.0
            )
            
            return {
                "total_vectors": len(self.vector_index),
                "vector_dimensions": self.vector_dimension or 0,
                "index_size_mb": round(total_memory_mb, 2),
                "search_performance_ms": round(avg_search_time * 1000, 2),
                "similarity_methods": list(self.search_methods.keys()),
                "memory_usage_mb": round(total_memory_mb, 2),
                "search_cache_size": len(self.search_cache),
                "cache_hit_rate": round(
                    self.stats["cache_hits"] / max(self.stats["cache_hits"] + self.stats["cache_misses"], 1),
                    4
                ),
                "lifetime_stats": self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"❌ Get vector stats error: {e}")
            return {"error": str(e)}
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length"""
        
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    async def _rebuild_vectors_matrix(self):
        """Rebuild the vectors matrix for efficient computation"""
        
        try:
            if not self.vector_index:
                self.vectors_matrix = None
                self.key_to_index = {}
                self.index_to_key = {}
                return
            
            # Create matrix
            vectors = []
            keys = []
            
            for i, (key, entry) in enumerate(self.vector_index.items()):
                vectors.append(entry["vector"])
                keys.append(key)
                self.key_to_index[key] = i
                self.index_to_key[i] = key
            
            self.vectors_matrix = np.array(vectors, dtype=np.float32)
            self.stats["index_rebuilds"] += 1
            
            logger.debug(f"🔄 Vector matrix rebuilt: {len(keys)} vectors")
            
        except Exception as e:
            logger.error(f"❌ Rebuild vectors matrix error: {e}")
    
    def _cosine_similarity_search(self, query_vector: np.ndarray, 
                                 filter_metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform cosine similarity search"""
        
        if self.vectors_matrix is None or len(self.vectors_matrix) == 0:
            return []
        
        try:
            # Calculate cosine similarities
            similarities = np.dot(self.vectors_matrix, query_vector)
            
            results = []
            for i, similarity in enumerate(similarities):
                key = self.index_to_key[i]
                entry = self.vector_index[key]
                
                # Apply metadata filter
                if filter_metadata and not self._matches_filter(entry["metadata"], filter_metadata):
                    continue
                
                results.append({
                    "key": key,
                    "similarity_score": float(similarity),
                    "distance": float(1 - similarity),
                    "metadata": entry["metadata"],
                    "tags": entry["tags"]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Cosine similarity search error: {e}")
            return []
    
    def _euclidean_similarity_search(self, query_vector: np.ndarray, 
                                   filter_metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform Euclidean distance similarity search"""
        
        if self.vectors_matrix is None or len(self.vectors_matrix) == 0:
            return []
        
        try:
            # Calculate Euclidean distances
            distances = np.linalg.norm(self.vectors_matrix - query_vector, axis=1)
            
            # Convert distances to similarities (inverse relationship)
            max_distance = np.max(distances) if len(distances) > 0 else 1.0
            similarities = 1 - (distances / max_distance)
            
            results = []
            for i, (distance, similarity) in enumerate(zip(distances, similarities)):
                key = self.index_to_key[i]
                entry = self.vector_index[key]
                
                # Apply metadata filter
                if filter_metadata and not self._matches_filter(entry["metadata"], filter_metadata):
                    continue
                
                results.append({
                    "key": key,
                    "similarity_score": float(similarity),
                    "distance": float(distance),
                    "metadata": entry["metadata"],
                    "tags": entry["tags"]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Euclidean similarity search error: {e}")
            return []
    
    def _dot_product_search(self, query_vector: np.ndarray, 
                           filter_metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform dot product similarity search"""
        
        if self.vectors_matrix is None or len(self.vectors_matrix) == 0:
            return []
        
        try:
            # Calculate dot products
            dot_products = np.dot(self.vectors_matrix, query_vector)
            
            # Normalize to [0, 1] range
            min_dot = np.min(dot_products)
            max_dot = np.max(dot_products)
            
            if max_dot > min_dot:
                similarities = (dot_products - min_dot) / (max_dot - min_dot)
            else:
                similarities = np.ones_like(dot_products)
            
            results = []
            for i, (dot_product, similarity) in enumerate(zip(dot_products, similarities)):
                key = self.index_to_key[i]
                entry = self.vector_index[key]
                
                # Apply metadata filter
                if filter_metadata and not self._matches_filter(entry["metadata"], filter_metadata):
                    continue
                
                results.append({
                    "key": key,
                    "similarity_score": float(similarity),
                    "distance": float(-dot_product),  # Negative for distance interpretation
                    "metadata": entry["metadata"],
                    "tags": entry["tags"]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Dot product search error: {e}")
            return []
    
    def _manhattan_similarity_search(self, query_vector: np.ndarray, 
                                   filter_metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform Manhattan distance similarity search"""
        
        if self.vectors_matrix is None or len(self.vectors_matrix) == 0:
            return []
        
        try:
            # Calculate Manhattan distances
            distances = np.sum(np.abs(self.vectors_matrix - query_vector), axis=1)
            
            # Convert distances to similarities
            max_distance = np.max(distances) if len(distances) > 0 else 1.0
            similarities = 1 - (distances / max_distance)
            
            results = []
            for i, (distance, similarity) in enumerate(zip(distances, similarities)):
                key = self.index_to_key[i]
                entry = self.vector_index[key]
                
                # Apply metadata filter
                if filter_metadata and not self._matches_filter(entry["metadata"], filter_metadata):
                    continue
                
                results.append({
                    "key": key,
                    "similarity_score": float(similarity),
                    "distance": float(distance),
                    "metadata": entry["metadata"],
                    "tags": entry["tags"]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Manhattan similarity search error: {e}")
            return []
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        
        try:
            for key, value in filter_metadata.items():
                if key not in metadata:
                    return False
                
                metadata_value = metadata[key]
                
                # Exact match for strings and numbers
                if isinstance(value, (str, int, float, bool)):
                    if metadata_value != value:
                        return False
                
                # List membership check
                elif isinstance(value, list):
                    if metadata_value not in value:
                        return False
                
                # Dictionary subset check
                elif isinstance(value, dict):
                    if not isinstance(metadata_value, dict):
                        return False
                    for sub_key, sub_value in value.items():
                        if sub_key not in metadata_value or metadata_value[sub_key] != sub_value:
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Filter matching error: {e}")
            return False
    
    def _get_search_cache_key(self, request: VectorSearchRequest) -> str:
        """Generate cache key for search request"""
        
        try:
            # Create deterministic key from request
            cache_data = {
                "query_vector": request.query_vector,
                "top_k": request.top_k,
                "similarity_threshold": request.similarity_threshold,
                "filter_metadata": request.filter_metadata,
                "search_method": request.search_method
            }
            
            cache_string = json.dumps(cache_data, sort_keys=True)
            return hashlib.md5(cache_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"❌ Search cache key generation error: {e}")
            return f"error_{time.time()}"
    
    def _cache_search_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache search result"""
        
        try:
            # Implement LRU eviction
            if len(self.search_cache) >= self.cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self.search_cache))
                del self.search_cache[oldest_key]
            
            # Remove processing time and timestamp for caching
            cached_result = result.copy()
            cached_result.pop("processing_time", None)
            cached_result.pop("timestamp", None)
            
            self.search_cache[cache_key] = cached_result
            
        except Exception as e:
            logger.error(f"❌ Cache search result error: {e}")
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        
        try:
            self.stats["searches_performed"] += 1
            
            # Update search times for rolling average
            self.search_times.append(processing_time)
            if len(self.search_times) > self.max_search_times:
                self.search_times = self.search_times[-self.max_search_times:]
            
            # Update average search time
            self.stats["average_search_time"] = sum(self.search_times) / len(self.search_times)
            
        except Exception as e:
            logger.error(f"❌ Update performance stats error: {e}")

# Initialize vector search engine
vector_search_engine = VectorSearchEngine()

@vector_router.post("/index")
async def index_vector(request: VectorIndexRequest):
    """Index a vector for similarity search"""
    
    try:
        result = await vector_search_engine.index_vector(request)
        
        if result.get("success"):
            logger.info(f"🔍 Vector indexed: {request.key}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Index vector endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@vector_router.post("/search", response_model=VectorSearchResponse)
async def search_vectors(request: VectorSearchRequest):
    """Search for similar vectors"""
    
    try:
        result = await vector_search_engine.search_vectors(request)
        
        # Add timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        response = VectorSearchResponse(**result)
        
        logger.info(
            f"🔍 Vector search: {len(result.get('results', []))} results "
            f"({request.search_method})"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Search vectors endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@vector_router.delete("/remove/{key}")
async def remove_vector(key: str):
    """Remove vector from index"""
    
    try:
        result = await vector_search_engine.remove_vector(key)
        
        if result.get("success"):
            logger.info(f"🗑️ Vector removed: {key}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Remove vector endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@vector_router.get("/info/{key}")
async def get_vector_info(key: str):
    """Get information about indexed vector"""
    
    try:
        info = await vector_search_engine.get_vector_info(key)
        
        if info is None:
            raise HTTPException(status_code=404, detail="Vector not found")
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get vector info endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@vector_router.get("/stats", response_model=VectorStatsResponse)
async def get_vector_stats():
    """Get vector search engine statistics"""
    
    try:
        stats = await vector_search_engine.get_stats()
        
        return VectorStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"❌ Get vector stats endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@vector_router.get("/methods")
async def get_search_methods():
    """Get available search methods"""
    
    return {
        "search_methods": list(vector_search_engine.search_methods.keys()),
        "method_descriptions": {
            "cosine": "Cosine similarity - measures angle between vectors",
            "euclidean": "Euclidean distance - measures straight-line distance",
            "dot_product": "Dot product - measures vector alignment",
            "manhattan": "Manhattan distance - measures city-block distance"
        },
        "default_method": "cosine",
        "recommended_thresholds": {
            "cosine": 0.7,
            "euclidean": 0.6,
            "dot_product": 0.8,
            "manhattan": 0.5
        },
        "timestamp": datetime.now().isoformat()
    }

@vector_router.post("/bulk-index")
async def bulk_index_vectors(vectors: List[VectorIndexRequest]):
    """Bulk index multiple vectors"""
    
    try:
        results = []
        successful = 0
        failed = 0
        
        for vector_request in vectors[:100]:  # Limit to 100 vectors
            result = await vector_search_engine.index_vector(vector_request)
            
            if result.get("success"):
                successful += 1
            else:
                failed += 1
            
            results.append({
                "key": vector_request.key,
                "success": result.get("success", False),
                "error": result.get("error")
            })
        
        logger.info(f"🔍 Bulk indexing: {successful} successful, {failed} failed")
        
        return {
            "total_processed": len(results),
            "successful": successful,
            "failed": failed,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Bulk index vectors endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@vector_router.post("/clear")
async def clear_vector_index():
    """Clear all vectors from index"""
    
    try:
        total_vectors = len(vector_search_engine.vector_index)
        
        # Clear all data structures
        vector_search_engine.vector_index.clear()
        vector_search_engine.vectors_matrix = None
        vector_search_engine.key_to_index.clear()
        vector_search_engine.index_to_key.clear()
        vector_search_engine.search_cache.clear()
        vector_search_engine.vector_dimension = None
        
        logger.info(f"🧹 Vector index cleared: {total_vectors} vectors removed")
        
        return {
            "success": True,
            "vectors_removed": total_vectors,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Clear vector index endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))