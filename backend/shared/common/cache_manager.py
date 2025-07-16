# backend/shared/common/cache_manager.py

import time
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple
import json
import re
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    response: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_accessed: float = 0.0
    similarity_score: float = 1.0
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() > (self.timestamp + self.ttl)
    
    @property
    def age(self) -> float:
        """Get age of cache entry in seconds"""
        return time.time() - self.timestamp

class ResponseCache:
    """Intelligent response cache with semantic similarity"""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.similarity_threshold = 0.85
        self.hit_count = 0
        self.miss_count = 0
        
        logger.info(f"💾 Response Cache initialized (max_size: {max_size}, ttl: {default_ttl}s)")
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for better matching"""
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common punctuation for similarity matching
        normalized = re.sub(r'[.,!?;:]', '', normalized)
        
        return normalized
    
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries"""
        
        # Normalize both queries
        norm1 = self._normalize_query(query1)
        norm2 = self._normalize_query(query2)
        
        # Exact match
        if norm1 == norm2:
            return 1.0
        
        # Simple word overlap similarity
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union)
        
        # Length similarity factor
        length_similarity = 1 - abs(len(norm1) - len(norm2)) / max(len(norm1), len(norm2), 1)
        
        # Combined similarity score
        return (jaccard_similarity * 0.7) + (length_similarity * 0.3)
    
    def _generate_cache_key(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate cache key for query"""
        
        # Create a consistent key based on normalized query
        normalized = self._normalize_query(query)
        
        # Add context if provided
        if context:
            context_str = json.dumps(context, sort_keys=True)
            key_material = f"{normalized}|{context_str}"
        else:
            key_material = normalized
        
        # Create hash for consistent key
        return hashlib.md5(key_material.encode()).hexdigest()
    
    def get_similar_response(self, query: str, context: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Get cached response for similar query"""
        
        # First, try exact match
        cache_key = self._generate_cache_key(query, context)
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if not entry.is_expired:
                entry.access_count += 1
                entry.last_accessed = time.time()
                self.hit_count += 1
                
                logger.info(f"💾 Cache HIT (exact): {query[:50]}...")
                # ✅ FIX: Return dict format compatible with ChatResponse
                if isinstance(entry.response, dict):
                    return {
                        **entry.response,
                        "cache_hit": True,
                        "similarity_score": 1.0,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # Fallback for non-dict responses
                    return {
                        "response": str(entry.response),
                        "model_used": "cache",
                        "confidence": 0.95,
                        "cache_hit": True,
                        "similarity_score": 1.0,
                        "response_time_ms": 50,
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                # Remove expired entry
                del self.cache[cache_key]
        
        # Then, try similarity matching
        best_match = None
        best_similarity = 0.0
        
        normalized_query = self._normalize_query(query)
        
        for key, entry in self.cache.items():
            if entry.is_expired:
                continue
                
            # Calculate similarity with cached queries
            # Note: In production, you'd want to store original queries with cache entries
            # For now, we'll use a simplified approach
            
            similarity = self._calculate_similarity(query, key)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = entry
        
        if best_match:
            best_match.access_count += 1
            best_match.last_accessed = time.time()
            self.hit_count += 1
            
            logger.info(f"💾 Cache HIT (similar): {query[:50]}... (similarity: {best_similarity:.2f})")
            # ✅ FIX: Return dict format
            if isinstance(best_match.response, dict):
                return {
                    **best_match.response,
                    "cache_hit": True,
                    "similarity_score": best_similarity
                }
            else:
                # Fallback for non-dict responses
                return {
                    "response": str(best_match.response),
                    "model_used": "cache-similar",
                    "confidence": 0.90,
                    "cache_hit": True,
                    "similarity_score": best_similarity,
                    "response_time_ms": 50
                }
        
        self.miss_count += 1
        logger.info(f"💾 Cache MISS: {query[:50]}...")
        return None
    
    def cache_response(self, query: str, response: Any, context: Optional[Dict] = None, ttl: Optional[float] = None):
        """Cache response for query"""
        
        cache_key = self._generate_cache_key(query, context)
        cache_ttl = ttl or self.default_ttl
        
        # Create cache entry
        entry = CacheEntry(
            response=response,
            timestamp=time.time(),
            ttl=cache_ttl,
            access_count=1,
            last_accessed=time.time()
        )
        
        # Store in cache
        self.cache[cache_key] = entry
        
        # Manage cache size
        self._evict_if_needed()
        
        logger.info(f"💾 Response cached: {query[:50]}... (TTL: {cache_ttl}s)")
    
    def _evict_if_needed(self):
        """Evict old entries if cache is full"""
        
        if len(self.cache) <= self.max_size:
            return
        
        # Remove expired entries first
        expired_keys = [
            key for key, entry in self.cache.items() 
            if entry.is_expired
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        # If still over limit, remove least recently used
        if len(self.cache) > self.max_size:
            # Sort by last accessed time (oldest first)
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Remove oldest entries
            entries_to_remove = len(self.cache) - self.max_size
            for i in range(entries_to_remove):
                key_to_remove = sorted_entries[i][0]
                del self.cache[key_to_remove]
                logger.info(f"💾 Cache evicted (LRU): {key_to_remove}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0.0
        
        # Count expired entries
        expired_count = sum(1 for entry in self.cache.values() if entry.is_expired)
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'expired_entries': expired_count,
            'memory_usage': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> str:
        """Estimate cache memory usage"""
        
        # Rough estimate based on number of entries
        estimated_bytes = len(self.cache) * 1024  # 1KB per entry estimate
        
        if estimated_bytes < 1024:
            return f"{estimated_bytes} bytes"
        elif estimated_bytes < 1024 * 1024:
            return f"{estimated_bytes / 1024:.1f} KB"
        else:
            return f"{estimated_bytes / (1024 * 1024):.1f} MB"
    
    def clear_cache(self):
        """Clear all cache entries"""
        cache_size = len(self.cache)
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        
        logger.info(f"💾 Cache cleared ({cache_size} entries removed)")
    
    def clear_expired(self):
        """Clear only expired entries"""
        
        expired_keys = [
            key for key, entry in self.cache.items() 
            if entry.is_expired
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        logger.info(f"💾 Expired entries cleared ({len(expired_keys)} entries)")
        
        return len(expired_keys)
    
    def warm_cache(self, queries_responses: Dict[str, Any], context: Optional[Dict] = None):
        """Pre-warm cache with common queries"""
        
        logger.info(f"💾 Warming cache with {len(queries_responses)} entries")
        
        for query, response in queries_responses.items():
            self.cache_response(query, response, context, ttl=self.default_ttl * 2)  # Longer TTL for warm entries
        
        logger.info(f"💾 Cache warmed successfully")
    
    def get_top_queries(self, limit: int = 10) -> list:
        """Get most accessed queries"""
        
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].access_count,
            reverse=True
        )
        
        return [
            {
                'query_hash': key,
                'access_count': entry.access_count,
                'age': entry.age,
                'last_accessed': entry.last_accessed
            }
            for key, entry in sorted_entries[:limit]
        ]

# Create global cache instance using the actual class name
global_cache = ResponseCache()

# Export both class and instance for flexibility  
cache_manager = global_cache  # Alias for compatibility
IntelligentCacheManager = ResponseCache  # Alias for backward compatibility

# Export convenience functions
async def get_cached_response(message: str, model: str = "gpt", context: Optional[Dict] = None):
    """Get cached response - convenience function"""
    result = global_cache.get_similar_response(message, context)
    if result:
        return result  # Now returns dict instead of tuple
    return None

async def cache_response(message: str, model: str, response: str, confidence: float = 0.8, 
                        response_time: float = 1.0, context: Optional[Dict] = None):
    """Cache response - convenience function"""
    global_cache.cache_response(message, response, context)
    return True

def get_cache_stats() -> Dict[str, Any]:
    """Get cache stats - convenience function"""
    return global_cache.get_cache_stats()

def clear_cache():
    """Clear cache - convenience function"""
    global_cache.clear_cache()

# Initialize logging
logger.info("💾 Cache manager module initialized")