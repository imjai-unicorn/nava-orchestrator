# backend/services/05-enhanced-intelligence/cache-engine/app/cache_manager.py
"""
Cache Manager
Multi-layer intelligent caching system
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import logging
import time
import hashlib
import json
import asyncio
from datetime import datetime, timedelta
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)

# Create router
cache_router = APIRouter()

# Models
class CacheRequest(BaseModel):
    key: str = Field(..., description="Cache key")
    value: Any = Field(..., description="Value to cache")
    ttl_seconds: Optional[int] = Field(None, description="Time to live in seconds")
    cache_level: str = Field(default="auto", description="memory, redis, database, auto")
    tags: Optional[List[str]] = Field(default_factory=list, description="Cache tags for grouping")
    priority: int = Field(default=1, description="Cache priority (1-10)")

class CacheGetRequest(BaseModel):
    key: str = Field(..., description="Cache key to retrieve")
    include_metadata: bool = Field(default=False, description="Include cache metadata")

class CacheResponse(BaseModel):
    key: str
    value: Any
    hit: bool
    cache_level: str
    ttl_remaining: Optional[int]
    created_at: str
    last_accessed: str
    access_count: int
    size_bytes: int

class CacheStatsResponse(BaseModel):
    total_keys: int
    memory_keys: int
    redis_keys: int
    database_keys: int
    hit_rate: float
    miss_rate: float
    total_hits: int
    total_misses: int
    memory_usage_mb: float
    cache_efficiency: float

class MultiLayerCache:
    """Multi-layer intelligent caching system"""
    
    def __init__(self):
        # Memory cache (Level 1)
        self.memory_cache = OrderedDict()
        self.memory_metadata = {}
        self.memory_lock = threading.RLock()
        
        # Cache configurations
        self.memory_max_size = 1000  # Max items in memory
        self.memory_max_mb = 100     # Max MB in memory
        self.default_ttl = 3600      # 1 hour
        
        # Cache statistics
        self.stats = {
            "total_hits": 0,
            "total_misses": 0,
            "memory_hits": 0,
            "redis_hits": 0,
            "database_hits": 0,
            "evictions": 0,
            "writes": 0
        }
        
        # Cache levels configuration
        self.cache_levels = {
            "memory": {
                "max_size": 1000,
                "max_mb": 100,
                "ttl_default": 1800,  # 30 minutes
                "priority_threshold": 3
            },
            "redis": {
                "max_size": 10000,
                "max_mb": 500,
                "ttl_default": 3600,  # 1 hour
                "priority_threshold": 1
            },
            "database": {
                "max_size": 100000,
                "ttl_default": 86400,  # 24 hours
                "priority_threshold": 0
            }
        }
        
        # Start background cleanup
        self._start_cleanup_task()
    
    async def set(self, request: CacheRequest) -> Dict[str, Any]:
        """Set value in cache with intelligent level selection"""
        
        try:
            # Generate cache key hash
            key_hash = self._generate_key_hash(request.key)
            
            # Serialize value
            serialized_value = self._serialize_value(request.value)
            value_size = len(serialized_value)
            
            # Determine optimal cache level
            cache_level = self._determine_cache_level(
                request.cache_level,
                value_size,
                request.priority,
                request.ttl_seconds
            )
            
            # Calculate TTL
            ttl = request.ttl_seconds or self.cache_levels[cache_level]["ttl_default"]
            expires_at = datetime.now() + timedelta(seconds=ttl)
            
            # Create cache entry
            cache_entry = {
                "value": serialized_value,
                "original_value": request.value,
                "created_at": datetime.now(),
                "expires_at": expires_at,
                "last_accessed": datetime.now(),
                "access_count": 0,
                "size_bytes": value_size,
                "priority": request.priority,
                "tags": request.tags or [],
                "cache_level": cache_level
            }
            
            # Store in appropriate cache level
            success = await self._store_in_cache_level(
                cache_level, 
                key_hash, 
                request.key, 
                cache_entry
            )
            
            if success:
                self.stats["writes"] += 1
                logger.info(f"🗄️ Cached {request.key} in {cache_level} (size: {value_size} bytes)")
            
            return {
                "success": success,
                "key": request.key,
                "cache_level": cache_level,
                "size_bytes": value_size,
                "ttl_seconds": ttl,
                "expires_at": expires_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Cache set error for {request.key}: {e}")
            return {"success": False, "error": str(e)}
    
    async def get(self, key: str, include_metadata: bool = False) -> Optional[Dict[str, Any]]:
        """Get value from cache with multi-level lookup"""
        
        try:
            key_hash = self._generate_key_hash(key)
            
            # Try memory cache first
            result = await self._get_from_memory(key_hash, key)
            if result:
                self.stats["total_hits"] += 1
                self.stats["memory_hits"] += 1
                
                if include_metadata:
                    return self._format_cache_response(key, result, "memory", True)
                return {"value": result["original_value"], "hit": True, "cache_level": "memory"}
            
            # Try Redis cache
            result = await self._get_from_redis(key_hash, key)
            if result:
                self.stats["total_hits"] += 1
                self.stats["redis_hits"] += 1
                
                # Promote to memory if high priority
                if result.get("priority", 1) >= 3:
                    await self._promote_to_memory(key_hash, key, result)
                
                if include_metadata:
                    return self._format_cache_response(key, result, "redis", True)
                return {"value": result["original_value"], "hit": True, "cache_level": "redis"}
            
            # Try Database cache
            result = await self._get_from_database(key_hash, key)
            if result:
                self.stats["total_hits"] += 1
                self.stats["database_hits"] += 1
                
                # Promote to Redis if accessed frequently
                if result.get("access_count", 0) > 5:
                    await self._promote_to_redis(key_hash, key, result)
                
                if include_metadata:
                    return self._format_cache_response(key, result, "database", True)
                return {"value": result["original_value"], "hit": True, "cache_level": "database"}
            
            # Cache miss
            self.stats["total_misses"] += 1
            logger.debug(f"🔍 Cache miss for {key}")
            
            return {"value": None, "hit": False, "cache_level": None}
            
        except Exception as e:
            logger.error(f"❌ Cache get error for {key}: {e}")
            return {"value": None, "hit": False, "error": str(e)}
    
    async def delete(self, key: str) -> Dict[str, Any]:
        """Delete key from all cache levels"""
        
        try:
            key_hash = self._generate_key_hash(key)
            deleted_levels = []
            
            # Delete from memory
            if await self._delete_from_memory(key_hash):
                deleted_levels.append("memory")
            
            # Delete from Redis
            if await self._delete_from_redis(key_hash):
                deleted_levels.append("redis")
            
            # Delete from Database
            if await self._delete_from_database(key_hash):
                deleted_levels.append("database")
            
            return {
                "success": len(deleted_levels) > 0,
                "key": key,
                "deleted_from": deleted_levels
            }
            
        except Exception as e:
            logger.error(f"❌ Cache delete error for {key}: {e}")
            return {"success": False, "error": str(e)}
    
    async def clear_by_tags(self, tags: List[str]) -> Dict[str, Any]:
        """Clear cache entries by tags"""
        
        try:
            deleted_count = 0
            
            # Clear from memory
            deleted_count += await self._clear_memory_by_tags(tags)
            
            # Clear from Redis
            deleted_count += await self._clear_redis_by_tags(tags)
            
            # Clear from Database
            deleted_count += await self._clear_database_by_tags(tags)
            
            return {
                "success": True,
                "deleted_count": deleted_count,
                "tags": tags
            }
            
        except Exception as e:
            logger.error(f"❌ Cache clear by tags error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        try:
            # Calculate hit rate
            total_requests = self.stats["total_hits"] + self.stats["total_misses"]
            hit_rate = self.stats["total_hits"] / max(total_requests, 1)
            
            # Memory usage calculation
            memory_size_bytes = sum(
                entry.get("size_bytes", 0) 
                for entry in self.memory_metadata.values()
            )
            memory_usage_mb = memory_size_bytes / (1024 * 1024)
            
            # Cache efficiency
            cache_efficiency = hit_rate * 0.7 + (1 - memory_usage_mb / 100) * 0.3
            
            return {
                "total_keys": len(self.memory_cache) + await self._count_redis_keys() + await self._count_database_keys(),
                "memory_keys": len(self.memory_cache),
                "redis_keys": await self._count_redis_keys(),
                "database_keys": await self._count_database_keys(),
                "hit_rate": round(hit_rate, 4),
                "miss_rate": round(1 - hit_rate, 4),
                "total_hits": self.stats["total_hits"],
                "total_misses": self.stats["total_misses"],
                "memory_hits": self.stats["memory_hits"],
                "redis_hits": self.stats["redis_hits"],
                "database_hits": self.stats["database_hits"],
                "memory_usage_mb": round(memory_usage_mb, 2),
                "cache_efficiency": round(cache_efficiency, 4),
                "evictions": self.stats["evictions"],
                "writes": self.stats["writes"]
            }
            
        except Exception as e:
            logger.error(f"❌ Cache stats error: {e}")
            return {"error": str(e)}
    
    def _generate_key_hash(self, key: str) -> str:
        """Generate consistent hash for cache key"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _serialize_value(self, value: Any) -> str:
        """Serialize value for storage"""
        return json.dumps(value, default=str, ensure_ascii=False)
    
    def _deserialize_value(self, serialized: str) -> Any:
        """Deserialize value from storage"""
        try:
            return json.loads(serialized)
        except:
            return serialized
    
    def _determine_cache_level(self, requested_level: str, size_bytes: int, 
                              priority: int, ttl_seconds: Optional[int]) -> str:
        """Determine optimal cache level for storage"""
        
        if requested_level != "auto":
            return requested_level
        
        # Memory for small, high-priority, short-lived data
        if (size_bytes < 10240 and  # < 10KB
            priority >= 3 and 
            (ttl_seconds is None or ttl_seconds <= 1800)):  # <= 30 min
            return "memory"
        
        # Redis for medium-sized, medium-priority data
        if (size_bytes < 102400 and  # < 100KB
            priority >= 1 and
            (ttl_seconds is None or ttl_seconds <= 3600)):  # <= 1 hour
            return "redis"
        
        # Database for large or long-lived data
        return "database"
    
    async def _store_in_cache_level(self, level: str, key_hash: str, 
                                   original_key: str, cache_entry: Dict[str, Any]) -> bool:
        """Store cache entry in specified level"""
        
        if level == "memory":
            return await self._store_in_memory(key_hash, original_key, cache_entry)
        elif level == "redis":
            return await self._store_in_redis(key_hash, original_key, cache_entry)
        elif level == "database":
            return await self._store_in_database(key_hash, original_key, cache_entry)
        
        return False
    
    async def _store_in_memory(self, key_hash: str, original_key: str, 
                              cache_entry: Dict[str, Any]) -> bool:
        """Store in memory cache"""
        
        try:
            with self.memory_lock:
                # Check if eviction needed
                if len(self.memory_cache) >= self.memory_max_size:
                    self._evict_memory_lru()
                
                # Store entry
                self.memory_cache[key_hash] = cache_entry["original_value"]
                self.memory_metadata[key_hash] = cache_entry
                
                # Move to end (most recently used)
                self.memory_cache.move_to_end(key_hash)
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Memory cache store error: {e}")
            return False
    
    async def _get_from_memory(self, key_hash: str, original_key: str) -> Optional[Dict[str, Any]]:
        """Get from memory cache"""
        
        try:
            with self.memory_lock:
                if key_hash in self.memory_cache:
                    # Check TTL
                    metadata = self.memory_metadata.get(key_hash, {})
                    if self._is_expired(metadata):
                        del self.memory_cache[key_hash]
                        del self.memory_metadata[key_hash]
                        return None
                    
                    # Update access info
                    metadata["last_accessed"] = datetime.now()
                    metadata["access_count"] = metadata.get("access_count", 0) + 1
                    
                    # Move to end (most recently used)
                    self.memory_cache.move_to_end(key_hash)
                    
                    return metadata
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Memory cache get error: {e}")
            return None
    
    async def _delete_from_memory(self, key_hash: str) -> bool:
        """Delete from memory cache"""
        
        try:
            with self.memory_lock:
                if key_hash in self.memory_cache:
                    del self.memory_cache[key_hash]
                    del self.memory_metadata[key_hash]
                    return True
                return False
                
        except Exception as e:
            logger.error(f"❌ Memory cache delete error: {e}")
            return False
    
    def _evict_memory_lru(self):
        """Evict least recently used items from memory"""
        
        try:
            # Remove oldest item
            if self.memory_cache:
                oldest_key = next(iter(self.memory_cache))
                del self.memory_cache[oldest_key]
                if oldest_key in self.memory_metadata:
                    del self.memory_metadata[oldest_key]
                self.stats["evictions"] += 1
                
        except Exception as e:
            logger.error(f"❌ Memory eviction error: {e}")
    
    async def _store_in_redis(self, key_hash: str, original_key: str, 
                             cache_entry: Dict[str, Any]) -> bool:
        """Store in Redis cache (simulated for now)"""
        
        try:
            # Simulate Redis storage
            # In real implementation, use aioredis
            logger.debug(f"📝 Redis store simulated for {original_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Redis cache store error: {e}")
            return False
    
    async def _get_from_redis(self, key_hash: str, original_key: str) -> Optional[Dict[str, Any]]:
        """Get from Redis cache (simulated for now)"""
        
        try:
            # Simulate Redis retrieval
            # In real implementation, use aioredis
            return None
            
        except Exception as e:
            logger.error(f"❌ Redis cache get error: {e}")
            return None
    
    async def _delete_from_redis(self, key_hash: str) -> bool:
        """Delete from Redis cache (simulated for now)"""
        
        try:
            # Simulate Redis deletion
            return False
            
        except Exception as e:
            logger.error(f"❌ Redis cache delete error: {e}")
            return False
    
    async def _store_in_database(self, key_hash: str, original_key: str, 
                                cache_entry: Dict[str, Any]) -> bool:
        """Store in database cache (simulated for now)"""
        
        try:
            # Simulate database storage
            # In real implementation, use asyncpg or similar
            logger.debug(f"💾 Database store simulated for {original_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database cache store error: {e}")
            return False
    
    async def _get_from_database(self, key_hash: str, original_key: str) -> Optional[Dict[str, Any]]:
        """Get from database cache (simulated for now)"""
        
        try:
            # Simulate database retrieval
            return None
            
        except Exception as e:
            logger.error(f"❌ Database cache get error: {e}")
            return None
    
    async def _delete_from_database(self, key_hash: str) -> bool:
        """Delete from database cache (simulated for now)"""
        
        try:
            # Simulate database deletion
            return False
            
        except Exception as e:
            logger.error(f"❌ Database cache delete error: {e}")
            return False
    
    async def _promote_to_memory(self, key_hash: str, original_key: str, cache_entry: Dict[str, Any]):
        """Promote cache entry to memory"""
        
        cache_entry["cache_level"] = "memory"
        await self._store_in_memory(key_hash, original_key, cache_entry)
    
    async def _promote_to_redis(self, key_hash: str, original_key: str, cache_entry: Dict[str, Any]):
        """Promote cache entry to Redis"""
        
        cache_entry["cache_level"] = "redis"
        await self._store_in_redis(key_hash, original_key, cache_entry)
    
    def _is_expired(self, metadata: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        
        expires_at = metadata.get("expires_at")
        if expires_at and isinstance(expires_at, datetime):
            return datetime.now() > expires_at
        return False
    
    def _format_cache_response(self, key: str, cache_entry: Dict[str, Any], 
                              cache_level: str, hit: bool) -> Dict[str, Any]:
        """Format cache response with metadata"""
        
        expires_at = cache_entry.get("expires_at")
        ttl_remaining = None
        
        if expires_at and isinstance(expires_at, datetime):
            ttl_remaining = max(0, int((expires_at - datetime.now()).total_seconds()))
        
        return {
            "key": key,
            "value": cache_entry.get("original_value"),
            "hit": hit,
            "cache_level": cache_level,
            "ttl_remaining": ttl_remaining,
            "created_at": cache_entry.get("created_at", datetime.now()).isoformat(),
            "last_accessed": cache_entry.get("last_accessed", datetime.now()).isoformat(),
            "access_count": cache_entry.get("access_count", 0),
            "size_bytes": cache_entry.get("size_bytes", 0)
        }
    
    async def _clear_memory_by_tags(self, tags: List[str]) -> int:
        """Clear memory cache entries by tags"""
        
        deleted_count = 0
        try:
            with self.memory_lock:
                keys_to_delete = []
                
                for key_hash, metadata in self.memory_metadata.items():
                    entry_tags = metadata.get("tags", [])
                    if any(tag in entry_tags for tag in tags):
                        keys_to_delete.append(key_hash)
                
                for key_hash in keys_to_delete:
                    if key_hash in self.memory_cache:
                        del self.memory_cache[key_hash]
                    if key_hash in self.memory_metadata:
                        del self.memory_metadata[key_hash]
                    deleted_count += 1
                    
        except Exception as e:
            logger.error(f"❌ Memory clear by tags error: {e}")
        
        return deleted_count
    
    async def _clear_redis_by_tags(self, tags: List[str]) -> int:
        """Clear Redis cache entries by tags (simulated)"""
        return 0
    
    async def _clear_database_by_tags(self, tags: List[str]) -> int:
        """Clear database cache entries by tags (simulated)"""
        return 0
    
    async def _count_redis_keys(self) -> int:
        """Count Redis keys (simulated)"""
        return 0
    
    async def _count_database_keys(self) -> int:
        """Count database keys (simulated)"""
        return 0
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        
        async def cleanup_expired():
            while True:
                try:
                    await self._cleanup_expired_entries()
                    await asyncio.sleep(300)  # Cleanup every 5 minutes
                except Exception as e:
                    logger.error(f"❌ Cleanup task error: {e}")
                    await asyncio.sleep(60)  # Retry after 1 minute
        
        # Start cleanup task
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(cleanup_expired())
        except:
            # If no event loop, skip background task
            pass
    
    async def _cleanup_expired_entries(self):
        """Cleanup expired cache entries"""
        
        try:
            with self.memory_lock:
                expired_keys = []
                
                for key_hash, metadata in self.memory_metadata.items():
                    if self._is_expired(metadata):
                        expired_keys.append(key_hash)
                
                for key_hash in expired_keys:
                    if key_hash in self.memory_cache:
                        del self.memory_cache[key_hash]
                    if key_hash in self.memory_metadata:
                        del self.memory_metadata[key_hash]
                
                if expired_keys:
                    logger.info(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
                    
        except Exception as e:
            logger.error(f"❌ Cleanup expired entries error: {e}")

# Initialize cache manager
cache_manager = MultiLayerCache()

@cache_router.post("/set")
async def set_cache(request: CacheRequest):
    """Set value in cache"""
    
    try:
        result = await cache_manager.set(request)
        
        if result.get("success"):
            logger.info(f"💾 Cache set: {request.key} → {result['cache_level']}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Cache set endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@cache_router.post("/get", response_model=Union[CacheResponse, Dict[str, Any]])
async def get_cache(request: CacheGetRequest):
    """Get value from cache"""
    
    try:
        result = await cache_manager.get(request.key, request.include_metadata)
        
        if result.get("hit"):
            logger.debug(f"✅ Cache hit: {request.key} from {result['cache_level']}")
        else:
            logger.debug(f"❌ Cache miss: {request.key}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Cache get endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@cache_router.delete("/delete/{key}")
async def delete_cache(key: str):
    """Delete key from cache"""
    
    try:
        result = await cache_manager.delete(key)
        
        if result.get("success"):
            logger.info(f"🗑️ Cache deleted: {key} from {result['deleted_from']}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Cache delete endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@cache_router.post("/clear-by-tags")
async def clear_cache_by_tags(tags: List[str]):
    """Clear cache entries by tags"""
    
    try:
        result = await cache_manager.clear_by_tags(tags)
        
        if result.get("success"):
            logger.info(f"🧹 Cache cleared by tags: {tags} ({result['deleted_count']} entries)")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Cache clear by tags endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@cache_router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Get cache statistics"""
    
    try:
        stats = await cache_manager.get_stats()
        
        return CacheStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"❌ Cache stats endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@cache_router.post("/flush")
async def flush_cache(level: Optional[str] = None):
    """Flush cache (all levels or specific level)"""
    
    try:
        flushed_levels = []
        
        if level is None or level == "memory":
            with cache_manager.memory_lock:
                cache_manager.memory_cache.clear()
                cache_manager.memory_metadata.clear()
                flushed_levels.append("memory")
        
        # Add Redis and Database flush when implemented
        
        logger.info(f"🧹 Cache flushed: {flushed_levels}")
        
        return {
            "success": True,
            "flushed_levels": flushed_levels,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Cache flush endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))