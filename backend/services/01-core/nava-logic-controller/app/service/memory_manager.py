# backend/services/shared/common/memory_manager.py
"""
Memory Manager
Advanced memory optimization and context management system
"""

import logging
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta
import threading
import weakref

logger = logging.getLogger(__name__)

class MemoryType(str, Enum):
    """Types of memory storage"""
    SHORT_TERM = "short_term"       # Current session/conversation
    WORKING = "working"             # Active processing context
    EPISODIC = "episodic"          # Specific events/interactions
    SEMANTIC = "semantic"          # General knowledge/patterns
    PROCEDURAL = "procedural"      # Learned procedures/skills
    CACHE = "cache"                # Performance optimization cache

class PriorityLevel(str, Enum):
    """Memory priority levels"""
    CRITICAL = "critical"          # Must not be removed
    HIGH = "high"                  # Important, rarely removed
    MEDIUM = "medium"              # Moderate importance
    LOW = "low"                    # Can be removed if needed
    DISPOSABLE = "disposable"      # Remove first

class CompressionType(str, Enum):
    """Memory compression strategies"""
    NONE = "none"                  # No compression
    SUMMARY = "summary"            # Summarize content
    KEYWORDS = "keywords"          # Extract key information
    HIERARCHICAL = "hierarchical"  # Hierarchical compression
    SEMANTIC = "semantic"          # Semantic compression
    HYBRID = "hybrid"              # Multiple strategies

@dataclass
class MemoryItem:
    """Individual memory item"""
    memory_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    priority: PriorityLevel
    
    # Metadata
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    
    # Relationships
    related_items: List[str] = None
    parent_id: Optional[str] = None
    child_ids: List[str] = None
    
    # Compression
    is_compressed: bool = False
    compression_type: Optional[CompressionType] = None
    original_size: Optional[int] = None
    
    # Context
    context_tags: List[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    def __post_init__(self):
        if self.related_items is None:
            self.related_items = []
        if self.child_ids is None:
            self.child_ids = []
        if self.context_tags is None:
            self.context_tags = []
        
        # Calculate size if not provided
        if self.size_bytes == 0:
            self.size_bytes = len(json.dumps(self.content, default=str).encode('utf-8'))

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_items: int
    total_size_bytes: int
    items_by_type: Dict[MemoryType, int]
    size_by_type: Dict[MemoryType, int]
    items_by_priority: Dict[PriorityLevel, int]
    compressed_items: int
    compression_ratio: float
    cache_hit_rate: float
    average_access_time: float
    last_cleanup: datetime
    
class MemoryManager:
    """Advanced memory optimization and context management"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Memory storage
        self.memories: OrderedDict[str, MemoryItem] = OrderedDict()
        self.memory_index: Dict[str, List[str]] = defaultdict(list)  # Context-based indexing
        
        # Configuration
        self.max_memory_size = self.config.get('max_memory_size', 100 * 1024 * 1024)  # 100MB
        self.max_items = self.config.get('max_items', 10000)
        self.cleanup_threshold = self.config.get('cleanup_threshold', 0.8)  # Cleanup at 80% capacity
        self.compression_threshold = self.config.get('compression_threshold', 1024)  # 1KB
        
        # Performance tracking
        self.access_times: List[float] = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_cleanup = datetime.now()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Optimization features
        self.auto_compression = self.config.get('auto_compression', True)
        self.smart_eviction = self.config.get('smart_eviction', True)
        self.context_awareness = self.config.get('context_awareness', True)
        
        logger.info("🧠 Memory Manager initialized with intelligent optimization")
    
    def store_memory(self, 
                    memory_id: str,
                    content: Dict[str, Any],
                    memory_type: MemoryType,
                    priority: PriorityLevel = PriorityLevel.MEDIUM,
                    context_tags: Optional[List[str]] = None,
                    session_id: Optional[str] = None,
                    user_id: Optional[str] = None) -> bool:
        """Store a memory item with intelligent optimization"""
        
        with self.lock:
            try:
                # Create memory item
                memory_item = MemoryItem(
                    memory_id=memory_id,
                    memory_type=memory_type,
                    content=content,
                    priority=priority,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    context_tags=context_tags or [],
                    session_id=session_id,
                    user_id=user_id
                )
                
                # Check if compression is needed
                if self.auto_compression and memory_item.size_bytes > self.compression_threshold:
                    memory_item = self._compress_memory_item(memory_item)
                
                # Check capacity and cleanup if needed
                if self._should_cleanup():
                    self._intelligent_cleanup()
                
                # Store memory
                self.memories[memory_id] = memory_item
                
                # Update indexes
                self._update_indexes(memory_item)
                
                logger.debug(f"✅ Stored memory: {memory_id} ({memory_item.size_bytes} bytes)")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to store memory {memory_id}: {e}")
                return False
    
    def retrieve_memory(self, 
                       memory_id: str,
                       update_access: bool = True) -> Optional[MemoryItem]:
        """Retrieve a memory item"""
        
        start_time = time.time()
        
        with self.lock:
            memory_item = self.memories.get(memory_id)
            
            if memory_item:
                # Update access statistics
                if update_access:
                    memory_item.last_accessed = datetime.now()
                    memory_item.access_count += 1
                    
                    # Move to end (LRU optimization)
                    self.memories.move_to_end(memory_id)
                
                # Decompress if needed
                if memory_item.is_compressed:
                    memory_item = self._decompress_memory_item(memory_item)
                
                self.cache_hits += 1
                logger.debug(f"✅ Retrieved memory: {memory_id}")
            else:
                self.cache_misses += 1
                logger.debug(f"❌ Memory not found: {memory_id}")
            
            # Track access time
            access_time = time.time() - start_time
            self.access_times.append(access_time)
            if len(self.access_times) > 1000:  # Keep last 1000 measurements
                self.access_times = self.access_times[-1000:]
            
            return memory_item
    
    def search_memories(self, 
                       query: str,
                       memory_type: Optional[MemoryType] = None,
                       context_tags: Optional[List[str]] = None,
                       session_id: Optional[str] = None,
                       limit: int = 10) -> List[MemoryItem]:
        """Search memories with intelligent matching"""
        
        with self.lock:
            candidates = list(self.memories.values())
            
            # Filter by type
            if memory_type:
                candidates = [m for m in candidates if m.memory_type == memory_type]
            
            # Filter by session
            if session_id:
                candidates = [m for m in candidates if m.session_id == session_id]
            
            # Filter by context tags
            if context_tags:
                candidates = [m for m in candidates if any(tag in m.context_tags for tag in context_tags)]
            
            # Score and rank candidates
            scored_candidates = []
            for memory in candidates:
                score = self._calculate_relevance_score(memory, query, context_tags)
                if score > 0:
                    scored_candidates.append((score, memory))
            
            # Sort by score and return top results
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            results = [memory for _, memory in scored_candidates[:limit]]
            
            # Update access for retrieved memories
            for memory in results:
                memory.last_accessed = datetime.now()
                memory.access_count += 1
            
            logger.debug(f"🔍 Found {len(results)} memories for query: {query[:50]}...")
            return results
    
    def get_related_memories(self, 
                           memory_id: str,
                           depth: int = 1,
                           max_results: int = 20) -> List[MemoryItem]:
        """Get memories related to a specific memory item"""
        
        with self.lock:
            base_memory = self.memories.get(memory_id)
            if not base_memory:
                return []
            
            related_ids = set()
            to_explore = [(memory_id, 0)]  # (id, current_depth)
            
            while to_explore and len(related_ids) < max_results:
                current_id, current_depth = to_explore.pop(0)
                
                if current_depth >= depth:
                    continue
                
                current_memory = self.memories.get(current_id)
                if not current_memory:
                    continue
                
                # Add directly related items
                for related_id in current_memory.related_items:
                    if related_id not in related_ids and related_id != memory_id:
                        related_ids.add(related_id)
                        to_explore.append((related_id, current_depth + 1))
                
                # Add parent and children
                if current_memory.parent_id and current_memory.parent_id not in related_ids:
                    related_ids.add(current_memory.parent_id)
                    to_explore.append((current_memory.parent_id, current_depth + 1))
                
                for child_id in current_memory.child_ids:
                    if child_id not in related_ids:
                        related_ids.add(child_id)
                        to_explore.append((child_id, current_depth + 1))
            
            # Return actual memory objects
            related_memories = []
            for related_id in related_ids:
                memory = self.memories.get(related_id)
                if memory:
                    related_memories.append(memory)
            
            # Sort by relevance
            related_memories.sort(key=lambda m: (m.priority.value, -m.access_count, -m.last_accessed.timestamp()))
            
            return related_memories[:max_results]
    
    def update_memory(self, 
                     memory_id: str,
                     content: Optional[Dict[str, Any]] = None,
                     priority: Optional[PriorityLevel] = None,
                     context_tags: Optional[List[str]] = None) -> bool:
        """Update an existing memory item"""
        
        with self.lock:
            memory_item = self.memories.get(memory_id)
            if not memory_item:
                return False
            
            try:
                # Update content
                if content is not None:
                    memory_item.content = content
                    memory_item.size_bytes = len(json.dumps(content, default=str).encode('utf-8'))
                    
                    # Recompress if needed
                    if self.auto_compression and memory_item.size_bytes > self.compression_threshold:
                        memory_item = self._compress_memory_item(memory_item)
                
                # Update priority
                if priority is not None:
                    memory_item.priority = priority
                
                # Update context tags
                if context_tags is not None:
                    memory_item.context_tags = context_tags
                
                # Update access info
                memory_item.last_accessed = datetime.now()
                memory_item.access_count += 1
                
                # Update indexes
                self._update_indexes(memory_item)
                
                logger.debug(f"✅ Updated memory: {memory_id}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to update memory {memory_id}: {e}")
                return False
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory item and cleanup references"""
        
        with self.lock:
            memory_item = self.memories.get(memory_id)
            if not memory_item:
                return False
            
            try:
                # Remove from related items
                for related_id in memory_item.related_items:
                    related_memory = self.memories.get(related_id)
                    if related_memory and memory_id in related_memory.related_items:
                        related_memory.related_items.remove(memory_id)
                
                # Update parent-child relationships
                if memory_item.parent_id:
                    parent = self.memories.get(memory_item.parent_id)
                    if parent and memory_id in parent.child_ids:
                        parent.child_ids.remove(memory_id)
                
                for child_id in memory_item.child_ids:
                    child = self.memories.get(child_id)
                    if child:
                        child.parent_id = None
                
                # Remove from indexes
                self._remove_from_indexes(memory_item)
                
                # Delete the memory
                del self.memories[memory_id]
                
                logger.debug(f"✅ Deleted memory: {memory_id}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to delete memory {memory_id}: {e}")
                return False
    
    def add_relationship(self, memory_id1: str, memory_id2: str, bidirectional: bool = True) -> bool:
        """Add relationship between two memories"""
        
        with self.lock:
            memory1 = self.memories.get(memory_id1)
            memory2 = self.memories.get(memory_id2)
            
            if not memory1 or not memory2:
                return False
            
            # Add relationship
            if memory_id2 not in memory1.related_items:
                memory1.related_items.append(memory_id2)
            
            if bidirectional and memory_id1 not in memory2.related_items:
                memory2.related_items.append(memory_id1)
            
            return True
    
    def create_memory_hierarchy(self, parent_id: str, child_ids: List[str]) -> bool:
        """Create parent-child relationships between memories"""
        
        with self.lock:
            parent = self.memories.get(parent_id)
            if not parent:
                return False
            
            for child_id in child_ids:
                child = self.memories.get(child_id)
                if child:
                    child.parent_id = parent_id
                    if child_id not in parent.child_ids:
                        parent.child_ids.append(child_id)
            
            return True
    
    def compress_memory_type(self, memory_type: MemoryType) -> int:
        """Compress all memories of a specific type"""
        
        compressed_count = 0
        
        with self.lock:
            for memory in self.memories.values():
                if memory.memory_type == memory_type and not memory.is_compressed:
                    if memory.size_bytes > self.compression_threshold:
                        compressed_memory = self._compress_memory_item(memory)
                        if compressed_memory.is_compressed:
                            compressed_count += 1
        
        logger.info(f"🗜️ Compressed {compressed_count} memories of type {memory_type}")
        return compressed_count
    
    def cleanup_old_memories(self, max_age_days: int = 30) -> int:
        """Remove memories older than specified age"""
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        removed_count = 0
        
        with self.lock:
            to_remove = []
            
            for memory_id, memory in self.memories.items():
                if memory.last_accessed < cutoff_date and memory.priority != PriorityLevel.CRITICAL:
                    to_remove.append(memory_id)
            
            for memory_id in to_remove:
                if self.delete_memory(memory_id):
                    removed_count += 1
        
        logger.info(f"🧹 Cleaned up {removed_count} old memories")
        return removed_count
    
    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory usage statistics"""
        
        with self.lock:
            total_size = sum(memory.size_bytes for memory in self.memories.values())
            
            # Count by type
            items_by_type = defaultdict(int)
            size_by_type = defaultdict(int)
            for memory in self.memories.values():
                items_by_type[memory.memory_type] += 1
                size_by_type[memory.memory_type] += memory.size_bytes
            
            # Count by priority
            items_by_priority = defaultdict(int)
            for memory in self.memories.values():
                items_by_priority[memory.priority] += 1
            
            # Compression stats
            compressed_items = sum(1 for memory in self.memories.values() if memory.is_compressed)
            total_original_size = sum(
                memory.original_size or memory.size_bytes 
                for memory in self.memories.values()
            )
            compression_ratio = 1 - (total_size / total_original_size) if total_original_size > 0 else 0
            
            # Cache stats
            total_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
            
            # Average access time
            avg_access_time = sum(self.access_times) / len(self.access_times) if self.access_times else 0
            
            return MemoryStats(
                total_items=len(self.memories),
                total_size_bytes=total_size,
                items_by_type=dict(items_by_type),
                size_by_type=dict(size_by_type),
                items_by_priority=dict(items_by_priority),
                compressed_items=compressed_items,
                compression_ratio=compression_ratio,
                cache_hit_rate=cache_hit_rate,
                average_access_time=avg_access_time,
                last_cleanup=self.last_cleanup
            )
    
    def optimize_memory(self) -> Dict[str, int]:
        """Perform comprehensive memory optimization"""
        
        optimization_results = {
            "items_compressed": 0,
            "items_cleaned": 0,
            "relationships_optimized": 0,
            "indexes_rebuilt": 0
        }
        
        with self.lock:
            # Compress large items
            for memory in self.memories.values():
                if (not memory.is_compressed and 
                    memory.size_bytes > self.compression_threshold and
                    memory.access_count < 5):  # Rarely accessed items
                    compressed = self._compress_memory_item(memory)
                    if compressed.is_compressed:
                        optimization_results["items_compressed"] += 1
            
            # Clean up broken relationships
            for memory in self.memories.values():
                # Remove invalid related items
                valid_related = [rid for rid in memory.related_items if rid in self.memories]
                if len(valid_related) != len(memory.related_items):
                    memory.related_items = valid_related
                    optimization_results["relationships_optimized"] += 1
                
                # Remove invalid child references
                valid_children = [cid for cid in memory.child_ids if cid in self.memories]
                if len(valid_children) != len(memory.child_ids):
                    memory.child_ids = valid_children
                    optimization_results["relationships_optimized"] += 1
            
            # Rebuild indexes
            self._rebuild_indexes()
            optimization_results["indexes_rebuilt"] = 1
            
            # Intelligent cleanup
            if self._should_cleanup():
                removed = self._intelligent_cleanup()
                optimization_results["items_cleaned"] = removed
        
        logger.info(f"🚀 Memory optimization completed: {optimization_results}")
        return optimization_results
    
    def _should_cleanup(self) -> bool:
        """Check if memory cleanup is needed"""
        current_size = sum(memory.size_bytes for memory in self.memories.values())
        size_ratio = current_size / self.max_memory_size
        item_ratio = len(self.memories) / self.max_items
        
        return size_ratio > self.cleanup_threshold or item_ratio > self.cleanup_threshold
    
    def _intelligent_cleanup(self) -> int:
        """Perform intelligent memory cleanup based on usage patterns"""
        
        if not self.smart_eviction:
            return self._simple_lru_cleanup()
        
        removed_count = 0
        target_removal = max(100, len(self.memories) // 10)  # Remove at least 10%
        
        # Score memories for removal (lower score = higher priority for removal)
        memory_scores = []
        
        for memory_id, memory in self.memories.items():
            if memory.priority == PriorityLevel.CRITICAL:
                continue  # Never remove critical memories
            
            score = self._calculate_eviction_score(memory)
            memory_scores.append((score, memory_id))
        
        # Sort by score (lowest first) and remove
        memory_scores.sort(key=lambda x: x[0])
        
        for score, memory_id in memory_scores[:target_removal]:
            if self.delete_memory(memory_id):
                removed_count += 1
        
        self.last_cleanup = datetime.now()
        return removed_count
    
    def _simple_lru_cleanup(self) -> int:
        """Simple LRU-based cleanup"""
        removed_count = 0
        target_removal = len(self.memories) // 10  # Remove 10%
        
        # Sort by last access time
        sorted_memories = sorted(
            self.memories.items(),
            key=lambda x: x[1].last_accessed
        )
        
        for memory_id, memory in sorted_memories[:target_removal]:
            if memory.priority != PriorityLevel.CRITICAL:
                if self.delete_memory(memory_id):
                    removed_count += 1
        
        return removed_count
    
    def _calculate_eviction_score(self, memory: MemoryItem) -> float:
        """Calculate eviction score (lower = more likely to be removed)"""
        
        # Base score from priority
        priority_scores = {
            PriorityLevel.CRITICAL: 1000,
            PriorityLevel.HIGH: 100,
            PriorityLevel.MEDIUM: 50,
            PriorityLevel.LOW: 20,
            PriorityLevel.DISPOSABLE: 1
        }
        
        score = priority_scores.get(memory.priority, 50)
        
        # Adjust for access frequency
        score += memory.access_count * 5
        
        # Adjust for recency (more recent = higher score)
        hours_since_access = (datetime.now() - memory.last_accessed).total_seconds() / 3600
        score *= max(0.1, 1 - (hours_since_access / 168))  # Decay over a week
        
        # Adjust for relationships (more connected = higher score)
        relationship_bonus = len(memory.related_items) + len(memory.child_ids)
        score += relationship_bonus * 2
        
        # Adjust for compression (compressed items are cheaper to keep)
        if memory.is_compressed:
            score += 10
        
        return score
    
    def _calculate_relevance_score(self, 
                                 memory: MemoryItem, 
                                 query: str,
                                 context_tags: Optional[List[str]] = None) -> float:
        """Calculate relevance score for search"""
        
        score = 0.0
        query_lower = query.lower()
        
        # Content matching
        content_str = json.dumps(memory.content, default=str).lower()
        
        # Simple keyword matching
        query_words = query_lower.split()
        for word in query_words:
            if word in content_str:
                score += 10
        
        # Context tag matching
        if context_tags:
            matching_tags = set(context_tags) & set(memory.context_tags)
            score += len(matching_tags) * 20
        
        # Boost for recent access
        hours_since_access = (datetime.now() - memory.last_accessed).total_seconds() / 3600
        recency_boost = max(0, 50 - hours_since_access)
        score += recency_boost
        
        # Boost for high priority
        priority_boost = {
            PriorityLevel.CRITICAL: 100,
            PriorityLevel.HIGH: 50,
            PriorityLevel.MEDIUM: 20,
            PriorityLevel.LOW: 10,
            PriorityLevel.DISPOSABLE: 0
        }
        score += priority_boost.get(memory.priority, 0)
        
        return score
    
    def _compress_memory_item(self, memory: MemoryItem) -> MemoryItem:
        """Compress a memory item"""
        
        if memory.is_compressed:
            return memory
        
        try:
            original_size = memory.size_bytes
            
            # Choose compression strategy based on content type and size
            if original_size > 10000:  # Large items get more aggressive compression
                compression_type = CompressionType.HIERARCHICAL
            elif memory.memory_type == MemoryType.SEMANTIC:
                compression_type = CompressionType.SEMANTIC
            else:
                compression_type = CompressionType.SUMMARY
            
            # Apply compression
            compressed_content = self._apply_compression(memory.content, compression_type)
            
            # Update memory item
            memory.original_size = original_size
            memory.content = compressed_content
            memory.is_compressed = True
            memory.compression_type = compression_type
            memory.size_bytes = len(json.dumps(compressed_content, default=str).encode('utf-8'))
            
            compression_ratio = 1 - (memory.size_bytes / original_size)
            logger.debug(f"🗜️ Compressed memory {memory.memory_id}: {compression_ratio:.2%} reduction")
            
        except Exception as e:
            logger.error(f"❌ Failed to compress memory {memory.memory_id}: {e}")
        
        return memory
    
    def _decompress_memory_item(self, memory: MemoryItem) -> MemoryItem:
        """Decompress a memory item for use"""
        
        if not memory.is_compressed:
            return memory
        
        try:
            # In a real implementation, this would reverse the compression
            # For now, we'll assume the compressed content is still usable
            # but we could implement actual decompression logic here
            
            logger.debug(f"📤 Accessed compressed memory: {memory.memory_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to decompress memory {memory.memory_id}: {e}")
        
        return memory
    
    def _apply_compression(self, content: Dict[str, Any], compression_type: CompressionType) -> Dict[str, Any]:
        """Apply specific compression strategy"""
        
        if compression_type == CompressionType.SUMMARY:
            # Summarize text content
            return {
                "summary": self._summarize_content(content),
                "compression_type": "summary",
                "key_fields": list(content.keys())[:10]  # Keep key field names
            }
        
        elif compression_type == CompressionType.KEYWORDS:
            # Extract keywords
            return {
                "keywords": self._extract_keywords(content),
                "compression_type": "keywords",
                "original_structure": self._get_content_structure(content)
            }
        
        elif compression_type == CompressionType.HIERARCHICAL:
            # Hierarchical compression
            return {
                "hierarchy": self._create_content_hierarchy(content),
                "compression_type": "hierarchical",
                "depth_levels": 3
            }
        
        else:
            # Default: keep essential fields only
            essential_fields = ["id", "type", "title", "summary", "key_data"]
            return {k: v for k, v in content.items() if k in essential_fields}
    
    def _summarize_content(self, content: Dict[str, Any]) -> str:
        """Create summary of content"""
        # Simple summarization - in practice would use more sophisticated methods
        text_parts = []
        for key, value in content.items():
            if isinstance(value, str) and len(value) > 20:
                text_parts.append(f"{key}: {value[:100]}...")
        
        return " | ".join(text_parts[:3])
    
    def _extract_keywords(self, content: Dict[str, Any]) -> List[str]:
        """Extract keywords from content"""
        # Simple keyword extraction
        text = json.dumps(content, default=str).lower()
        words = text.split()
        
        # Filter common words and return most frequent
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word for word in words if len(word) > 3 and word not in common_words]
        
        # Return top keywords by frequency
        word_freq = defaultdict(int)
        for word in keywords:
            word_freq[word] += 1
        
        return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:20]
    
    def _get_content_structure(self, content: Dict[str, Any]) -> Dict[str, str]:
        """Get structural information about content"""
        structure = {}
        for key, value in content.items():
            if isinstance(value, dict):
                structure[key] = "object"
            elif isinstance(value, list):
                structure[key] = f"array[{len(value)}]"
            elif isinstance(value, str):
                structure[key] = f"string[{len(value)}]"
            else:
                structure[key] = type(value).__name__
        
        return structure
    
    def _create_content_hierarchy(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create hierarchical representation of content"""
        hierarchy = {
            "level_1": {},  # Most important
            "level_2": {},  # Moderately important
            "level_3": {}   # Least important
        }
        
        important_keys = {"id", "title", "type", "status", "priority"}
        moderate_keys = {"description", "content", "data", "metadata"}
        
        for key, value in content.items():
            if key in important_keys:
                hierarchy["level_1"][key] = value
            elif key in moderate_keys:
                hierarchy["level_2"][key] = value
            else:
                hierarchy["level_3"][key] = value
        
        return hierarchy
    
    def _update_indexes(self, memory: MemoryItem):
        """Update memory indexes for fast searching"""
        
        # Index by memory type
        self.memory_index[f"type:{memory.memory_type}"].append(memory.memory_id)
        
        # Index by priority
        self.memory_index[f"priority:{memory.priority}"].append(memory.memory_id)
        
        # Index by context tags
        for tag in memory.context_tags:
            self.memory_index[f"tag:{tag}"].append(memory.memory_id)
        
        # Index by session
        if memory.session_id:
            self.memory_index[f"session:{memory.session_id}"].append(memory.memory_id)
        
        # Index by user
        if memory.user_id:
            self.memory_index[f"user:{memory.user_id}"].append(memory.memory_id)
    
    def _remove_from_indexes(self, memory: MemoryItem):
        """Remove memory from all indexes"""
        
        indexes_to_clean = [
            f"type:{memory.memory_type}",
            f"priority:{memory.priority}",
            f"session:{memory.session_id}",
            f"user:{memory.user_id}"
        ]
        
        for tag in memory.context_tags:
            indexes_to_clean.append(f"tag:{tag}")
        
        for index_key in indexes_to_clean:
            if index_key in self.memory_index and memory.memory_id in self.memory_index[index_key]:
                self.memory_index[index_key].remove(memory.memory_id)
    
    def _rebuild_indexes(self):
        """Rebuild all memory indexes"""
        self.memory_index.clear()
        
        for memory in self.memories.values():
            self._update_indexes(memory)
        
        logger.debug("📚 Memory indexes rebuilt")

# Utility functions
def create_memory_manager(config: Optional[Dict[str, Any]] = None) -> MemoryManager:
    """Create and configure a memory manager instance"""
    return MemoryManager(config)

def calculate_memory_efficiency(stats: MemoryStats) -> Dict[str, float]:
    """Calculate memory efficiency metrics"""
    return {
        "storage_efficiency": 1 - (stats.total_size_bytes / (stats.total_items * 1024)) if stats.total_items > 0 else 0,
        "compression_effectiveness": stats.compression_ratio,
        "cache_performance": stats.cache_hit_rate,
        "access_speed": 1 / max(0.001, stats.average_access_time)  # Inverse of access time
    }

# Export main classes and functions
__all__ = [
    "MemoryType",
    "PriorityLevel", 
    "CompressionType",
    "MemoryItem",
    "MemoryStats",
    "MemoryManager",
    "create_memory_manager",
    "calculate_memory_efficiency"
]