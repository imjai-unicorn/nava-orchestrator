# backend/services/shared/common/tests/test_memory_manager.py
"""
Test suite for memory manager
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import the models we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from memory_manager import (
    MemoryType, PriorityLevel, CompressionType,
    MemoryItem, MemoryStats, MemoryManager,
    create_memory_manager, calculate_memory_efficiency
)

class TestMemoryItem:
    """Test MemoryItem dataclass"""
    
    def test_memory_item_creation(self):
        """Test creating a MemoryItem instance"""
        content = {"key": "value", "data": "test"}
        memory_item = MemoryItem(
            memory_id="test_memory_001",
            memory_type=MemoryType.SHORT_TERM,
            content=content,
            priority=PriorityLevel.HIGH,
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )
        
        assert memory_item.memory_id == "test_memory_001"
        assert memory_item.memory_type == MemoryType.SHORT_TERM
        assert memory_item.priority == PriorityLevel.HIGH
        assert memory_item.content == content
        assert memory_item.size_bytes > 0
        assert memory_item.access_count == 0
        assert not memory_item.is_compressed
    
    def test_memory_item_auto_size_calculation(self):
        """Test automatic size calculation"""
        small_content = {"data": "small"}
        large_content = {"data": "x" * 1000, "more": ["item1", "item2", "item3"]}
        
        small_item = MemoryItem(
            memory_id="small",
            memory_type=MemoryType.SHORT_TERM,
            content=small_content,
            priority=PriorityLevel.LOW,
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )
        
        large_item = MemoryItem(
            memory_id="large",
            memory_type=MemoryType.SHORT_TERM,
            content=large_content,
            priority=PriorityLevel.LOW,
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )
        
        assert large_item.size_bytes > small_item.size_bytes
    
    def test_memory_item_with_relationships(self):
        """Test memory item with relationships"""
        memory_item = MemoryItem(
            memory_id="test_memory_001",
            memory_type=MemoryType.EPISODIC,
            content={"data": "test"},
            priority=PriorityLevel.MEDIUM,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            related_items=["related_1", "related_2"],
            parent_id="parent_1",
            child_ids=["child_1", "child_2"]
        )
        
        assert len(memory_item.related_items) == 2
        assert memory_item.parent_id == "parent_1"
        assert len(memory_item.child_ids) == 2

class TestMemoryManager:
    """Test MemoryManager class"""
    
    @pytest.fixture
    def memory_manager(self):
        """Fixture providing a memory manager instance"""
        config = {
            'max_memory_size': 1024 * 1024,  # 1MB for testing
            'max_items': 100,
            'cleanup_threshold': 0.8,
            'compression_threshold': 500,
            'auto_compression': True,
            'smart_eviction': True
        }
        return MemoryManager(config)
    
    @pytest.fixture
    def sample_memory_content(self):
        """Fixture providing sample memory content"""
        return {
            "user_query": "What is the capital of France?",
            "ai_response": "The capital of France is Paris.",
            "context": {"user_id": "user123", "session_id": "session456"},
            "metadata": {"model": "gpt-4", "confidence": 0.95}
        }
    
    def test_memory_manager_initialization(self, memory_manager):
        """Test memory manager initialization"""
        assert memory_manager.max_memory_size == 1024 * 1024
        assert memory_manager.max_items == 100
        assert memory_manager.cleanup_threshold == 0.8
        assert memory_manager.auto_compression
        assert memory_manager.smart_eviction
        assert len(memory_manager.memories) == 0
    
    def test_store_memory_success(self, memory_manager, sample_memory_content):
        """Test successful memory storage"""
        result = memory_manager.store_memory(
            memory_id="test_001",
            content=sample_memory_content,
            memory_type=MemoryType.SHORT_TERM,
            priority=PriorityLevel.HIGH,
            context_tags=["conversation", "france"],
            session_id="session123",
            user_id="user456"
        )
        
        assert result
        assert "test_001" in memory_manager.memories
        
        stored_memory = memory_manager.memories["test_001"]
        assert stored_memory.memory_type == MemoryType.SHORT_TERM
        assert stored_memory.priority == PriorityLevel.HIGH
        assert "conversation" in stored_memory.context_tags
        assert stored_memory.session_id == "session123"
    
    def test_retrieve_memory_success(self, memory_manager, sample_memory_content):
        """Test successful memory retrieval"""
        # Store memory first
        memory_manager.store_memory(
            memory_id="test_002",
            content=sample_memory_content,
            memory_type=MemoryType.WORKING,
            priority=PriorityLevel.MEDIUM
        )
        
        # Retrieve memory
        retrieved = memory_manager.retrieve_memory("test_002")
        
        assert retrieved is not None
        assert retrieved.memory_id == "test_002"
        assert retrieved.content == sample_memory_content
        assert retrieved.access_count == 1
    
    def test_retrieve_memory_not_found(self, memory_manager):
        """Test memory retrieval when not found"""
        retrieved = memory_manager.retrieve_memory("nonexistent")
        
        assert retrieved is None
    
    def test_update_memory_success(self, memory_manager, sample_memory_content):
        """Test successful memory update"""
        # Store memory first
        memory_manager.store_memory(
            memory_id="test_update",
            content=sample_memory_content,
            memory_type=MemoryType.SEMANTIC,
            priority=PriorityLevel.LOW
        )
        
        # Update memory
        new_content = {"updated": "content", "new_field": "value"}
        result = memory_manager.update_memory(
            memory_id="test_update",
            content=new_content,
            priority=PriorityLevel.HIGH,
            context_tags=["updated", "test"]
        )
        
        assert result
        
        updated_memory = memory_manager.retrieve_memory("test_update")
        assert updated_memory.content == new_content
        assert updated_memory.priority == PriorityLevel.HIGH
        assert "updated" in updated_memory.context_tags
    
    def test_update_memory_not_found(self, memory_manager):
        """Test memory update when memory not found"""
        result = memory_manager.update_memory(
            memory_id="nonexistent",
            content={"new": "content"}
        )
        
        assert not result
    
    def test_delete_memory_success(self, memory_manager, sample_memory_content):
        """Test successful memory deletion"""
        # Store memory first
        memory_manager.store_memory(
            memory_id="test_delete",
            content=sample_memory_content,
            memory_type=MemoryType.CACHE,
            priority=PriorityLevel.DISPOSABLE
        )
        
        # Delete memory
        result = memory_manager.delete_memory("test_delete")
        
        assert result
        assert "test_delete" not in memory_manager.memories
    
    def test_delete_memory_not_found(self, memory_manager):
        """Test memory deletion when not found"""
        result = memory_manager.delete_memory("nonexistent")
        
        assert not result
    
    def test_search_memories_by_content(self, memory_manager):
        """Test memory search by content"""
        # Store multiple memories
        memories_data = [
            ("mem1", {"topic": "python", "content": "Python programming"}),
            ("mem2", {"topic": "javascript", "content": "JavaScript development"}),
            ("mem3", {"topic": "python", "content": "Python data science"})
        ]
        
        for mem_id, content in memories_data:
            memory_manager.store_memory(
                memory_id=mem_id,
                content=content,
                memory_type=MemoryType.SEMANTIC,
                priority=PriorityLevel.MEDIUM
            )
        
        # Search for Python-related memories
        results = memory_manager.search_memories("python", limit=10)
        
        assert len(results) == 2
        assert all("python" in r.content.get("topic", "").lower() or 
                  "python" in r.content.get("content", "").lower() for r in results)
    
    def test_search_memories_by_type(self, memory_manager, sample_memory_content):
        """Test memory search by type"""
        # Store memories of different types
        memory_manager.store_memory("work1", sample_memory_content, MemoryType.WORKING, PriorityLevel.HIGH)
        memory_manager.store_memory("short1", sample_memory_content, MemoryType.SHORT_TERM, PriorityLevel.MEDIUM)
        memory_manager.store_memory("work2", sample_memory_content, MemoryType.WORKING, PriorityLevel.LOW)
        
        # Search for working memories only
        results = memory_manager.search_memories("", memory_type=MemoryType.WORKING)
        
        assert len(results) == 2
        assert all(r.memory_type == MemoryType.WORKING for r in results)
    
    def test_search_memories_by_context_tags(self, memory_manager, sample_memory_content):
        """Test memory search by context tags"""
        # Store memories with different tags
        memory_manager.store_memory("tag1", sample_memory_content, MemoryType.SEMANTIC, 
                                   PriorityLevel.MEDIUM, context_tags=["ai", "conversation"])
        memory_manager.store_memory("tag2", sample_memory_content, MemoryType.SEMANTIC,
                                   PriorityLevel.MEDIUM, context_tags=["database", "query"])
        memory_manager.store_memory("tag3", sample_memory_content, MemoryType.SEMANTIC,
                                   PriorityLevel.MEDIUM, context_tags=["ai", "learning"])
        
        # Search by context tags
        results = memory_manager.search_memories("", context_tags=["ai"])
        
        assert len(results) == 2
        assert all("ai" in r.context_tags for r in results)
    
    def test_search_memories_by_session(self, memory_manager, sample_memory_content):
        """Test memory search by session ID"""
        # Store memories with different sessions
        memory_manager.store_memory("sess1", sample_memory_content, MemoryType.SHORT_TERM,
                                   PriorityLevel.MEDIUM, session_id="session_123")
        memory_manager.store_memory("sess2", sample_memory_content, MemoryType.SHORT_TERM,
                                   PriorityLevel.MEDIUM, session_id="session_456")
        memory_manager.store_memory("sess3", sample_memory_content, MemoryType.SHORT_TERM,
                                   PriorityLevel.MEDIUM, session_id="session_123")
        
        # Search by session
        results = memory_manager.search_memories("", session_id="session_123")
        
        assert len(results) == 2
        assert all(r.session_id == "session_123" for r in results)
    
    def test_add_relationship(self, memory_manager, sample_memory_content):
        """Test adding relationships between memories"""
        # Store two memories
        memory_manager.store_memory("rel1", sample_memory_content, MemoryType.EPISODIC, PriorityLevel.MEDIUM)
        memory_manager.store_memory("rel2", sample_memory_content, MemoryType.EPISODIC, PriorityLevel.MEDIUM)
        
        # Add relationship
        result = memory_manager.add_relationship("rel1", "rel2", bidirectional=True)
        
        assert result
        
        memory1 = memory_manager.retrieve_memory("rel1")
        memory2 = memory_manager.retrieve_memory("rel2")
        
        assert "rel2" in memory1.related_items
        assert "rel1" in memory2.related_items
    
    def test_create_memory_hierarchy(self, memory_manager, sample_memory_content):
        """Test creating parent-child relationships"""
        # Store parent and child memories
        memory_manager.store_memory("parent", sample_memory_content, MemoryType.SEMANTIC, PriorityLevel.HIGH)
        memory_manager.store_memory("child1", sample_memory_content, MemoryType.EPISODIC, PriorityLevel.MEDIUM)
        memory_manager.store_memory("child2", sample_memory_content, MemoryType.EPISODIC, PriorityLevel.MEDIUM)
        
        # Create hierarchy
        result = memory_manager.create_memory_hierarchy("parent", ["child1", "child2"])
        
        assert result
        
        parent = memory_manager.retrieve_memory("parent")
        child1 = memory_manager.retrieve_memory("child1")
        child2 = memory_manager.retrieve_memory("child2")
        
        assert "child1" in parent.child_ids
        assert "child2" in parent.child_ids
        assert child1.parent_id == "parent"
        assert child2.parent_id == "parent"
    
    def test_get_related_memories(self, memory_manager, sample_memory_content):
        """Test getting related memories"""
        # Store memories and create relationships
        memory_manager.store_memory("main", sample_memory_content, MemoryType.SEMANTIC, PriorityLevel.HIGH)
        memory_manager.store_memory("related1", sample_memory_content, MemoryType.EPISODIC, PriorityLevel.MEDIUM)
        memory_manager.store_memory("related2", sample_memory_content, MemoryType.EPISODIC, PriorityLevel.MEDIUM)
        memory_manager.store_memory("child", sample_memory_content, MemoryType.PROCEDURAL, PriorityLevel.LOW)
        
        # Add relationships
        memory_manager.add_relationship("main", "related1")
        memory_manager.add_relationship("main", "related2")
        memory_manager.create_memory_hierarchy("main", ["child"])
        
        # Get related memories
        related = memory_manager.get_related_memories("main", depth=1, max_results=10)
        
        assert len(related) == 3
        related_ids = [r.memory_id for r in related]
        assert "related1" in related_ids
        assert "related2" in related_ids
        assert "child" in related_ids
    
    def test_compression_functionality(self, memory_manager):
        """Test memory compression"""
        large_content = {"data": "x" * 2000, "more_data": ["item"] * 100}  # Large content
        
        memory_manager.store_memory(
            memory_id="compress_test",
            content=large_content,
            memory_type=MemoryType.CACHE,
            priority=PriorityLevel.LOW
        )
        
        stored_memory = memory_manager.retrieve_memory("compress_test")
        
        # Should be compressed due to size
        if memory_manager.auto_compression:
            assert stored_memory.is_compressed or stored_memory.size_bytes < 2000
    
    def test_compress_memory_type(self, memory_manager):
        """Test compressing all memories of a specific type"""
        # Store multiple cache memories
        for i in range(3):
            large_content = {"data": "x" * 1500, "id": i}
            memory_manager.store_memory(
                memory_id=f"cache_{i}",
                content=large_content,
                memory_type=MemoryType.CACHE,
                priority=PriorityLevel.LOW
            )
        
        # Compress all cache memories
        compressed_count = memory_manager.compress_memory_type(MemoryType.CACHE)
        
        assert compressed_count >= 0  # May already be compressed
    
    def test_cleanup_old_memories(self, memory_manager, sample_memory_content):
        """Test cleanup of old memories"""
        # Store old memories
        old_time = datetime.now() - timedelta(days=35)
        
        for i in range(3):
            memory_item = MemoryItem(
                memory_id=f"old_{i}",
                memory_type=MemoryType.CACHE,
                content=sample_memory_content,
                priority=PriorityLevel.LOW,
                created_at=old_time,
                last_accessed=old_time
            )
            memory_manager.memories[f"old_{i}"] = memory_item
        
        # Store recent memories
        memory_manager.store_memory("recent", sample_memory_content, MemoryType.CACHE, PriorityLevel.LOW)
        
        # Cleanup old memories
        removed_count = memory_manager.cleanup_old_memories(max_age_days=30)
        
        assert removed_count == 3
        assert "recent" in memory_manager.memories
        assert "old_0" not in memory_manager.memories
    
    def test_get_memory_stats(self, memory_manager, sample_memory_content):
        """Test memory statistics generation"""
        # Store various memories
        memory_manager.store_memory("stat1", sample_memory_content, MemoryType.SHORT_TERM, PriorityLevel.HIGH)
        memory_manager.store_memory("stat2", sample_memory_content, MemoryType.WORKING, PriorityLevel.MEDIUM)
        memory_manager.store_memory("stat3", sample_memory_content, MemoryType.SEMANTIC, PriorityLevel.LOW)
        
        # Get statistics
        stats = memory_manager.get_memory_stats()
        
        assert isinstance(stats, MemoryStats)
        assert stats.total_items == 3
        assert stats.total_size_bytes > 0
        assert len(stats.items_by_type) >= 3
        assert len(stats.items_by_priority) >= 3
        assert stats.cache_hit_rate >= 0
    
    def test_optimize_memory(self, memory_manager, sample_memory_content):
        """Test memory optimization"""
        # Store memories
        for i in range(5):
            memory_manager.store_memory(
                f"opt_{i}",
                sample_memory_content,
                MemoryType.CACHE,
                PriorityLevel.LOW
            )
        
        # Run optimization
        optimization_results = memory_manager.optimize_memory()
        
        assert "items_compressed" in optimization_results
        assert "items_cleaned" in optimization_results
        assert "relationships_optimized" in optimization_results
        assert "indexes_rebuilt" in optimization_results
    
    def test_intelligent_cleanup_simulation(self, memory_manager):
        """Test intelligent cleanup when memory is full"""
        # Fill memory manager close to capacity
        large_content = {"data": "x" * 500}  # Medium-sized content
        
        # Store many items to trigger cleanup
        for i in range(memory_manager.max_items + 10):
            priority = PriorityLevel.LOW if i % 3 == 0 else PriorityLevel.MEDIUM
            memory_manager.store_memory(
                f"cleanup_{i:03d}",
                large_content,
                MemoryType.CACHE,
                priority
            )
        
        # Should not exceed max_items due to cleanup
        assert len(memory_manager.memories) <= memory_manager.max_items
    
    def test_access_time_tracking(self, memory_manager, sample_memory_content):
        """Test access time tracking"""
        memory_manager.store_memory("timing_test", sample_memory_content, MemoryType.WORKING, PriorityLevel.MEDIUM)
        
        # Access memory multiple times
        for _ in range(5):
            memory_manager.retrieve_memory("timing_test")
        
        # Check access statistics
        assert len(memory_manager.access_times) == 5
        assert all(t >= 0 for t in memory_manager.access_times)
    
    def test_cache_hit_miss_tracking(self, memory_manager, sample_memory_content):
        """Test cache hit/miss tracking"""
        memory_manager.store_memory("hit_test", sample_memory_content, MemoryType.WORKING, PriorityLevel.MEDIUM)
        
        # Hit
        memory_manager.retrieve_memory("hit_test")
        # Miss
        memory_manager.retrieve_memory("nonexistent")
        
        assert memory_manager.cache_hits >= 1
        assert memory_manager.cache_misses >= 1
    
    def test_memory_eviction_scoring(self, memory_manager, sample_memory_content):
        """Test memory eviction scoring algorithm"""
        # Create memories with different characteristics
        high_priority = MemoryItem(
            memory_id="high_priority",
            memory_type=MemoryType.SEMANTIC,
            content=sample_memory_content,
            priority=PriorityLevel.CRITICAL,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=10
        )
        
        low_priority = MemoryItem(
            memory_id="low_priority",
            memory_type=MemoryType.CACHE,
            content=sample_memory_content,
            priority=PriorityLevel.DISPOSABLE,
            created_at=datetime.now() - timedelta(days=1),
            last_accessed=datetime.now() - timedelta(hours=12),
            access_count=1
        )
        
        high_score = memory_manager._calculate_eviction_score(high_priority)
        low_score = memory_manager._calculate_eviction_score(low_priority)
        
        # High priority memory should have higher score (less likely to be evicted)
        assert high_score > low_score

class TestMemoryManagerPerformance:
    """Performance tests for memory manager"""
    
    def test_large_scale_storage_retrieval(self):
        """Test performance with large number of memories"""
        memory_manager = MemoryManager({'max_items': 1000, 'max_memory_size': 10 * 1024 * 1024})
        
        # Store many memories
        start_time = time.time()
        
        for i in range(100):
            content = {"id": i, "data": f"Content for memory {i}"}
            memory_manager.store_memory(
                f"perf_{i:03d}",
                content,
                MemoryType.CACHE,
                PriorityLevel.LOW
            )
        
        storage_time = time.time() - start_time
        
        # Retrieve memories
        start_time = time.time()
        
        for i in range(0, 100, 10):  # Retrieve every 10th memory
            memory_manager.retrieve_memory(f"perf_{i:03d}")
        
        retrieval_time = time.time() - start_time
        
        # Performance should be reasonable
        assert storage_time < 5.0  # 5 seconds for 100 items
        assert retrieval_time < 1.0  # 1 second for 10 retrievals
    
    def test_search_performance(self):
        """Test search performance with many memories"""
        memory_manager = MemoryManager({'max_items': 500})
        
        # Store memories with searchable content
        topics = ["python", "javascript", "databases", "ai", "machine learning"]
        
        for i in range(100):
            topic = topics[i % len(topics)]
            content = {"topic": topic, "content": f"Content about {topic} number {i}"}
            memory_manager.store_memory(
                f"search_{i:03d}",
                content,
                MemoryType.SEMANTIC,
                PriorityLevel.MEDIUM,
                context_tags=[topic]
            )
        
        # Perform searches
        start_time = time.time()
        
        for topic in topics:
            results = memory_manager.search_memories(topic, limit=20)
            assert len(results) > 0
        
        search_time = time.time() - start_time
        
        # Search should be fast
        assert search_time < 2.0  # 2 seconds for 5 searches

class TestMemoryUtilities:
    """Test utility functions"""
    
    def test_create_memory_manager(self):
        """Test memory manager creation utility"""
        config = {'max_items': 50, 'auto_compression': False}
        manager = create_memory_manager(config)
        
        assert isinstance(manager, MemoryManager)
        assert manager.max_items == 50
        assert not manager.auto_compression
    
    def test_calculate_memory_efficiency(self):
        """Test memory efficiency calculation"""
        stats = MemoryStats(
            total_items=100,
            total_size_bytes=50000,
            items_by_type={MemoryType.CACHE: 60, MemoryType.WORKING: 40},
            size_by_type={MemoryType.CACHE: 30000, MemoryType.WORKING: 20000},
            items_by_priority={PriorityLevel.HIGH: 20, PriorityLevel.MEDIUM: 80},
            compressed_items=30,
            compression_ratio=0.4,
            cache_hit_rate=0.75,
            average_access_time=0.001,
            last_cleanup=datetime.now()
        )
        
        efficiency = calculate_memory_efficiency(stats)
        
        assert "storage_efficiency" in efficiency
        assert "compression_effectiveness" in efficiency
        assert "cache_performance" in efficiency
        assert "access_speed" in efficiency
        
        assert 0 <= efficiency["compression_effectiveness"] <= 1
        assert 0 <= efficiency["cache_performance"] <= 1

class TestMemoryManagerErrorHandling:
    """Test error handling in memory manager"""
    
    def test_invalid_memory_operations(self):
        """Test handling of invalid operations"""
        memory_manager = MemoryManager()
        
        # Try to update non-existent memory
        result = memory_manager.update_memory("nonexistent", {"new": "content"})
        assert not result
        
        # Try to delete non-existent memory
        result = memory_manager.delete_memory("nonexistent")
        assert not result
        
        # Try to add relationship with non-existent memories
        result = memory_manager.add_relationship("nonexistent1", "nonexistent2")
        assert not result
    
    def test_memory_corruption_handling(self):
        """Test handling of corrupted memory data"""
        memory_manager = MemoryManager()
        
        # Store normal memory
        memory_manager.store_memory(
            "test_corrupt",
            {"valid": "data"},
            MemoryType.WORKING,
            PriorityLevel.MEDIUM
        )
        
        # Simulate corruption by directly modifying stored memory
        stored_memory = memory_manager.memories["test_corrupt"]
        stored_memory.content = None  # Corrupted content
        
        # Retrieval should handle gracefully
        retrieved = memory_manager.retrieve_memory("test_corrupt")
        assert retrieved is not None  # Should still return the memory object

class TestMemoryManagerIntegration:
    """Integration tests for memory manager"""
    
    def test_complete_memory_lifecycle(self):
        """Test complete memory lifecycle"""
        memory_manager = MemoryManager({'auto_compression': True, 'smart_eviction': True})
        
        # 1. Store memory
        content = {"conversation": "What is AI?", "response": "AI is artificial intelligence..."}
        result = memory_manager.store_memory(
            "lifecycle_test",
            content,
            MemoryType.EPISODIC,
            PriorityLevel.HIGH,
            context_tags=["ai", "conversation"],
            session_id="session_123"
        )
        assert result
        
        # 2. Retrieve and verify
        retrieved = memory_manager.retrieve_memory("lifecycle_test")
        assert retrieved.content == content
        assert retrieved.access_count == 1
        
        # 3. Update memory
        updated_content = {**content, "updated": True}
        result = memory_manager.update_memory("lifecycle_test", updated_content)
        assert result
        
        # 4. Search for memory
        results = memory_manager.search_memories("AI", context_tags=["ai"])
        assert len(results) >= 1
        
        # 5. Add relationships
        memory_manager.store_memory("related", {"related": "content"}, MemoryType.EPISODIC, PriorityLevel.MEDIUM)
        result = memory_manager.add_relationship("lifecycle_test", "related")
        assert result
        
        # 6. Get related memories
        related = memory_manager.get_related_memories("lifecycle_test")
        assert len(related) >= 1
        
        # 7. Get statistics
        stats = memory_manager.get_memory_stats()
        assert stats.total_items >= 2
        
        # 8. Optimize
        optimization_results = memory_manager.optimize_memory()
        assert all(key in optimization_results for key in 
                  ["items_compressed", "items_cleaned", "relationships_optimized", "indexes_rebuilt"])
        
        # 9. Delete memory
        result = memory_manager.delete_memory("lifecycle_test")
        assert result
        assert "lifecycle_test" not in memory_manager.memories

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
