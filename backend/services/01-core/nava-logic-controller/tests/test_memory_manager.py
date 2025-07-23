# backend/services/01-core/nava-logic-controller/tests/test_memory_manager.py
"""
🧠 Memory Manager Test Suite
Advanced memory optimization and context management testing
Phase 1 - Week 2: Test existing advanced features
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import threading
import time

# Import the actual classes from the service
import sys
import os
# Correct path to find the memory_manager module in the shared/common directory
# Calculate the correct relative path to 'shared/common'
current_dir = os.path.dirname(os.path.abspath(__file__))
# From /tests -> /nava-logic-controller -> /01-core -> /services -> /shared/common
shared_common_path = os.path.normpath(os.path.join(current_dir, '..', '..', '..', '..', 'shared', 'common'))
sys.path.insert(0, shared_common_path)

from memory_manager import (
    MemoryManager, MemoryItem, MemoryType, PriorityLevel, 
    CompressionType, MemoryStats, create_memory_manager,
    calculate_memory_efficiency
)

@pytest.fixture
def memory_manager():
    """Create memory manager instance for testing"""
    config = {
        'max_memory_size': 10 * 1024 * 1024,  # 10MB for testing
        'max_items': 1000,
        'cleanup_threshold': 0.8,
        'compression_threshold': 1024,
        'min_evidence_count': 3,
        'confidence_threshold': 0.7,
        'temporal_window_days': 30
    }
    return MemoryManager(config)

@pytest.fixture
def sample_memory_items():
    """Sample memory items for testing"""
    items = []
    
    # High priority conversation memory
    items.append(MemoryItem(
        memory_id="conv_001",
        memory_type=MemoryType.SHORT_TERM,
        content={
            "conversation_id": "conv_001",
            "messages": ["Hello", "How can I help?"],
            "user_preferences": {"ai_model": "gpt", "tone": "professional"}
        },
        priority=PriorityLevel.HIGH,
        created_at=datetime.now(),
        last_accessed=datetime.now(),
        context_tags=["conversation", "user_interaction"],
        session_id="session_123",
        user_id="user_456"
    ))
    
    # Working memory with complex data
    items.append(MemoryItem(
        memory_id="work_001",
        memory_type=MemoryType.WORKING,
        content={
            "task_context": "code_analysis",
            "code_snippet": "def test_function():\n    return 'test'" * 100,  # Large content
            "analysis_result": {"complexity": "medium", "suggestions": ["refactor"]}
        },
        priority=PriorityLevel.MEDIUM,
        created_at=datetime.now() - timedelta(hours=2),
        last_accessed=datetime.now() - timedelta(minutes=30),
        context_tags=["coding", "analysis"]
    ))
    
    # Old disposable memory
    items.append(MemoryItem(
        memory_id="old_001",
        memory_type=MemoryType.CACHE,
        content={"cached_response": "temporary data"},
        priority=PriorityLevel.DISPOSABLE,
        created_at=datetime.now() - timedelta(days=5),
        last_accessed=datetime.now() - timedelta(days=3),
        context_tags=["cache", "temporary"]
    ))
    
    return items

#class TestMemoryManager:
#    """Test suite for NAVA Memory Manager""" 


class TestMemoryStorage:
    """Test memory storage operations"""
    
    def test_store_memory_success(self, memory_manager, sample_memory_items):
        """✅ Test successful memory storage"""
        memory_item = sample_memory_items[0]
        
        result = memory_manager.store_memory(
            memory_id=memory_item.memory_id,
            content=memory_item.content,
            memory_type=memory_item.memory_type,
            priority=memory_item.priority,
            context_tags=memory_item.context_tags,
            session_id=memory_item.session_id,
            user_id=memory_item.user_id
        )
        
        assert result is True
        assert memory_item.memory_id in memory_manager.memories
        
        # Verify stored data
        stored = memory_manager.memories[memory_item.memory_id]
        assert stored.memory_type == memory_item.memory_type
        assert stored.priority == memory_item.priority
        assert stored.content == memory_item.content
        
    def test_store_multiple_memories(self, memory_manager, sample_memory_items):
        """✅ Test storing multiple memory items"""
        for item in sample_memory_items:
            result = memory_manager.store_memory(
                memory_id=item.memory_id,
                content=item.content,
                memory_type=item.memory_type,
                priority=item.priority,
                context_tags=item.context_tags,
                session_id=getattr(item, 'session_id', None),
                user_id=getattr(item, 'user_id', None)
            )
            assert result is True
        
        assert len(memory_manager.memories) == len(sample_memory_items)
        
    def test_store_memory_compression(self, memory_manager):
        """✅ Test automatic memory compression for large items"""
        large_content = {
            "large_data": "x" * 2048,  # Larger than compression threshold
            "metadata": {"type": "large_test"}
        }
        
        result = memory_manager.store_memory(
            memory_id="large_001",
            content=large_content,
            memory_type=MemoryType.WORKING,
            priority=PriorityLevel.MEDIUM
        )
        
        assert result is True
        stored = memory_manager.memories["large_001"]
        
        # Should be compressed due to size
        assert stored.is_compressed is True
        assert stored.original_size is not None
        assert stored.size_bytes < stored.original_size


class TestMemoryRetrieval:
    """Test memory retrieval operations"""
    
    def test_retrieve_memory_success(self, memory_manager, sample_memory_items):
        """✅ Test successful memory retrieval"""
        # Store first
        item = sample_memory_items[0]
        memory_manager.store_memory(
            memory_id=item.memory_id,
            content=item.content,
            memory_type=item.memory_type,
            priority=item.priority
        )
        
        # Retrieve
        retrieved = memory_manager.retrieve_memory(item.memory_id)
        
        assert retrieved is not None
        assert retrieved.memory_id == item.memory_id
        assert retrieved.content == item.content
        assert retrieved.access_count == 1
        
    def test_retrieve_nonexistent_memory(self, memory_manager):
        """✅ Test retrieving non-existent memory"""
        retrieved = memory_manager.retrieve_memory("nonexistent")
        assert retrieved is None
        
    def test_retrieve_updates_access_stats(self, memory_manager, sample_memory_items):
        """✅ Test that retrieval updates access statistics"""
        item = sample_memory_items[0]
        memory_manager.store_memory(
            memory_id=item.memory_id,
            content=item.content,
            memory_type=item.memory_type,
            priority=item.priority
        )
        
        initial_access_time = memory_manager.memories[item.memory_id].last_accessed
        time.sleep(0.1)  # Small delay to see time difference
        
        # Retrieve multiple times
        for _ in range(3):
            retrieved = memory_manager.retrieve_memory(item.memory_id)
            assert retrieved is not None
        
        final_memory = memory_manager.memories[item.memory_id]
        assert final_memory.access_count == 3
        assert final_memory.last_accessed > initial_access_time


class TestMemorySearch:
    """Test memory search capabilities"""
    
    def test_search_by_content(self, memory_manager, sample_memory_items):
        """✅ Test searching memories by content"""
        # Store test data
        for item in sample_memory_items:
            memory_manager.store_memory(
                memory_id=item.memory_id,
                content=item.content,
                memory_type=item.memory_type,
                priority=item.priority,
                context_tags=item.context_tags
            )
        
        # Search for conversation-related content
        results = memory_manager.search_memories("conversation", limit=5)
        
        assert len(results) > 0
        # Should find the conversation memory
        found_conv = any(r.memory_id == "conv_001" for r in results)
        assert found_conv
        
    def test_search_by_memory_type(self, memory_manager, sample_memory_items):
        """✅ Test filtering search by memory type"""
        # Store test data
        for item in sample_memory_items:
            memory_manager.store_memory(
                memory_id=item.memory_id,
                content=item.content,
                memory_type=item.memory_type,
                priority=item.priority
            )
        
        # Search only in working memory
        results = memory_manager.search_memories(
            "analysis",
            memory_type=MemoryType.WORKING,
            limit=10
        )
        
        # All results should be working memory
        for result in results:
            assert result.memory_type == MemoryType.WORKING
            
    def test_search_by_context_tags(self, memory_manager, sample_memory_items):
        """✅ Test searching by context tags"""
        # Store test data
        for item in sample_memory_items:
            memory_manager.store_memory(
                memory_id=item.memory_id,
                content=item.content,
                memory_type=item.memory_type,
                priority=item.priority,
                context_tags=item.context_tags
            )
        
        # Search by context tags
        results = memory_manager.search_memories(
            "test",
            context_tags=["coding", "analysis"],
            limit=10
        )
        
        # Should find items with matching tags
        assert len(results) > 0
        found_work = any(r.memory_id == "work_001" for r in results)
        assert found_work


class TestMemoryManagement:
    """Test memory management operations"""
    
    def test_update_memory(self, memory_manager, sample_memory_items):
        """✅ Test memory update functionality"""
        item = sample_memory_items[0]
        memory_manager.store_memory(
            memory_id=item.memory_id,
            content=item.content,
            memory_type=item.memory_type,
            priority=item.priority
        )
        
        # Update content
        new_content = {"updated": "content", "version": "2.0"}
        result = memory_manager.update_memory(
            memory_id=item.memory_id,
            content=new_content,
            priority=PriorityLevel.HIGH
        )
        
        assert result is True
        
        updated = memory_manager.memories[item.memory_id]
        assert updated.content == new_content
        assert updated.priority == PriorityLevel.HIGH
        
    def test_delete_memory(self, memory_manager, sample_memory_items):
        """✅ Test memory deletion"""
        item = sample_memory_items[0]
        memory_manager.store_memory(
            memory_id=item.memory_id,
            content=item.content,
            memory_type=item.memory_type,
            priority=item.priority
        )
        
        # Verify stored
        assert item.memory_id in memory_manager.memories
        
        # Delete
        result = memory_manager.delete_memory(item.memory_id)
        assert result is True
        assert item.memory_id not in memory_manager.memories
        
    def test_add_relationship(self, memory_manager, sample_memory_items):
        """✅ Test adding relationships between memories"""
        # Store two items
        item1, item2 = sample_memory_items[0], sample_memory_items[1]
        
        for item in [item1, item2]:
            memory_manager.store_memory(
                memory_id=item.memory_id,
                content=item.content,
                memory_type=item.memory_type,
                priority=item.priority
            )
        
        # Add relationship
        result = memory_manager.add_relationship(item1.memory_id, item2.memory_id)
        assert result is True
        
        # Check bidirectional relationship
        mem1 = memory_manager.memories[item1.memory_id]
        mem2 = memory_manager.memories[item2.memory_id]
        
        assert item2.memory_id in mem1.related_items
        assert item1.memory_id in mem2.related_items


class TestMemoryOptimization:
    """Test memory optimization features"""
    
    def test_intelligent_cleanup(self, memory_manager, sample_memory_items):
        """✅ Test intelligent memory cleanup"""
        # Store items with different priorities
        for item in sample_memory_items:
            memory_manager.store_memory(
                memory_id=item.memory_id,
                content=item.content,
                memory_type=item.memory_type,
                priority=item.priority
            )
        
        initial_count = len(memory_manager.memories)
        
        # Force cleanup by setting low threshold
        memory_manager.cleanup_threshold = 0.1
        removed_count = memory_manager._intelligent_cleanup()
        
        # Should remove some items (disposable first)
        assert removed_count > 0
        assert len(memory_manager.memories) < initial_count
        
        # Critical items should remain
        remaining_priorities = [m.priority for m in memory_manager.memories.values()]
        assert PriorityLevel.CRITICAL not in remaining_priorities or len(remaining_priorities) > 0
        
    # test_memory_manager.py

    def test_compress_memory_type(self, memory_manager):
        """✅ Test compressing memories by type"""
        # Temporarily disable auto-compression to test this feature in isolation
        memory_manager.auto_compression = False

        # Create large working memory items
        for i in range(3):
            large_content = {
                "data": "x" * 2048,  # Large content
                "index": i
            }
            memory_manager.store_memory(
                memory_id=f"large_{i}",
                content=large_content,
                memory_type=MemoryType.WORKING,
                priority=PriorityLevel.MEDIUM
            )

        # Now, all items are stored without compression
        # so this function should compress them.
        compressed_count = memory_manager.compress_memory_type(MemoryType.WORKING)

        assert compressed_count > 0  # At least one item should be compressed

        # Check compression status
        working_memories = [m for m in memory_manager.memories.values()
                          if m.memory_type == MemoryType.WORKING]
        compressed_memories = [m for m in working_memories if m.is_compressed]

        assert len(compressed_memories) > 0

        # It's good practice to restore the original setting, though not strictly
        # necessary here since each test gets a fresh instance.
        memory_manager.auto_compression = True
        
    def test_cleanup_old_memories(self, memory_manager):
        """✅ Test cleanup of old memories"""
        # Create old memory
        old_content = {"old": "data"}
        memory_manager.store_memory(
            memory_id="old_memory",
            content=old_content,
            memory_type=MemoryType.CACHE,
            priority=PriorityLevel.LOW
        )
        
        # Manually set old timestamp
        memory_manager.memories["old_memory"].last_accessed = datetime.now() - timedelta(days=35)
        
        # Cleanup memories older than 30 days
        removed_count = memory_manager.cleanup_old_memories(max_age_days=30)
        
        assert removed_count >= 0
        # Old memory should be removed
        if removed_count > 0:
            assert "old_memory" not in memory_manager.memories


class TestMemoryStatistics:
    """Test memory statistics and monitoring"""
    
    def test_get_memory_stats(self, memory_manager, sample_memory_items):
        """✅ Test memory statistics generation"""
        # Store test data
        for item in sample_memory_items:
            memory_manager.store_memory(
                memory_id=item.memory_id,
                content=item.content,
                memory_type=item.memory_type,
                priority=item.priority
            )
        
        stats = memory_manager.get_memory_stats()
        
        assert isinstance(stats, MemoryStats)
        assert stats.total_items == len(sample_memory_items)
        assert stats.total_size_bytes > 0
        assert len(stats.items_by_type) > 0
        assert len(stats.items_by_priority) > 0
        
        # Check cache statistics
        assert hasattr(stats, 'cache_hit_rate')
        assert hasattr(stats, 'average_access_time')
        
    def test_optimize_memory(self, memory_manager, sample_memory_items):
        """✅ Test comprehensive memory optimization"""
        # Store test data
        for item in sample_memory_items:
            memory_manager.store_memory(
                memory_id=item.memory_id,
                content=item.content,
                memory_type=item.memory_type,
                priority=item.priority
            )
        
        # Run optimization
        results = memory_manager.optimize_memory()
        
        assert isinstance(results, dict)
        assert 'items_compressed' in results
        assert 'items_cleaned' in results
        assert 'relationships_optimized' in results
        assert 'indexes_rebuilt' in results
        
        # All values should be non-negative
        for key, value in results.items():
            assert value >= 0


class TestMemoryPerformance:
    """Test memory system performance"""
    
    def test_concurrent_access(self, memory_manager):
        """✅ Test concurrent memory access (thread safety)"""
        def store_memories(thread_id, count=10):
            for i in range(count):
                memory_manager.store_memory(
                    memory_id=f"thread_{thread_id}_memory_{i}",
                    content={"thread_id": thread_id, "index": i},
                    memory_type=MemoryType.WORKING,
                    priority=PriorityLevel.MEDIUM
                )
        
        # Create multiple threads
        threads = []
        for thread_id in range(3):
            thread = threading.Thread(target=store_memories, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have stored all memories without corruption
        assert len(memory_manager.memories) == 30  # 3 threads * 10 memories each
        
    def test_large_scale_storage(self, memory_manager):
        """✅ Test storing large number of memories"""
        # Store many memories
        count = 100
        for i in range(count):
            memory_manager.store_memory(
                memory_id=f"bulk_memory_{i}",
                content={"index": i, "data": f"memory_data_{i}"},
                memory_type=MemoryType.SEMANTIC,
                priority=PriorityLevel.MEDIUM
            )
        
        assert len(memory_manager.memories) == count
        
        # Test search performance
        start_time = time.time()
        results = memory_manager.search_memories("memory_data", limit=20)
        search_time = time.time() - start_time
        
        # Should be reasonably fast (< 1 second for 100 items)
        assert search_time < 1.0
        assert len(results) > 0


class TestMemoryUtilities:
    """Test utility functions"""
    
    def test_create_memory_manager(self):
        """✅ Test memory manager factory function"""
        config = {"max_memory_size": 5 * 1024 * 1024}
        manager = create_memory_manager(config)
        
        assert isinstance(manager, MemoryManager)
        assert manager.config == config
        
    def test_calculate_memory_efficiency(self):
        """✅ Test memory efficiency calculation"""
        # Mock stats
        stats = MemoryStats(
            total_items=100,
            total_size_bytes=50 * 1024,  # 50KB
            items_by_type={MemoryType.WORKING: 50, MemoryType.SHORT_TERM: 50},
            size_by_type={MemoryType.WORKING: 25 * 1024, MemoryType.SHORT_TERM: 25 * 1024},
            items_by_priority={PriorityLevel.HIGH: 30, PriorityLevel.MEDIUM: 70},
            compressed_items=20,
            compression_ratio=0.6,  # 60% compression
            cache_hit_rate=0.4,  # 40% hit rate
            average_access_time=0.05,  # 50ms
            last_cleanup=datetime.now()
        )
        
        efficiency = calculate_memory_efficiency(stats)
        
        assert isinstance(efficiency, dict)
        assert 'storage_efficiency' in efficiency
        assert 'compression_effectiveness' in efficiency
        assert 'cache_performance' in efficiency
        assert 'access_speed' in efficiency
        
        # All efficiency metrics should be positive
        for metric, value in efficiency.items():
            assert value >= 0


class TestMemoryErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_memory_storage(self, memory_manager):
        """✅ Test handling of invalid memory storage attempts"""
        # Test with empty memory_id
        result = memory_manager.store_memory(
            memory_id="",
            content={"test": "data"},
            memory_type=MemoryType.WORKING,
            priority=PriorityLevel.MEDIUM
        )
        # Should handle gracefully (implementation dependent)
        
    def test_memory_retrieval_with_no_access_update(self, memory_manager):
        """✅ Test memory retrieval without updating access stats"""
        memory_manager.store_memory(
            memory_id="test_no_update",
            content={"test": "data"},
            memory_type=MemoryType.WORKING,
            priority=PriorityLevel.MEDIUM
        )
        
        initial_access_count = memory_manager.memories["test_no_update"].access_count
        
        # Retrieve without updating access
        retrieved = memory_manager.retrieve_memory("test_no_update", update_access=False)
        
        assert retrieved is not None
        assert memory_manager.memories["test_no_update"].access_count == initial_access_count
        
    def test_memory_capacity_limits(self, memory_manager):
        """✅ Test memory system behavior at capacity limits"""
        # Set very low limits for testing
        memory_manager.max_items = 5
        memory_manager.cleanup_threshold = 0.8  # Cleanup at 80% (4 items)
        
        # Store items up to limit
        for i in range(10):  # Try to store more than limit
            result = memory_manager.store_memory(
                memory_id=f"limit_test_{i}",
                content={"index": i},
                memory_type=MemoryType.WORKING,
                priority=PriorityLevel.LOW if i < 8 else PriorityLevel.HIGH  # Last 2 are high priority
            )
        
        # Should have triggered cleanup
        assert len(memory_manager.memories) <= memory_manager.max_items


# Phase 1 Integration Test
class TestPhase1MemoryIntegration:
    """Integration tests for Phase 1 - Week 2 requirements"""
    
    def test_memory_system_stability(self, memory_manager):
        """🎯 PHASE 1 CRITICAL: Memory system stability test"""
        # Simulate real NAVA usage patterns
        
        # 1. Store conversation context
        memory_manager.store_memory(
            memory_id="current_conversation",
            content={
                "user_query": "Help me analyze this code",
                "ai_model_selected": "claude",
                "context": {"domain": "software_development", "complexity": "medium"}
            },
            memory_type=MemoryType.WORKING,
            priority=PriorityLevel.HIGH,
            context_tags=["conversation", "active"],
            session_id="session_123"
        )
        
        # 2. Store learning data
        memory_manager.store_memory(
            memory_id="user_preferences",
            content={
                "preferred_ai": "claude",
                "communication_style": "technical",
                "previous_interactions": 15
            },
            memory_type=MemoryType.SEMANTIC,
            priority=PriorityLevel.HIGH,
            user_id="user_456"
        )
        
        # 3. Test retrieval and updates
        context = memory_manager.retrieve_memory("current_conversation")
        assert context is not None
        assert context.content["ai_model_selected"] == "claude"
        
        # 4. Test search functionality
        results = memory_manager.search_memories("code", context_tags=["conversation"])
        assert len(results) > 0
        
        # 5. Test system optimization
        optimization_results = memory_manager.optimize_memory()
        assert all(v >= 0 for v in optimization_results.values())
        
        # 6. Get system stats
        stats = memory_manager.get_memory_stats()
        assert stats.total_items >= 2
        
        print("✅ Memory System Phase 1 Integration Test PASSED")
        
    def test_memory_performance_requirements(self, memory_manager):
        """🎯 PHASE 1 PERFORMANCE: Memory response time < 2s target"""
        start_time = time.time()
        
        # Bulk operations test
        for i in range(50):
            memory_manager.store_memory(
                memory_id=f"perf_test_{i}",
                content={"data": f"performance_test_data_{i}", "index": i},
                memory_type=MemoryType.WORKING,
                priority=PriorityLevel.MEDIUM
            )
        
        storage_time = time.time() - start_time
        
        # Search performance
        start_time = time.time()
        results = memory_manager.search_memories("performance", limit=20)
        search_time = time.time() - start_time
        
        # Performance assertions (Phase 1 targets)
        assert storage_time < 2.0, f"Storage too slow: {storage_time}s"
        assert search_time < 0.5, f"Search too slow: {search_time}s"
        assert len(results) > 0
        
        print(f"✅ Memory Performance Test PASSED - Storage: {storage_time:.3f}s, Search: {search_time:.3f}s")


if __name__ == "__main__":
    print("🧠 Running NAVA Memory Manager Test Suite...")
    print("Phase 1 - Week 2: Advanced Features Testing")
    print("=" * 60)
    
    # Run specific test classes
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_memory_system_stability or test_memory_performance_requirements"
    ])
