# backend/services/01-core/nava-logic-controller/app/core/performance_optimizer.py
"""
Performance Optimizer - Week 1 Enhancement
เพิ่มความเร็วของระบบให้จาก 4.6s → <2s
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional
import sys
import os

# Add backend path to sys.path for imports
backend_path = os.path.join(os.path.dirname(__file__), '../../../../..')
sys.path.insert(0, backend_path)

from services.shared.common.cache_manager import global_cache
from services.shared.common.circuit_breaker import circuit_breaker

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Performance optimization for NAVA responses"""
    
    def __init__(self):
        self.optimization_enabled = True
        self.performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'fast_routes': 0,
            'slow_routes': 0,
            'avg_response_time': 0.0
        }
        
    async def optimize_request_routing(self, message: str, complexity: str = "medium") -> Dict[str, Any]:
        """Optimize routing based on request characteristics"""
        
        # Fast route for simple requests
        if complexity == "simple" or len(message) < 50:
            return {
                'route': 'fast',
                'timeout': 10,
                'priority_models': ['gpt', 'local'],
                'cache_priority': 'high'
            }
        
        # Complex requests need more time but better models
        elif complexity in ["complex", "critical"]:
            return {
                'route': 'quality',
                'timeout': 25,
                'priority_models': ['claude', 'gemini'],
                'cache_priority': 'medium'
            }
        
        # Default medium complexity
        return {
            'route': 'balanced',
            'timeout': 15,
            'priority_models': ['gpt', 'claude'],
            'cache_priority': 'high'
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        total_requests = self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']
        hit_rate = (self.performance_stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.performance_stats,
            'cache_hit_rate': round(hit_rate, 2),
            'total_requests': total_requests
        }

# Global optimizer instance
performance_optimizer = PerformanceOptimizer()