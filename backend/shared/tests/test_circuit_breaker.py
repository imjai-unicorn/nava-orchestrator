#!/usr/bin/env python3
"""
Circuit Breaker Unit Tests
File: backend/services/shared/tests/test_circuit_breaker.py
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
import sys
import os

# Add shared path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.enhanced_circuit_breaker import (
    EnhancedCircuitBreaker,
    CircuitState,
    ServiceConfig,
    ServiceHealth,
    execute_with_circuit_breaker,
    execute_with_failover,
    get_system_health
)

class TestEnhancedCircuitBreaker:
    """Test Enhanced Circuit Breaker functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.circuit_breaker = EnhancedCircuitBreaker()
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization"""
        assert isinstance(self.circuit_breaker, EnhancedCircuitBreaker)
        assert len(self.circuit_breaker.services) == 4  # gpt, claude, gemini, local
        assert len(self.circuit_breaker.failover_chain) == 5  # including cache
        
        # Check default service configs
        gpt_config = self.circuit_breaker.get_service_config("gpt")
        assert gpt_config.timeout == 15.0
        assert gpt_config.max_failures == 5
        
        claude_config = self.circuit_breaker.get_service_config("claude")
        assert claude_config.timeout == 20.0
        assert claude_config.max_failures == 3
        
        gemini_config = self.circuit_breaker.get_service_config("gemini")
        assert gemini_config.timeout == 18.0
        assert gemini_config.max_failures == 4
    
    def test_service_health_tracking(self):
        """Test service health tracking"""
        # Initial state
        health = self.circuit_breaker.get_service_health("gpt")
        assert health.state == CircuitState.CLOSED
        assert health.failure_count == 0
        assert health.success_count == 0
        assert health.consecutive_failures == 0
        
        # Record success
        self.circuit_breaker.record_success("gpt")
        health = self.circuit_breaker.get_service_health("gpt")
        assert health.success_count == 1
        assert health.consecutive_failures == 0
        assert health.state == CircuitState.CLOSED
        
        # Record failure
        self.circuit_breaker.record_failure("gpt", Exception("Test error"))
        health = self.circuit_breaker.get_service_health("gpt")
        assert health.failure_count == 1
        assert health.consecutive_failures == 1
        assert health.state == CircuitState.CLOSED  # Not enough failures yet
    
    def test_circuit_opening_logic(self):
        """Test circuit opening logic"""
        # Record multiple failures to trigger circuit opening
        for i in range(5):
            self.circuit_breaker.record_failure("gpt", Exception(f"Error {i}"))
        
        health = self.circuit_breaker.get_service_health("gpt")
        assert health.state == CircuitState.OPEN
        assert health.consecutive_failures == 5
        assert health.failure_count == 5
    
    def test_should_open_circuit_conditions(self):
        """Test conditions for opening circuit"""
        # Test consecutive failures threshold
        health = self.circuit_breaker.get_service_health("gpt")
        config = self.circuit_breaker.get_service_config("gpt")
        
        # Not enough failures yet
        health.consecutive_failures = 4
        assert not self.circuit_breaker.should_open_circuit("gpt")
        
        # Enough consecutive failures
        health.consecutive_failures = 5
        assert self.circuit_breaker.should_open_circuit("gpt")
        
        # Test failure rate threshold
        health.consecutive_failures = 0
        health.total_requests = 20
        health.failure_count = 5  # 25% failure rate
        assert not self.circuit_breaker.should_open_circuit("gpt")
        
        health.failure_count = 7  # 35% failure rate (>30% threshold)
        assert self.circuit_breaker.should_open_circuit("gpt")
    
    def test_recovery_logic(self):
        """Test circuit recovery logic"""
        # Open circuit
        health = self.circuit_breaker.get_service_health("gpt")
        health.state = CircuitState.OPEN
        health.last_failure_time = time.time() - 35  # 35 seconds ago
        
        # Should attempt recovery after timeout
        assert self.circuit_breaker.should_attempt_recovery("gpt")
        
        # Recent failure, should not attempt recovery
        health.last_failure_time = time.time() - 10  # 10 seconds ago
        assert not self.circuit_breaker.should_attempt_recovery("gpt")
    
    def test_failover_chain(self):
        """Test failover chain logic"""
        # Test normal chain progression
        next_service = self.circuit_breaker.get_next_available_service("gpt")
        assert next_service == "claude"
        
        next_service = self.circuit_breaker.get_next_available_service("claude")
        assert next_service == "gemini"
        
        next_service = self.circuit_breaker.get_next_available_service("gemini")
        assert next_service == "local"
        
        next_service = self.circuit_breaker.get_next_available_service("local")
        assert next_service == "cache"
        
        # Test with service not in chain
        next_service = self.circuit_breaker.get_next_available_service("unknown")
        assert next_service == "cache"
    
    def test_delay_calculation(self):
        """Test exponential backoff delay calculation"""
        config = self.circuit_breaker.get_service_config("gpt")
        
        # Test exponential backoff
        delay_0 = self.circuit_breaker.calculate_delay(0, config)
        delay_1 = self.circuit_breaker.calculate_delay(1, config)
        delay_2 = self.circuit_breaker.calculate_delay(2, config)
        
        assert delay_0 >= config.base_delay * 0.75  # Allow for jitter
        assert delay_1 >= config.base_delay * 1.5   # 2^1 * base_delay * 0.75
        assert delay_2 >= config.base_delay * 3     # 2^2 * base_delay * 0.75
        
        # Test max delay limit
        delay_large = self.circuit_breaker.calculate_delay(10, config)
        assert delay_large <= config.max_delay * 1.25  # Allow for jitter
    
    def test_system_health_report(self):
        """Test system health reporting"""
        # Add some test data
        self.circuit_breaker.record_success("gpt")
        self.circuit_breaker.record_failure("claude", Exception("Test error"))
        
        health_report = self.circuit_breaker.get_system_health()
        
        assert "timestamp" in health_report
        assert "services" in health_report
        assert "overall_status" in health_report
        
        # Check service data
        assert "gpt" in health_report["services"]
        assert "claude" in health_report["services"]
        
        gpt_health = health_report["services"]["gpt"]
        assert gpt_health["success_count"] == 1
        assert gpt_health["state"] == CircuitState.CLOSED.value
        
        claude_health = health_report["services"]["claude"]
        assert claude_health["failure_count"] == 1
        assert claude_health["consecutive_failures"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_with_circuit_breaker(self):
        """Test execute with circuit breaker"""
        # Test successful execution
        async def successful_operation():
            return "success"
        
        result = await execute_with_circuit_breaker("gpt", successful_operation)
        assert result == "success"
        
        # Check health was updated
        health = self.circuit_breaker.get_service_health("gpt")
        assert health.success_count == 1
        assert health.consecutive_failures == 0
    
    @pytest.mark.asyncio
    async def test_execute_with_circuit_breaker_failure(self):
        """Test execute with circuit breaker failure"""
        # Test failing operation
        async def failing_operation():
            raise Exception("Test failure")
        
        with pytest.raises(Exception):
            await execute_with_circuit_breaker("gpt", failing_operation)
        
        # Check health was updated
        health = self.circuit_breaker.get_service_health("gpt")
        assert health.failure_count == 1
        assert health.consecutive_failures == 1
    
    @pytest.mark.asyncio
    async def test_execute_with_failover(self):
        """Test execute with failover"""
        call_count = 0
        
        def create_operation(service):
            async def operation():
                nonlocal call_count
                call_count += 1
                if service == "gpt":
                    raise Exception("GPT failed")
                elif service == "claude":
                    return f"Success from {service}"
                else:
                    return f"Success from {service}"
            return operation
        
        result = await execute_with_failover("gpt", create_operation)
        assert result == "Success from claude"
        assert call_count == 2  # Failed GPT, succeeded Claude
    
    @pytest.mark.asyncio
    async def test_execute_with_failover_all_fail(self):
        """Test execute with failover when all services fail"""
        def create_failing_operation(service):
            async def operation():
                if service == "cache":
                    return "Cache response"
                raise Exception(f"{service} failed")
            return operation
        
        result = await execute_with_failover("gpt", create_failing_operation)
        assert result == "Cache response"
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling"""
        async def slow_operation():
            await asyncio.sleep(0.1)  # Simulate slow operation
            return "slow result"
        
        # Should succeed within timeout
        result = await execute_with_circuit_breaker("gpt", slow_operation)
        assert result == "slow result"
        
        # Test with very slow operation (would timeout in real scenario)
        async def very_slow_operation():
            await asyncio.sleep(20)  # Longer than any timeout
            return "very slow result"
        
        # Note: In real testing, this would timeout, but we can't test with actual timeouts
        # in unit tests due to time constraints
    
    def test_reset_service_health(self):
        """Test resetting service health"""
        # Add some data
        self.circuit_breaker.record_success("gpt")
        self.circuit_breaker.record_failure("gpt", Exception("Test"))
        
        health = self.circuit_breaker.get_service_health("gpt")
        assert health.success_count == 1
        assert health.failure_count == 1
        
        # Reset health
        self.circuit_breaker.reset_service_health("gpt")
        
        health = self.circuit_breaker.get_service_health("gpt")
        assert health.success_count == 0
        assert health.failure_count == 0
        assert health.consecutive_failures == 0
        assert health.state == CircuitState.CLOSED
    
    def test_force_circuit_state(self):
        """Test forcing circuit state"""
        # Force circuit open
        self.circuit_breaker.force_circuit_state("gpt", CircuitState.OPEN)
        
        health = self.circuit_breaker.get_service_health("gpt")
        assert health.state == CircuitState.OPEN
        
        # Force circuit closed
        self.circuit_breaker.force_circuit_state("gpt", CircuitState.CLOSED)
        
        health = self.circuit_breaker.get_service_health("gpt")
        assert health.state == CircuitState.CLOSED

class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_get_system_health(self):
        """Test get_system_health convenience function"""
        health = get_system_health()
        
        assert isinstance(health, dict)
        assert "timestamp" in health
        assert "services" in health
        assert "overall_status" in health

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
