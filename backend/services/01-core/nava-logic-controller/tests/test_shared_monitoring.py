# backend/services/01-core/nava-logic-controller/tests/test_shared_monitoring.py
"""
Tests for Shared Monitoring System
Validates system monitoring, metrics collection, and alerting
"""

import pytest
import sys
import os
import time
import threading
from datetime import datetime, timedelta

# 🔧 FIX: Correct path to shared directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up from tests/ to nava-logic-controller/
service_dir = os.path.dirname(current_dir)
# Go up from nava-logic-controller/ to 01-core/
core_dir = os.path.dirname(service_dir)
# Go up from 01-core/ to services/
services_dir = os.path.dirname(core_dir)
# Go up from services/ to backend/
backend_dir = os.path.dirname(services_dir)
# Now go to shared/
shared_dir = os.path.join(backend_dir, "shared")

# Add shared directory to Python path
sys.path.insert(0, shared_dir)

# Now import should work
from common.monitoring import (
    SystemMonitor,
    MetricPoint,
    ServiceHealth,
    system_monitor,
    record_metric,
    record_request,
    record_ai_request,
    update_service_health,
    get_system_status,
    create_alert,
    run_health_checks,
    cleanup_monitoring_data
)

class TestSystemMonitor:
    """Test suite for SystemMonitor"""
    
    def setup_method(self):
        """Setup for each test"""
        self.monitor = SystemMonitor(retention_hours=1)  # Short retention for testing
        
        # 🔧 Enhanced cleanup - Clear all existing data completely
        self.monitor.metrics.clear()
        self.monitor.service_health.clear()
        self.monitor.alerts.clear()
        self.monitor.request_counters.clear()
        self.monitor.response_times.clear()
        self.monitor.error_counts.clear()
        
        # Also clear any alert rules and health checks
        if hasattr(self.monitor, 'alert_rules'):
            self.monitor.alert_rules.clear()
        if hasattr(self.monitor, 'health_checks'):
            self.monitor.health_checks.clear()
        
        print(f"🧹 Test setup complete - All data cleared")
    
    def test_monitor_initialization(self):
        """Test monitor initializes correctly"""
        assert self.monitor is not None
        assert self.monitor.retention_hours == 1
        assert isinstance(self.monitor.metrics, dict)
        assert isinstance(self.monitor.service_health, dict)
        assert isinstance(self.monitor.alerts, list)
        assert self.monitor.start_time is not None
        print("✅ Monitor initialization test passed")
    
    def test_record_metric(self):
        """Test recording metrics"""
        self.monitor.record_metric("test.metric", 42.5)
        
        assert "test.metric" in self.monitor.metrics
        points = list(self.monitor.metrics["test.metric"])
        assert len(points) == 1
        assert points[0].value == 42.5
        assert isinstance(points[0].timestamp, datetime)
        print("✅ Record metric test passed")
    
    def test_record_metric_with_tags(self):
        """Test recording metrics with tags"""
        tags = {"service": "nava", "environment": "test"}
        metadata = {"additional": "info"}
        
        self.monitor.record_metric("test.tagged", 100, tags=tags, metadata=metadata)
        
        points = list(self.monitor.metrics["test.tagged"])
        assert len(points) == 1
        assert points[0].tags == tags
        assert points[0].metadata == metadata
        print("✅ Record metric with tags test passed")
    
    def test_record_request(self):
        """Test recording API requests"""
        metric_key = "nava-controller.chat"
        
        # 🔧 Debug: Check initial state
        print(f"🔍 Initial - Counter: {self.monitor.request_counters.get(metric_key, 0)}")
        print(f"🔍 Initial - Errors: {self.monitor.error_counts.get(metric_key, 0)}")
        
        self.monitor.record_request("nava-controller", "chat", 1.5, 200)
        
        print(f"🔍 After record - Counter: {self.monitor.request_counters.get(metric_key, 0)}")
        print(f"🔍 After record - Errors: {self.monitor.error_counts.get(metric_key, 0)}")
        
        assert self.monitor.request_counters[metric_key] == 1
        assert len(self.monitor.response_times[metric_key]) == 1
        assert self.monitor.response_times[metric_key][0] == 1.5
        assert self.monitor.error_counts[metric_key] == 0
        print("✅ Record request test passed")
    
    def test_record_request_with_error(self):
        """Test recording API requests with errors"""
        # 🔧 Use different service name to avoid conflicts
        service_name = "error-test-service"
        endpoint = "error-endpoint"
        metric_key = f"{service_name}.{endpoint}"
        
        print(f"🔍 Testing with unique key: {metric_key}")
        print(f"🔍 Before test - Counter: {self.monitor.request_counters.get(metric_key, 0)}")
        print(f"🔍 Before test - Errors: {self.monitor.error_counts.get(metric_key, 0)}")
        
        self.monitor.record_request(service_name, endpoint, 2.0, 500)
        
        print(f"🔍 After test - Counter: {self.monitor.request_counters.get(metric_key, 0)}")
        print(f"🔍 After test - Errors: {self.monitor.error_counts.get(metric_key, 0)}")
        
        assert self.monitor.request_counters[metric_key] == 1
        assert self.monitor.error_counts[metric_key] == 1
        print("✅ Record request with error test passed")
    
    def test_record_ai_request(self):
        """Test recording AI model requests"""
        self.monitor.record_ai_request("gpt", 2.5, True, 0.85)
        
        metric_key = "ai.gpt"
        assert self.monitor.request_counters[metric_key] == 1
        assert len(self.monitor.response_times[metric_key]) == 1
        assert self.monitor.response_times[metric_key][0] == 2.5
        assert self.monitor.error_counts[metric_key] == 0
        
        # Check metrics were recorded
        assert "ai.gpt.requests" in self.monitor.metrics
        assert "ai.gpt.response_time" in self.monitor.metrics
        assert "ai.gpt.quality_score" in self.monitor.metrics
        print("✅ Record AI request test passed")
    
    def test_update_service_health(self):
        """Test updating service health"""
        self.monitor.update_service_health(
            "nava-controller",
            "healthy",
            response_time=1.2,
            error_rate=0.05,
            issues=[],
            metadata={"version": "2.0.0"}
        )
        
        assert "nava-controller" in self.monitor.service_health
        health = self.monitor.service_health["nava-controller"]
        
        assert health.service_name == "nava-controller"
        assert health.status == "healthy"
        assert health.response_time == 1.2
        assert health.error_rate == 0.05
        assert health.metadata["version"] == "2.0.0"
        assert health.uptime_percentage > 0
        print("✅ Update service health test passed")
    
    def test_get_system_overview(self):
        """Test getting system overview"""
        # 🔧 Clear all service health data first to ensure clean test
        print("🔍 Clearing existing service health data...")
        self.monitor.service_health.clear()
        self.monitor.request_counters.clear()
        self.monitor.error_counts.clear()
        self.monitor.response_times.clear()
        
        print("🔍 Setting up fresh test data for system overview...")
        
        # Add test requests with unique keys
        self.monitor.record_request("overview-service1", "endpoint1", 1.0, 200)
        self.monitor.record_request("overview-service1", "endpoint1", 2.0, 500)
        
        # Add service health data
        self.monitor.update_service_health("overview-service1", "healthy")
        self.monitor.update_service_health("overview-service2", "degraded")
        
        print(f"🔍 Service health count: {len(self.monitor.service_health)}")
        print(f"🔍 Services: {list(self.monitor.service_health.keys())}")
        
        overview = self.monitor.get_system_overview()
        
        print(f"🔍 Overview services: {overview['services']}")
        
        assert "status" in overview
        assert "timestamp" in overview
        assert "uptime_seconds" in overview
        assert "services" in overview
        assert "requests" in overview
        assert "alerts" in overview
        
        # Check services summary
        services = overview["services"]
        assert services["total"] == 2, f"Expected 2 services, got {services['total']}"
        assert services["healthy"] == 1, f"Expected 1 healthy service, got {services['healthy']}"
        assert services["degraded"] == 1, f"Expected 1 degraded service, got {services['degraded']}"
        
        # Check requests summary
        requests = overview["requests"]
        assert requests["total"] == 2
        assert requests["error_count"] == 1
        assert requests["error_rate_percent"] == 50.0
        print("✅ Get system overview test passed")
    
    def test_create_alert(self):
        """Test creating alerts"""
        alert_id = self.monitor.create_alert(
            "high_error_rate",
            "Error rate exceeded threshold",
            severity="warning",
            service="test-service",
            metadata={"threshold": 0.1, "current": 0.15}
        )
        
        assert alert_id is not None
        assert len(self.monitor.alerts) == 1
        
        alert = self.monitor.alerts[0]
        assert alert["id"] == alert_id
        assert alert["type"] == "high_error_rate"
        assert alert["message"] == "Error rate exceeded threshold"
        assert alert["severity"] == "warning"
        assert alert["service"] == "test-service"
        assert alert["status"] == "active"
        assert "created_at" in alert
        assert alert["metadata"]["threshold"] == 0.1
        print("✅ Create alert test passed")

class TestMonitoringFunctions:
    """Test standalone monitoring functions"""
    
    def test_record_metric_function(self):
        """Test record_metric function"""
        record_metric("test.function.metric", 123.4)
        
        # Should be recorded in global monitor
        assert "test.function.metric" in system_monitor.metrics
        print("✅ Record metric function test passed")
    
    def test_get_system_status_function(self):
        """Test get_system_status function"""
        status = get_system_status()
        
        assert isinstance(status, dict)
        assert "status" in status
        assert "timestamp" in status
        print("✅ Get system status function test passed")

# Simplified test runner
if __name__ == "__main__":
    # Run tests manually for better debugging
    print("🧪 Starting Monitoring Tests...")
    
    # Test basic functionality
    test_monitor = TestSystemMonitor()
    test_monitor.setup_method()
    
    try:
        # Test basic functionality with unique keys for each test
        test_monitor.test_monitor_initialization()
        test_monitor.test_record_metric()
        test_monitor.test_record_metric_with_tags()
        
        # Create fresh instance for request tests to avoid conflicts
        print("\n🔄 Creating fresh monitor for request tests...")
        test_monitor = TestSystemMonitor()
        test_monitor.setup_method()
        test_monitor.test_record_request()
        
        # Use same instance for error test but with unique keys
        test_monitor.test_record_request_with_error()
        test_monitor.test_record_ai_request()
        test_monitor.test_update_service_health()
        
        # Continue with same instance for system overview (needs accumulated data)
        test_monitor.test_get_system_overview()
        test_monitor.test_create_alert()
        
        # Test functions
        test_functions = TestMonitoringFunctions()
        test_functions.test_record_metric_function()
        test_functions.test_get_system_status_function()
        
        print("🎉 All monitoring tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
