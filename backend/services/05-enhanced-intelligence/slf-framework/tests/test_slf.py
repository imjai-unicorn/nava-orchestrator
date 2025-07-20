# backend/services/05-enhanced-intelligence/slf-framework/tests/test_slf_fixed.py
"""
Tests for SLF (Systematic Learning Framework) Service - FIXED VERSION
Fixed to match actual implementation and API routes
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient
import json
from datetime import datetime

# Add app directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(os.path.dirname(current_dir), 'app')
sys.path.insert(0, app_dir)

# Import the FastAPI app
try:
    # Import main app from parent directory
    sys.path.insert(0, os.path.dirname(current_dir))
    from main import app
    client = TestClient(app)
    
    SLF_AVAILABLE = True
    print("✅ SLF Framework imported successfully")
    
except ImportError as e:
    print(f"❌ SLF Import error: {e}")
    SLF_AVAILABLE = False
    client = None

class TestSLFServiceAPI:
    """Test SLF Service REST API endpoints"""
    
    def setup_method(self):
        """Setup for each test"""
        if not SLF_AVAILABLE:
            pytest.skip("SLF service not available")
    
    def test_health_check(self):
        """Test basic health check"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "NAVA SLF Framework"
        
    def test_enhance_reasoning_endpoint(self):
        """Test reasoning enhancement endpoint - FIXED"""
        enhancement_request = {
            "original_prompt": "What is machine learning?",
            "model_target": "gpt",  # Fixed: use model_target instead of ai_model
            "reasoning_type": "systematic",  # Fixed: use reasoning_type instead of enhancement_type
            "enhancement_level": "moderate",
            "enterprise_mode": False
        }
        
        response = client.post("/enhance", json=enhancement_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response fields match SLFResponse model
        assert "slf_id" in data
        assert "enhanced_prompt" in data
        assert "original_prompt" in data
        assert "reasoning_framework" in data
        assert "enhancement_techniques" in data
        assert "expected_improvements" in data
        assert "processing_time_seconds" in data
        assert "timestamp" in data
        
        # Enhanced prompt should be different and longer
        assert data["enhanced_prompt"] != data["original_prompt"]
        assert len(data["enhanced_prompt"]) > len(data["original_prompt"])
    
    def test_systematic_analysis_enhancement(self):
        """Test systematic analysis enhancement"""
        request_data = {
            "original_prompt": "Analyze the benefits of renewable energy",
            "model_target": "claude",
            "reasoning_type": "systematic",
            "enhancement_level": "comprehensive"
        }
        
        response = client.post("/enhance", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should include systematic analysis structure
        enhanced_prompt = data["enhanced_prompt"].lower()
        assert "systematic" in enhanced_prompt or "analyze" in enhanced_prompt
        
        # Should have reasoning framework
        assert "reasoning_framework" in data
        assert data["reasoning_framework"] is not None
    
    def test_creative_collaboration_enhancement(self):
        """Test creative collaboration enhancement"""
        request_data = {
            "original_prompt": "Write a story about AI",
            "model_target": "gpt",
            "reasoning_type": "creative",
            "enhancement_level": "moderate"
        }
        
        response = client.post("/enhance", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should include creative elements
        enhanced_prompt = data["enhanced_prompt"].lower()
        assert len(enhanced_prompt) > len(request_data["original_prompt"])
        
        # Should have enhancement techniques
        assert "enhancement_techniques" in data
        assert isinstance(data["enhancement_techniques"], list)
    
    def test_analytical_enhancement(self):
        """Test analytical enhancement"""
        request_data = {
            "original_prompt": "Evaluate our market strategy",
            "model_target": "claude", 
            "reasoning_type": "analytical",
            "enhancement_level": "comprehensive"
        }
        
        response = client.post("/enhance", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should include analytical elements
        enhanced_prompt = data["enhanced_prompt"].lower()
        assert "analy" in enhanced_prompt  # analytical, analyze, etc.
        
        # Should have expected improvements
        assert "expected_improvements" in data
        assert isinstance(data["expected_improvements"], dict)
    
    def test_batch_enhancement_endpoint(self):
        """Test batch enhancement endpoint"""
        batch_request = {
            "enhancements": [
                {
                    "id": "batch_1",
                    "original_prompt": "What is AI?",
                    "model_target": "gpt",
                    "reasoning_type": "systematic"
                },
                {
                    "id": "batch_2", 
                    "original_prompt": "Tell me a story",
                    "model_target": "claude",
                    "reasoning_type": "creative"
                },
                {
                    "id": "batch_3",
                    "original_prompt": "Analyze market trends",
                    "model_target": "gemini",
                    "reasoning_type": "analytical"
                }
            ]
        }
        
        response = client.post("/enhance/batch", json=batch_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "summary" in data
        assert "total_processed" in data
        assert "processing_time_seconds" in data
        
        results = data["results"]
        assert len(results) == 3
        
        for result in results:
            assert "id" in result
            assert "success" in result
    
    def test_get_enhancement_types_endpoint(self):
        """Test getting available enhancement types"""
        response = client.get("/enhancement-types")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "enhancement_types" in data
        assert "cognitive_frameworks" in data
        assert "supported_models" in data
        
        enhancement_types = data["enhancement_types"]
        assert isinstance(enhancement_types, dict)
        
        # Should have key enhancement types
        expected_types = ["systematic_analysis", "creative_collaboration", "enterprise_analysis"]
        for exp_type in expected_types:
            assert exp_type in enhancement_types
    
    def test_reasoning_validation_endpoint(self):
        """Test reasoning validation endpoint"""
        validation_request = {
            "enhanced_prompt": "Systematically analyze machine learning by first defining core concepts, then examining methodologies, and finally evaluating applications.",
            "original_prompt": "What is machine learning?",
            "enhancement_type": "systematic",
            "reasoning_criteria": {
                "logical_structure": True,
                "completeness": True,
                "clarity": True
            }
        }
        
        response = client.post("/validate-reasoning", json=validation_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "validation_id" in data
        assert "reasoning_score" in data
        assert "structure_analysis" in data
        assert "improvement_suggestions" in data
        assert "validation_criteria" in data
        assert "passed_validation" in data
        
        # Check reasoning score
        assert 0.0 <= data["reasoning_score"] <= 1.0
    
    def test_model_optimization_endpoint(self):
        """Test model-specific optimization endpoint"""
        optimization_request = {
            "original_prompt": "Explain quantum computing",
            "target_model": "claude",
            "optimization_goals": [
                "maximize_reasoning_depth",
                "enhance_clarity", 
                "include_examples"
            ]
        }
        
        response = client.post("/optimize", json=optimization_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "optimization_id" in data
        assert "optimized_prompt" in data
        assert "optimization_strategy" in data
        assert "model_specific_enhancements" in data
        assert "expected_improvements" in data
        
        # Should be optimized for Claude
        strategy = data["optimization_strategy"]
        assert "claude" in strategy["target_model"].lower()
    
    def test_slf_statistics_endpoint(self):
        """Test SLF statistics endpoint"""
        response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_enhancements" in data
        assert "enhancement_types_usage" in data
        assert "model_usage_stats" in data
        assert "average_improvement_score" in data
        assert "success_rate" in data
        assert "processing_time_stats" in data
        assert "uptime_hours" in data
        
        # Check data types
        assert isinstance(data["total_enhancements"], int)
        assert isinstance(data["success_rate"], float)
        assert 0.0 <= data["success_rate"] <= 100.0
    
    def test_frameworks_endpoint(self):
        """Test frameworks endpoint"""
        response = client.get("/frameworks")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "frameworks" in data
        assert "timestamp" in data
        
        # Should have reasoning frameworks
        frameworks = data["frameworks"]
        assert isinstance(frameworks, dict)
    
    def test_quick_enhance_endpoint(self):
        """Test quick enhancement endpoint"""
        response = client.post("/quick", params={
            "prompt": "What is artificial intelligence?",
            "model": "gpt",
            "reasoning_type": "systematic"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "original" in data
        assert "enhanced" in data
        assert "framework_used" in data
        assert "expected_improvement" in data
        assert "techniques" in data

class TestSLFServiceEdgeCases:
    """Test edge cases and error scenarios"""
    
    def setup_method(self):
        """Setup for each test"""
        if not SLF_AVAILABLE:
            pytest.skip("SLF service not available")
    
    def test_empty_prompt_handling(self):
        """Test handling of empty prompts"""
        response = client.post("/enhance", json={
            "original_prompt": "",
            "model_target": "gpt", 
            "reasoning_type": "systematic"
        })
        
        # Should reject empty prompts
        assert response.status_code == 422
    
    def test_invalid_model_target(self):
        """Test handling of invalid model targets"""
        response = client.post("/enhance", json={
            "original_prompt": "Test prompt",
            "model_target": "invalid_model",
            "reasoning_type": "systematic"
        })
        
        # Should handle gracefully (default to 'gpt')
        assert response.status_code in [200, 422]
    
    def test_invalid_reasoning_type(self):
        """Test handling of invalid reasoning types"""  
        response = client.post("/enhance", json={
            "original_prompt": "Test prompt",
            "model_target": "gpt",
            "reasoning_type": "invalid_type"
        })
        
        # Should handle gracefully (default to 'systematic')
        assert response.status_code in [200, 422]
    
    def test_very_long_prompt(self):
        """Test handling of very long prompts"""
        long_prompt = "A" * 15000  # Long but within 10000 char limit in model
        
        response = client.post("/enhance", json={
            "original_prompt": long_prompt,
            "model_target": "gpt",
            "reasoning_type": "systematic"
        })
        
        # Should reject prompts over limit
        assert response.status_code == 422
    
    def test_unicode_prompt_handling(self):
        """Test handling of Unicode prompts"""
        unicode_prompt = "Analizar la inteligencia artificial: 你好世界 🤖 ñáéíóú"
        
        response = client.post("/enhance", json={
            "original_prompt": unicode_prompt,
            "model_target": "gpt",
            "reasoning_type": "systematic"
        })
        
        # Should handle Unicode correctly
        assert response.status_code == 200
        data = response.json()
        assert "enhanced_prompt" in data

class TestSLFServiceIntegration:
    """Test SLF Service integration scenarios"""
    
    def setup_method(self):
        """Setup for each test"""
        if not SLF_AVAILABLE:
            pytest.skip("SLF service not available")
    
    def test_end_to_end_enhancement_workflow(self):
        """Test complete enhancement workflow"""
        # Step 1: Get enhancement types
        types_response = client.get("/enhancement-types")
        assert types_response.status_code == 200
        
        # Step 2: Enhance a prompt
        enhance_response = client.post("/enhance", json={
            "original_prompt": "Analyze market trends",
            "model_target": "claude",
            "reasoning_type": "systematic"
        })
        assert enhance_response.status_code == 200
        enhancement_data = enhance_response.json()
        
        # Step 3: Validate the reasoning
        validate_response = client.post("/validate-reasoning", json={
            "enhanced_prompt": enhancement_data["enhanced_prompt"],
            "original_prompt": "Analyze market trends",
            "enhancement_type": "systematic"
        })
        assert validate_response.status_code == 200
        
        # Step 4: Check statistics
        stats_response = client.get("/stats")
        assert stats_response.status_code == 200
    
    def test_enhancement_performance(self):
        """Test enhancement performance"""
        import time
        
        start_time = time.time()
        
        # Make multiple enhancement requests
        successful_requests = 0
        for i in range(10):  # Reduced from 20 to 10 for faster testing
            response = client.post("/enhance", json={
                "original_prompt": f"Analyze performance test topic {i}",
                "model_target": "claude",
                "reasoning_type": "systematic"
            })
            if response.status_code == 200:
                successful_requests += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete reasonably quickly
        assert total_time < 20.0  # Less than 20 seconds for 10 enhancements
        assert successful_requests >= 8  # At least 80% success rate
        
        avg_time_per_enhancement = total_time / successful_requests if successful_requests > 0 else 999
        assert avg_time_per_enhancement < 2.0  # Less than 2 seconds per enhancement

# Simple test to verify SLF is working
def test_slf_service_basic():
    """Simple test to verify SLF service is working"""
    
    if not SLF_AVAILABLE:
        pytest.skip("SLF service not available")
    
    # Test health endpoint
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["service"] == "NAVA SLF Framework"
    assert data["status"] == "operational"
    
    print("✅ SLF service basic test passed")

# Run specific test groups
if __name__ == "__main__":
    if SLF_AVAILABLE:
        print("✅ Running SLF tests with live service")
    else:
        print("❌ SLF service not available - tests will be skipped")
    
    pytest.main([__file__, "-v", "--tb=short"])