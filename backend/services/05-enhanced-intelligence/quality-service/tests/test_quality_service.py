# backend/services/05-enhanced-intelligence/quality-service/tests/test_quality.py
"""
Tests for Quality Service Microservice
Validates comprehensive quality validation API and functionality
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

# Import the FastAPI app and components
try:
    from quality_validator import quality_router, QualityValidator, QualityRequest, QualityResponse
    from fastapi import FastAPI
    
    # Create test app
    app = FastAPI()
    app.include_router(quality_router)
    client = TestClient(app)
    
except ImportError as e:
    print(f"Warning: Could not import quality service components: {e}")
    # Create mock components for testing
    client = None

class TestQualityServiceAPI:
    """Test Quality Service REST API endpoints"""
    
    def setup_method(self):
        """Setup for each test"""
        if client is None:
            pytest.skip("Quality service not available")
    
    def test_validate_endpoint_simple(self):
        """Test basic validation endpoint"""
        response = client.post("/validate", json={
            "response_text": "This is a good quality response with helpful information.",
            "original_query": "What is AI?",
            "model_used": "gpt"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "quality_id" in data
        assert "overall_score" in data
        assert "quality_level" in data
        assert "dimension_scores" in data
        assert "compliance_status" in data
        assert "improvement_areas" in data
        assert "passed_thresholds" in data
        assert "processing_time_seconds" in data
        assert "timestamp" in data
        
        # Check data types
        assert isinstance(data["overall_score"], float)
        assert 0.0 <= data["overall_score"] <= 1.0
        assert data["quality_level"] in ["excellent", "good", "acceptable", "poor", "unacceptable"]
    
    def test_validate_endpoint_comprehensive(self):
        """Test validation with all parameters"""
        request_data = {
            "response_text": "Thank you for your excellent question about artificial intelligence. AI is a comprehensive field that encompasses machine learning, natural language processing, and computer vision.",
            "original_query": "What is artificial intelligence?",
            "context": {
                "user_level": "beginner",
                "domain": "technology"
            },
            "quality_requirements": {
                "minimum_score": "0.8",
                "safety_score": "0.95"
            },
            "model_used": "claude",
            "response_time": 2.5
        }
        
        response = client.post("/validate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have good scores for comprehensive response
        assert data["overall_score"] > 0.7
        assert data["quality_level"] in ["good", "excellent"]
        
        # Check dimension scores
        dimensions = data["dimension_scores"]
        assert "accuracy" in dimensions
        assert "completeness" in dimensions
        assert "clarity" in dimensions
        assert "relevance" in dimensions
        assert "safety" in dimensions
        assert "compliance" in dimensions
    
    def test_validate_endpoint_poor_quality(self):
        """Test validation of poor quality response"""
        response = client.post("/validate", json={
            "response_text": "No.",
            "original_query": "Explain quantum computing in detail"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have low scores for poor response
        assert data["overall_score"] < 0.6
        assert data["quality_level"] in ["poor", "unacceptable"]
        assert len(data["improvement_areas"]) > 0
    
    def test_validate_endpoint_harmful_content(self):
        """Test validation detects harmful content"""
        response = client.post("/validate", json={
            "response_text": "This response contains violence and dangerous harmful content that could hurt people.",
            "original_query": "How to be safe?"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have low safety score
        assert data["dimension_scores"]["safety"] < 0.9
        assert data["passed_thresholds"]["safety_critical"] is False
    
    def test_get_dimensions_endpoint(self):
        """Test getting quality dimensions configuration"""
        response = client.get("/dimensions")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "dimensions" in data
        assert "thresholds" in data
        assert "quality_levels" in data
        assert "timestamp" in data
        
        # Check required dimensions
        dimensions = data["dimensions"]
        required_dims = ["accuracy", "completeness", "clarity", "relevance", "safety", "compliance"]
        for dim in required_dims:
            assert dim in dimensions
            assert "weight" in dimensions[dim]
            assert "description" in dimensions[dim]
    
    def test_quick_quality_check_endpoint(self):
        """Test quick quality check endpoint"""
        response = client.post("/quick", params={
            "response_text": "This is a reasonable response with useful information.",
            "min_threshold": 0.7
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "overall_score" in data
        assert "quality_level" in data
        assert "passes_threshold" in data
        assert "recommendation" in data
        assert "top_issue" in data
        assert "timestamp" in data
        
        # Check recommendation logic
        if data["passes_threshold"]:
            assert data["recommendation"] == "approve"
        else:
            assert data["recommendation"] == "review_needed"
    
    def test_batch_validation_endpoint(self):
        """Test batch validation endpoint"""
        batch_data = {
            "responses": [
                {
                    "response_text": "Good response with detailed information.",
                    "original_query": "Question 1"
                },
                {
                    "response_text": "Poor response.",
                    "original_query": "Question 2"
                },
                {
                    "response_text": "Excellent comprehensive response with examples and explanations.",
                    "original_query": "Question 3"
                }
            ],
            "threshold": 0.75
        }
        
        response = client.post("/batch", json=batch_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "summary" in data
        assert "timestamp" in data
        
        # Check results
        results = data["results"]
        assert len(results) == 3
        
        for result in results:
            assert "index" in result
            assert "overall_score" in result
            assert "quality_level" in result
            assert "passes_threshold" in result
        
        # Check summary
        summary = data["summary"]
        assert "total_responses" in summary
        assert "passed_threshold" in summary
        assert "pass_rate" in summary
        assert "average_score" in summary
    
    def test_simple_batch_validation_endpoint(self):
        """Test simple batch validation endpoint"""
        responses = [
            "This is a good response.",
            "Bad response.",
            "Excellent detailed response with comprehensive information."
        ]
        
        response = client.post("/batch/simple", json=responses, params={"threshold": 0.6})
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "summary" in data
        assert len(data["results"]) == 3
    
    def test_validation_error_handling(self):
        """Test error handling in validation"""
        # Test with missing required field
        response = client.post("/validate", json={
            # Missing response_text
            "original_query": "Test query"
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_validation_with_very_long_text(self):
        """Test validation with very long text"""
        long_text = "A" * 60000  # Longer than max_length
        
        response = client.post("/validate", json={
            "response_text": long_text,
            "original_query": "Test query"
        })
        
        assert response.status_code == 422  # Should reject too long text
    
    def test_validation_with_empty_text(self):
        """Test validation with empty text"""
        response = client.post("/validate", json={
            "response_text": "",
            "original_query": "Test query"
        })
        
        assert response.status_code == 422  # Should reject empty text

class TestQualityValidatorLogic:
    """Test Quality Validator core logic"""
    
    def setup_method(self):
        """Setup for each test"""
        if 'QualityValidator' not in globals():
            pytest.skip("QualityValidator not available")
        self.validator = QualityValidator()
    
    def test_validator_initialization(self):
        """Test validator initializes correctly"""
        assert self.validator.quality_dimensions is not None
        assert len(self.validator.quality_dimensions) == 6
        assert self.validator.enterprise_thresholds is not None
        assert self.validator.quality_levels is not None
    
    @pytest.mark.asyncio
    async def test_validate_quality_method(self):
        """Test validate_quality method"""
        request = QualityRequest(
            response_text="This is a comprehensive response with detailed information.",
            original_query="What is machine learning?"
        )
        
        result = await self.validator.validate_quality(request)
        
        assert "dimension_scores" in result
        assert "overall_score" in result
        assert "quality_level" in result
        assert "compliance_status" in result
    
    def test_dimension_analysis(self):
        """Test individual dimension analysis"""
        text = "This is a professional response with appropriate disclaimers."
        
        # Test accuracy analysis
        accuracy_score, accuracy_metrics = self.validator._analyze_accuracy(text, "Test query", {})
        assert 0.0 <= accuracy_score <= 1.0
        assert isinstance(accuracy_metrics, dict)
        
        # Test safety analysis
        safety_score, safety_metrics = self.validator._analyze_safety(text, {})
        assert 0.0 <= safety_score <= 1.0
        assert isinstance(safety_metrics, dict)
        
        # Test compliance analysis
        compliance_score, compliance_metrics = self.validator._analyze_compliance(text, {})
        assert 0.0 <= compliance_score <= 1.0
        assert isinstance(compliance_metrics, dict)
    
    def test_overall_score_calculation(self):
        """Test overall score calculation"""
        dimension_scores = {
            "accuracy": 0.8,
            "completeness": 0.7,
            "clarity": 0.9,
            "relevance": 0.8,
            "safety": 0.95,
            "compliance": 0.85
        }
        
        overall_score = self.validator._calculate_overall_score(dimension_scores)
        assert 0.0 <= overall_score <= 1.0
        assert 0.7 <= overall_score <= 0.9  # Should be in reasonable range
    
    def test_quality_level_determination(self):
        """Test quality level determination"""
        assert self.validator._determine_quality_level(0.95) == "excellent"
        assert self.validator._determine_quality_level(0.80) == "good"
        assert self.validator._determine_quality_level(0.65) == "acceptable"
        assert self.validator._determine_quality_level(0.45) == "poor"
        assert self.validator._determine_quality_level(0.25) == "unacceptable"
    
    def test_compliance_checking(self):
        """Test compliance checking"""
        dimension_scores = {
            "accuracy": 0.8,
            "safety": 0.96,
            "compliance": 0.91
        }
        
        compliance_status = self.validator._check_compliance(dimension_scores, {})
        
        assert "minimum_quality" in compliance_status
        assert "safety_threshold" in compliance_status
        assert "compliance_threshold" in compliance_status
    
    def test_improvement_areas_identification(self):
        """Test improvement areas identification"""
        dimension_scores = {
            "accuracy": 0.6,  # Below threshold
            "completeness": 0.8,
            "clarity": 0.5,   # Below threshold
            "relevance": 0.8,
            "safety": 0.95,
            "compliance": 0.85
        }
        
        detailed_metrics = {
            "accuracy": {"readability_concern": "Sentences too long"},
            "clarity": {}
        }
        
        improvements = self.validator._identify_improvement_areas(dimension_scores, detailed_metrics)
        
        assert len(improvements) > 0
        assert any("accuracy" in imp.lower() for imp in improvements)
        assert any("clarity" in imp.lower() for imp in improvements)
    
    def test_emergency_validation(self):
        """Test emergency validation fallback"""
        result = self.validator._emergency_validation("Test text")
        
        assert "dimension_scores" in result
        assert "overall_score" in result
        assert "quality_level" in result
        assert result["quality_level"] in ["acceptable", "poor"]

class TestQualityServiceIntegration:
    """Test Quality Service integration scenarios"""
    
    def setup_method(self):
        """Setup for each test"""
        if client is None:
            pytest.skip("Quality service not available")
    
    def test_end_to_end_validation_workflow(self):
        """Test complete validation workflow"""
        # Step 1: Get dimensions
        dimensions_response = client.get("/dimensions")
        assert dimensions_response.status_code == 200
        
        # Step 2: Validate a response
        validation_response = client.post("/validate", json={
            "response_text": "This is a comprehensive answer with detailed explanations.",
            "original_query": "Explain the concept"
        })
        assert validation_response.status_code == 200
        
        # Step 3: Quick check
        quick_response = client.post("/quick", params={
            "response_text": "Quick response",
            "min_threshold": 0.7
        })
        assert quick_response.status_code == 200
    
    def test_concurrent_validation_requests(self):
        """Test concurrent validation handling"""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request(text_suffix):
            try:
                response = client.post("/validate", json={
                    "response_text": f"Test response {text_suffix}",
                    "original_query": "Test query"
                })
                results.append(response.status_code)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0
        assert all(status == 200 for status in results)
    
    def test_performance_under_load(self):
        """Test performance under load"""
        import time
        
        start_time = time.time()
        
        # Make multiple requests
        for i in range(20):
            response = client.post("/validate", json={
                "response_text": f"Performance test response {i}",
                "original_query": "Performance test query"
            })
            assert response.status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete reasonably quickly
        assert total_time < 10.0  # Less than 10 seconds for 20 requests
        
        avg_time_per_request = total_time / 20
        assert avg_time_per_request < 0.5  # Less than 500ms per request

class TestQualityServiceEdgeCases:
    """Test edge cases and error scenarios"""
    
    def setup_method(self):
        """Setup for each test"""
        if client is None:
            pytest.skip("Quality service not available")
    
    def test_special_characters_handling(self):
        """Test handling of special characters"""
        special_text = "Response with émojis 🤖, symbols ©™®, and unicode: 你好世界"
        
        response = client.post("/validate", json={
            "response_text": special_text,
            "original_query": "Test query"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "overall_score" in data
    
    def test_malformed_json_handling(self):
        """Test handling of malformed requests"""
        # Invalid JSON structure
        response = client.post("/validate", 
                             data="invalid json",
                             headers={"Content-Type": "application/json"})
        
        assert response.status_code == 422
    
    def test_missing_optional_fields(self):
        """Test handling of missing optional fields"""
        response = client.post("/validate", json={
            "response_text": "Minimal request test"
            # Missing all optional fields
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "overall_score" in data
    
    def test_extreme_threshold_values(self):
        """Test extreme threshold values"""
        # Very high threshold
        response = client.post("/quick", params={
            "response_text": "Good response",
            "min_threshold": 0.99
        })
        assert response.status_code == 200
        
        # Very low threshold
        response = client.post("/quick", params={
            "response_text": "Poor response",
            "min_threshold": 0.01
        })
        assert response.status_code == 200

# Mock components for when service is not available
class MockQualityValidator:
    """Mock quality validator for testing when service is unavailable"""
    
    def __init__(self):
        self.quality_dimensions = {
            "accuracy": {"weight": 0.25},
            "completeness": {"weight": 0.20},
            "clarity": {"weight": 0.20},
            "relevance": {"weight": 0.15},
            "safety": {"weight": 0.10},
            "compliance": {"weight": 0.10}
        }
    
    async def validate_quality(self, request):
        return {
            "dimension_scores": {"accuracy": 0.8, "safety": 0.95},
            "overall_score": 0.8,
            "quality_level": "good",
            "compliance_status": {"passes": True},
            "improvement_areas": [],
            "threshold_results": {"passes": True}
        }

# Test configuration
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment"""
    if client is None:
        print("Warning: Running tests with mock components")

# Run specific test groups
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])