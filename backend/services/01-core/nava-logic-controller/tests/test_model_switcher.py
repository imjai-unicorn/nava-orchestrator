# backend/services/01-core/nava-logic-controller/tests/test_model_switcher.py
"""
Tests for Model Switcher
Validates AI model selection and switching logic
"""

import pytest
import sys
import os
from datetime import datetime, timedelta

# Add app directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(os.path.dirname(current_dir), 'app')
sys.path.insert(0, app_dir)

from core.model_switcher import (
    ModelSwitcher, 
    AIModel,
    select_ai_model,
    get_model_endpoint,
    report_success,
    report_failure,
    get_all_model_status
)

class TestModelSwitcher:
    """Test suite for ModelSwitcher"""
    
    def setup_method(self):
        """Setup for each test"""
        self.switcher = ModelSwitcher()
        # Reset all models to healthy state
        for model in AIModel:
            # ✅ เก็บ availability ตามที่ design ไว้ - ไม่ override
            if model != AIModel.LOCAL:  # Keep LOCAL as False (Phase 1)
                self.switcher.models[model]["availability"] = True
            self.switcher.models[model]["failure_count"] = 0
            self.switcher.models[model]["last_failure"] = None

    def test_switcher_initialization(self):
        """Test switcher initializes correctly"""
        assert self.switcher is not None
        assert len(self.switcher.models) == 4  # GPT, Claude, Gemini, Local
        assert AIModel.GPT in self.switcher.models
        assert AIModel.CLAUDE in self.switcher.models
        assert AIModel.GEMINI in self.switcher.models
        assert AIModel.LOCAL in self.switcher.models
    
    def test_model_configurations(self):
        """Test model configurations are valid"""
        for model, config in self.switcher.models.items():
            assert "name" in config
            assert "endpoint" in config
            assert "strengths" in config
            assert "performance_score" in config
            assert "availability" in config
            assert "cost_per_request" in config
            assert "avg_response_time" in config
            assert isinstance(config["strengths"], list)
            assert len(config["strengths"]) > 0
    
    def test_select_best_model_balanced(self):
        """Test balanced model selection"""
        selected = self.switcher.select_best_model(
            task_type="general",
            priority="balanced"
        )
        
        assert selected in [AIModel.GPT, AIModel.CLAUDE, AIModel.GEMINI]
        assert self.switcher._is_model_available(selected)
    
    def test_select_best_model_speed_priority(self):
        """Test speed-prioritized model selection"""
        selected = self.switcher.select_best_model(
            task_type="general",
            priority="speed"
        )
        
        # Should prefer faster models
        assert selected in [AIModel.GPT, AIModel.CLAUDE, AIModel.GEMINI]
        
        # Verify it's not the slowest option when speed is priority
        selected_config = self.switcher.models[selected]
        assert selected_config["avg_response_time"] <= 5.0
    
    def test_select_best_model_quality_priority(self):
        """Test quality-prioritized model selection"""
        selected = self.switcher.select_best_model(
            task_type="analysis",
            priority="quality"
        )
        
        # Should prefer higher performance models
        selected_config = self.switcher.models[selected]
        assert selected_config["performance_score"] >= 0.8
    
    def test_select_best_model_cost_priority(self):
        """Test cost-prioritized model selection"""
        selected = self.switcher.select_best_model(
            task_type="general",
            priority="cost"
        )
        
        # Should prefer cheaper models
        selected_config = self.switcher.models[selected]
        assert selected_config["cost_per_request"] <= 0.15
    
    def test_user_preference_selection(self):
        """Test user preference overrides"""
        # Test valid user preference
        selected = self.switcher.select_best_model(
            task_type="general",
            user_preference="claude"
        )
        assert selected == AIModel.CLAUDE
        
        # Test invalid user preference
        selected = self.switcher.select_best_model(
            task_type="general",
            user_preference="invalid_model"
        )
        # Should fall back to algorithmic selection
        assert selected in [AIModel.GPT, AIModel.CLAUDE, AIModel.GEMINI]
    
    def test_task_type_matching(self):
        """Test task type influences model selection"""
        # Test creative task
        creative_score_gpt = self.switcher._calculate_task_match(AIModel.GPT, "creative")
        creative_score_claude = self.switcher._calculate_task_match(AIModel.CLAUDE, "creative")
        
        # GPT should score well for creative tasks
        assert creative_score_gpt > 0
        
        # Test analysis task
        analysis_score_claude = self.switcher._calculate_task_match(AIModel.CLAUDE, "analysis")
        analysis_score_gpt = self.switcher._calculate_task_match(AIModel.GPT, "analysis")
        
        # Claude should score well for analysis tasks
        assert analysis_score_claude > 0
    
    def test_model_availability_check(self):
        """Test model availability checking"""
        # All models should be available initially
        for model in [AIModel.GPT, AIModel.CLAUDE, AIModel.GEMINI]:
            assert self.switcher._is_model_available(model) is True
        
        # Local should not be available in Phase 1
        assert self.switcher._is_model_available(AIModel.LOCAL) is False
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker triggers correctly"""
        model = AIModel.GPT
        
        # Report multiple failures
        for i in range(self.switcher.circuit_breaker_threshold):
            self.switcher.report_model_failure(model, "timeout")
        
        # Model should be unavailable due to circuit breaker
        assert self.switcher._is_model_available(model) is False
        
        # Check circuit breaker status
        status = self.switcher.get_model_status()
        assert status[model.value]["circuit_breaker_active"] is True
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout"""
        model = AIModel.GPT
        
        # Trigger circuit breaker
        for i in range(self.switcher.circuit_breaker_threshold):
            self.switcher.report_model_failure(model, "timeout")
        
        # Manually set last failure to past the timeout period
        past_time = datetime.now() - timedelta(seconds=self.switcher.circuit_breaker_timeout + 1)
        self.switcher.models[model]["last_failure"] = past_time
        
        # Should be available again
        assert self.switcher._is_model_available(model) is True
        
        # Failure count should be reset
        assert self.switcher.models[model]["failure_count"] == 0
    
    def test_success_reporting(self):
        """Test success reporting updates metrics"""
        model = AIModel.GPT
        original_response_time = self.switcher.models[model]["avg_response_time"]
        
        # Report a fast success
        self.switcher.report_model_success(model, 1.0)
        
        # Average response time should be updated
        new_response_time = self.switcher.models[model]["avg_response_time"]
        assert new_response_time != original_response_time
    
    def test_failure_reporting(self):
        """Test failure reporting updates metrics"""
        model = AIModel.GPT
        original_failure_count = self.switcher.models[model]["failure_count"]
        
        # Report failure
        self.switcher.report_model_failure(model, "timeout")
        
        # Failure count should increase
        new_failure_count = self.switcher.models[model]["failure_count"]
        assert new_failure_count == original_failure_count + 1
        assert self.switcher.models[model]["last_failure"] is not None
    
    def test_fallback_chain(self):
        """Test fallback chain when models are unavailable"""
        # Make all models except Gemini unavailable
        self.switcher.set_model_availability(AIModel.GPT, False)
        self.switcher.set_model_availability(AIModel.CLAUDE, False)
        # Local is already unavailable
        
        selected = self.switcher.select_best_model()
        assert selected == AIModel.GEMINI
    
    def test_no_models_available_fallback(self):
        """Test behavior when no models are available"""
        # Make all models unavailable
        for model in AIModel:
            self.switcher.set_model_availability(model, False)
        
        selected = self.switcher.select_best_model()
        # Should return GPT as last resort
        assert selected == AIModel.GPT
    
    def test_model_scoring_calculation(self):
        """Test model scoring calculation"""
        model = AIModel.GPT
        score = self.switcher._calculate_model_score(
            model, "general", "balanced", {}
        )
        
        assert isinstance(score, float)
        assert score >= 0
        # Score should be reasonable for a healthy model
        assert score > 0.3
    
    def test_priority_adjustments(self):
        """Test priority adjustments affect scores"""
        model = AIModel.GPT
        base_score = 0.5
        
        speed_score = self.switcher._apply_priority_adjustments(base_score, model, "speed")
        quality_score = self.switcher._apply_priority_adjustments(base_score, model, "quality")
        cost_score = self.switcher._apply_priority_adjustments(base_score, model, "cost")
        balanced_score = self.switcher._apply_priority_adjustments(base_score, model, "balanced")
        
        # Scores should be different based on priority
        assert balanced_score == base_score  # Balanced should not change score
        # Other priorities may adjust scores based on model characteristics
    
    def test_get_model_status(self):
        """Test getting model status"""
        status = self.switcher.get_model_status()
        
        assert isinstance(status, dict)
        assert len(status) == 4  # All 4 models
        
        for model_name, model_status in status.items():
            assert "available" in model_status
            assert "healthy" in model_status
            assert "performance_score" in model_status
            assert "avg_response_time" in model_status
            assert "failure_count" in model_status
            assert "circuit_breaker_active" in model_status
            assert "endpoint" in model_status
    
    def test_recommended_fallback_chain(self):
        """Test recommended fallback chain for specific tasks"""
        creative_chain = self.switcher.get_recommended_fallback_chain("creative")
        analysis_chain = self.switcher.get_recommended_fallback_chain("analysis")
        
        assert isinstance(creative_chain, list)
        assert isinstance(analysis_chain, list)
        assert len(creative_chain) > 0
        assert len(analysis_chain) > 0
        
        # Should contain available models
        for model in creative_chain:
            assert model in AIModel
            assert self.switcher.models[model]["availability"]
    
    def test_model_config_updates(self):
        """Test updating model configuration"""
        model = AIModel.GPT
        original_score = self.switcher.models[model]["performance_score"]
        
        # Update configuration
        self.switcher.update_model_config(model, {"performance_score": 0.95})
        
        # Should be updated
        assert self.switcher.models[model]["performance_score"] == 0.95
        assert self.switcher.models[model]["performance_score"] != original_score

class TestModelSwitcherFunctions:
    """Test standalone functions"""
    
    def test_select_ai_model_function(self):
        """Test select_ai_model function"""
        selected = select_ai_model("general")
        assert selected in AIModel
    
    def test_get_model_endpoint_function(self):
        """Test get_model_endpoint function"""
        endpoint = get_model_endpoint(AIModel.GPT)
        assert endpoint == "http://localhost:8002"
        
        endpoint = get_model_endpoint(AIModel.CLAUDE)
        assert endpoint == "http://localhost:8003"
    
    def test_report_success_function(self):
        """Test report_success function"""
        # Should not crash
        report_success(AIModel.GPT, 2.5)
    
    def test_report_failure_function(self):
        """Test report_failure function"""
        # Should not crash
        report_failure(AIModel.GPT, "timeout")
    
    def test_get_all_model_status_function(self):
        """Test get_all_model_status function"""
        status = get_all_model_status()
        
        assert isinstance(status, dict)
        assert len(status) >= 3  # At least GPT, Claude, Gemini

class TestModelSwitcherEdgeCases:
    """Test edge cases and error handling"""
    
    def setup_method(self):
        """Setup for each test"""
        self.switcher = ModelSwitcher()
    
    def test_invalid_task_type(self):
        """Test handling of invalid task type"""
        selected = self.switcher.select_best_model(task_type="invalid_task")
        
        # Should still return a valid model
        assert selected in AIModel
    
    def test_invalid_priority(self):
        """Test handling of invalid priority"""
        selected = self.switcher.select_best_model(priority="invalid_priority")
        
        # Should still return a valid model
        assert selected in AIModel
    
    def test_empty_context(self):
        """Test handling of empty context"""
        selected = self.switcher.select_best_model(context={})
        
        # Should still return a valid model
        assert selected in AIModel
    
    def test_none_context(self):
        """Test handling of None context"""
        selected = self.switcher.select_best_model(context=None)
        
        # Should still return a valid model
        assert selected in AIModel
    
    def test_concurrent_access(self):
        """Test concurrent access to model switcher"""
        import threading
        
        selections = []
        errors = []
        
        def select_model(task_type, index):
            try:
                selected = self.switcher.select_best_model(task_type=f"task_{index}")
                selections.append(selected)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=select_model, args=("general", i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0  # No errors should occur
        assert len(selections) == 10  # All selections should complete
        
        # All selections should be valid models
        for selection in selections:
            assert selection in AIModel

class TestModelSwitcherPerformance:
    """Test performance aspects of model switcher"""
    
    def setup_method(self):
        """Setup for each test"""
        self.switcher = ModelSwitcher()
    
    def test_selection_performance(self):
        """Test model selection performance"""
        import time
        
        start_time = time.time()
        
        # Perform multiple selections
        for i in range(100):
            selected = self.switcher.select_best_model(
                task_type="general",
                priority="balanced"
            )
            assert selected in AIModel
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete 100 selections quickly
        assert total_time < 1.0  # Less than 1 second
        
        # Average time per selection should be reasonable
        avg_time = total_time / 100
        assert avg_time < 0.01  # Less than 10ms per selection
    
    def test_status_check_performance(self):
        """Test status checking performance"""
        import time
        
        start_time = time.time()
        
        # Check status multiple times
        for i in range(50):
            status = self.switcher.get_model_status()
            assert isinstance(status, dict)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete 50 status checks quickly
        assert total_time < 0.5  # Less than 0.5 seconds

class TestModelSwitcherIntegration:
    """Integration tests for model switcher"""
    
    def setup_method(self):
        """Setup for each test"""
        self.switcher = ModelSwitcher()
    
    def test_full_workflow_simulation(self):
        """Test full workflow from selection to reporting"""
        # Select a model
        selected = self.switcher.select_best_model(
            task_type="conversation",
            priority="balanced"
        )
        
        # Get endpoint
        endpoint = get_model_endpoint(selected)
        assert endpoint.startswith("http://")
        
        # Simulate successful request
        response_time = 2.5
        self.switcher.report_model_success(selected, response_time)
        
        # Check status
        status = self.switcher.get_model_status()
        assert status[selected.value]["healthy"] is True
    
    def test_failure_recovery_workflow(self):
        """Test failure and recovery workflow"""
        model = AIModel.GPT
        
        # Simulate some failures
        for i in range(3):
            self.switcher.report_model_failure(model, "timeout")
        
        # Model should still be available (not enough failures)
        assert self.switcher._is_model_available(model) is True
        
        # Simulate more failures to trigger circuit breaker
        for i in range(3):
            self.switcher.report_model_failure(model, "timeout")
        
        # Model should be unavailable
        assert self.switcher._is_model_available(model) is False
        
        # Selection should fall back to other models
        selected = self.switcher.select_best_model()
        assert selected != model
        
        # Simulate recovery
        past_time = datetime.now() - timedelta(seconds=400)
        self.switcher.models[model]["last_failure"] = past_time
        
        # Model should be available again
        assert self.switcher._is_model_available(model) is True
    
    def test_load_balancing_simulation(self):
        """Test load balancing behavior"""
        selections = []
        
        # Make multiple selections with different priorities
        priorities = ["speed", "quality", "cost", "balanced"]
        
        for i in range(20):
            priority = priorities[i % len(priorities)]
            selected = self.switcher.select_best_model(
                task_type="general",
                priority=priority
            )
            selections.append((selected, priority))
        
        # Should have selections
        assert len(selections) == 20
        
        # Should use different models based on priorities
        selected_models = set(selection[0] for selection in selections)
        assert len(selected_models) > 1  # Should use multiple models

# Pytest fixtures
@pytest.fixture
def switcher():
    """Fixture providing a fresh model switcher"""
    return ModelSwitcher()

@pytest.fixture
def healthy_models():
    """Fixture providing all models in healthy state"""
    switcher = ModelSwitcher()
    for model in AIModel:
        if model != AIModel.LOCAL:  # Local not available in Phase 1
            switcher.set_model_availability(model, True)
            switcher.models[model]["failure_count"] = 0
            switcher.models[model]["last_failure"] = None
    return switcher

@pytest.fixture
def task_types():
    """Fixture providing various task types"""
    return [
        "conversation",
        "creative", 
        "analysis",
        "research",
        "code",
        "safety",
        "general"
    ]

# Integration with other components
class TestModelSwitcherWithOtherComponents:
    """Test integration with other NAVA components"""
    
    def test_integration_with_quality_validator(self):
        """Test that model switcher works with quality validator"""
        # This test would verify that model selection considers quality scores
        # For now, just ensure no conflicts
        switcher = ModelSwitcher()
        selected = switcher.select_best_model("general")
        assert selected in AIModel
    
    def test_integration_with_context_manager(self):
        """Test that model switcher works with context manager"""
        # This test would verify context-aware model selection
        # For now, just ensure no conflicts
        switcher = ModelSwitcher()
        context = {"user_preference": "claude", "conversation_length": 5}
        selected = switcher.select_best_model("general", context=context)
        assert selected in AIModel

# Run specific test groups
if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])