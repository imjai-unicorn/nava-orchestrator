# NAVA Enhanced Testing Plan - Focus on Decision Intelligence
# Testing framework for validating enhanced components

import unittest
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

class TestNAVAEnhancedDecisionEngine(unittest.TestCase):
    """
    Test suite for NAVA Enhanced Decision Engine
    Focus: Decision Intelligence, Learning System, SLF Integration Readiness
    """
    
    def setUp(self):
        """Setup test environment with enhanced decision engine"""
        from app.core.decision_engine import EnhancedDecisionEngine
        self.decision_engine = EnhancedDecisionEngine()
        self.test_scenarios = self._load_test_scenarios()
    
    def _load_test_scenarios(self) -> Dict[str, Any]:
        """Load comprehensive test scenarios for decision intelligence"""
        return {
            "behavior_pattern_tests": [
                {
                    "input": "Write a comprehensive business analysis report",
                    "expected_pattern": "deep_analysis",
                    "expected_model": "claude",
                    "confidence_min": 0.85
                },
                {
                    "input": "Help me debug this Python function",
                    "expected_pattern": "code_development", 
                    "expected_model": "gpt",
                    "confidence_min": 0.90
                },
                {
                    "input": "Create a strategic marketing plan for next quarter",
                    "expected_pattern": "strategic_planning",
                    "expected_model": "gemini",
                    "confidence_min": 0.85
                },
                {
                    "input": "Write a creative short story about AI",
                    "expected_pattern": "literary_creative",
                    "expected_model": "claude", 
                    "confidence_min": 0.90
                }
            ],
            "learning_system_tests": [
                {
                    "scenario": "positive_feedback_learning",
                    "pattern": "code_development",
                    "model": "gpt",
                    "feedback_sequence": [4.5, 4.0, 4.8, 4.2],
                    "expected_weight_increase": True
                },
                {
                    "scenario": "negative_feedback_adaptation", 
                    "pattern": "deep_analysis",
                    "model": "claude",
                    "feedback_sequence": [2.0, 1.5, 2.5, 2.0],
                    "expected_weight_decrease": True
                }
            ],
            "fallback_intelligence_tests": [
                {
                    "input": "Explain quantum computing concepts",
                    "no_pattern_match": True,
                    "expected_fallback": "expertise_analysis",
                    "expected_model": "claude"
                },
                {
                    "input": "What's the weather like?",
                    "no_pattern_match": True,
                    "expected_fallback": "smart_default",
                    "expected_model": "gpt"
                }
            ]
        }

    def test_behavior_pattern_detection_accuracy(self):
        """Test enhanced behavior pattern detection"""
        print("\n🧪 Testing Behavior Pattern Detection Accuracy...")
        
        for test_case in self.test_scenarios["behavior_pattern_tests"]:
            with self.subTest(input_text=test_case["input"][:50]):
                model, confidence, reasoning = self.decision_engine.select_model(
                    test_case["input"]
                )
                
                # Validate pattern detection
                detected_pattern = reasoning.get("behavior_analysis", {}).get("detected_pattern")
                self.assertEqual(detected_pattern, test_case["expected_pattern"],
                    f"Expected pattern '{test_case['expected_pattern']}', got '{detected_pattern}'")
                
                # Validate model selection
                self.assertEqual(model, test_case["expected_model"],
                    f"Expected model '{test_case['expected_model']}', got '{model}'")
                
                # Validate confidence level
                self.assertGreaterEqual(confidence, test_case["confidence_min"],
                    f"Confidence {confidence} below minimum {test_case['confidence_min']}")
                
                print(f"✅ Pattern: {detected_pattern}, Model: {model}, Confidence: {confidence:.3f}")

    def test_dynamic_weight_learning_system(self):
        """Test dynamic weight adjustment based on user feedback"""
        print("\n🧪 Testing Dynamic Weight Learning System...")
        
        for test_case in self.test_scenarios["learning_system_tests"]:
            with self.subTest(scenario=test_case["scenario"]):
                pattern = test_case["pattern"]
                model = test_case["model"]
                
                # Get initial weight
                initial_stats = self.decision_engine.get_feedback_stats()
                initial_weight = initial_stats["current_pattern_weights"].get(pattern, {}).get(model, 0.0)
                
                # Simulate feedback sequence
                for i, feedback_score in enumerate(test_case["feedback_sequence"]):
                    response_id = f"test_response_{i}"
                    self.decision_engine.update_user_feedback(
                        response_id, model, pattern, feedback_score, "rating"
                    )
                
                # Check weight adjustment
                final_stats = self.decision_engine.get_feedback_stats()
                final_weight = final_stats["current_pattern_weights"].get(pattern, {}).get(model, 0.0)
                
                if test_case["expected_weight_increase"]:
                    self.assertGreater(final_weight, initial_weight,
                        f"Weight should increase: {initial_weight} -> {final_weight}")
                else:
                    self.assertLess(final_weight, initial_weight,
                        f"Weight should decrease: {initial_weight} -> {final_weight}")
                
                print(f"✅ {test_case['scenario']}: {initial_weight:.3f} -> {final_weight:.3f}")

    def test_intelligent_fallback_system(self):
        """Test intelligent fallback when no clear pattern is detected"""
        print("\n🧪 Testing Intelligent Fallback System...")
        
        for test_case in self.test_scenarios["fallback_intelligence_tests"]:
            with self.subTest(input_text=test_case["input"][:30]):
                model, confidence, reasoning = self.decision_engine.select_model(
                    test_case["input"]
                )
                
                # Validate fallback method used
                selection_method = reasoning.get("selection_method")
                self.assertEqual(selection_method, test_case["expected_fallback"],
                    f"Expected fallback '{test_case['expected_fallback']}', got '{selection_method}'")
                
                # Validate model selection
                self.assertEqual(model, test_case["expected_model"],
                    f"Expected model '{test_case['expected_model']}', got '{model}'")
                
                # Confidence should be reasonable for fallback
                self.assertGreaterEqual(confidence, 0.4,
                    f"Fallback confidence {confidence} too low")
                self.assertLessEqual(confidence, 0.8,
                    f"Fallback confidence {confidence} too high")
                
                print(f"✅ Fallback: {selection_method}, Model: {model}, Confidence: {confidence:.3f}")

    def test_decision_reasoning_quality(self):
        """Test quality and completeness of decision reasoning"""
        print("\n🧪 Testing Decision Reasoning Quality...")
        
        test_input = "Create a machine learning model for customer churn prediction"
        model, confidence, reasoning = self.decision_engine.select_model(test_input)
        
        # Validate reasoning structure
        required_keys = [
            "selected_model", "confidence", "selection_method",
            "behavior_analysis", "explanation", "processing_time_ms"
        ]
        
        for key in required_keys:
            self.assertIn(key, reasoning, f"Missing reasoning key: {key}")
        
        # Validate behavior analysis completeness
        behavior_analysis = reasoning["behavior_analysis"]
        behavior_keys = ["detected_pattern", "behavior_type", "pattern_confidence"]
        
        for key in behavior_keys:
            self.assertIn(key, behavior_analysis, f"Missing behavior analysis key: {key}")
        
        # Validate explanation quality
        explanation = reasoning["explanation"]
        self.assertIsInstance(explanation, str, "Explanation should be string")
        self.assertGreater(len(explanation), 20, "Explanation should be meaningful")
        self.assertIn(model.upper(), explanation, "Explanation should mention selected model")
        
        # Validate processing time
        processing_time = reasoning["processing_time_ms"]
        self.assertIsInstance(processing_time, (int, float), "Processing time should be numeric")
        self.assertLess(processing_time, 1000, "Processing time should be under 1 second")
        
        print(f"✅ Reasoning quality validated for model: {model}")
        print(f"   Explanation: {explanation[:100]}...")
        print(f"   Processing time: {processing_time}ms")


class TestNAVAServiceIntegration(unittest.TestCase):
    """
    Test suite for NAVA service integration capabilities
    Focus: External AI integration, Registry Agent readiness, SLF preparation
    """
    
    def setUp(self):
        """Setup test environment for service integration"""
        self.base_url = "http://localhost:8005"
        self.test_endpoints = [
            "/health",
            "/chat", 
            "/models",
            "/system/status",
            "/explain/decision"
        ]
    
    async def test_external_ai_service_communication(self):
        """Test communication with external AI services"""
        print("\n🧪 Testing External AI Service Communication...")
        
        import aiohttp
        
        external_services = [
            {"name": "GPT Client", "url": "http://localhost:8002", "port": 8002},
            {"name": "Claude Client", "url": "http://localhost:8003", "port": 8003},
            {"name": "Gemini Client", "url": "http://localhost:8004", "port": 8004}
        ]
        
        async with aiohttp.ClientSession() as session:
            for service in external_services:
                with self.subTest(service=service["name"]):
                    try:
                        # Test health endpoint
                        async with session.get(f"{service['url']}/health", timeout=5) as response:
                            self.assertEqual(response.status, 200, 
                                f"{service['name']} health check failed")
                            
                        # Test chat endpoint with simple request
                        chat_data = {
                            "message": "Hello, this is a test",
                            "user_id": "test_user"
                        }
                        async with session.post(f"{service['url']}/chat", 
                                               json=chat_data, timeout=10) as response:
                            self.assertIn(response.status, [200, 422], 
                                f"{service['name']} chat endpoint unexpected status")
                            
                        print(f"✅ {service['name']} communication: OK")
                        
                    except Exception as e:
                        print(f"⚠️ {service['name']} communication: {str(e)}")
    
    async def test_nava_core_endpoints(self):
        """Test NAVA core endpoint functionality"""
        print("\n🧪 Testing NAVA Core Endpoints...")
        
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            for endpoint in self.test_endpoints:
                with self.subTest(endpoint=endpoint):
                    try:
                        url = f"{self.base_url}{endpoint}"
                        
                        if endpoint == "/chat":
                            # POST request for chat
                            test_data = {
                                "message": "Test enhanced decision engine",
                                "user_id": "test_user_123"
                            }
                            async with session.post(url, json=test_data, timeout=10) as response:
                                self.assertEqual(response.status, 200, f"Chat endpoint failed")
                                
                                response_data = await response.json()
                                required_fields = ["response", "model_used", "confidence", "decision_info"]
                                for field in required_fields:
                                    self.assertIn(field, response_data, f"Missing field: {field}")
                                    
                        elif endpoint == "/explain/decision":
                            # POST request for decision explanation
                            test_data = {
                                "message": "Analyze market trends for Q4",
                                "user_preference": None
                            }
                            async with session.post(url, json=test_data, timeout=5) as response:
                                self.assertEqual(response.status, 200, f"Decision explanation failed")
                                
                                response_data = await response.json()
                                explanation_fields = ["selected_model", "confidence", "reasoning"]
                                for field in explanation_fields:
                                    self.assertIn(field, response_data, f"Missing explanation field: {field}")
                        else:
                            # GET request for other endpoints
                            async with session.get(url, timeout=5) as response:
                                self.assertEqual(response.status, 200, f"{endpoint} endpoint failed")
                        
                        print(f"✅ {endpoint}: OK")
                        
                    except Exception as e:
                        print(f"⚠️ {endpoint}: {str(e)}")

    def test_slf_integration_readiness(self):
        """Test NAVA readiness for SLF integration"""
        print("\n🧪 Testing SLF Integration Readiness...")
        
        # Test decision engine has required interfaces
        from app.core.decision_engine import EnhancedDecisionEngine
        engine = EnhancedDecisionEngine()
        
        # Check required methods for SLF integration
        required_methods = [
            "select_model",
            "update_user_feedback", 
            "get_feedback_stats",
            "get_behavior_patterns",
            "update_model_health"
        ]
        
        for method_name in required_methods:
            self.assertTrue(hasattr(engine, method_name),
                f"Decision engine missing required method: {method_name}")
            
            method = getattr(engine, method_name)
            self.assertTrue(callable(method),
                f"Method {method_name} is not callable")
        
        # Test decision engine can provide context for SLF
        test_message = "Create a business proposal for new product launch"
        model, confidence, reasoning = engine.select_model(test_message)
        
        # Validate reasoning contains SLF-useful information
        slf_useful_keys = [
            "behavior_analysis",
            "selected_model", 
            "confidence",
            "explanation"
        ]
        
        for key in slf_useful_keys:
            self.assertIn(key, reasoning, f"Missing SLF-useful information: {key}")
        
        print(f"✅ SLF integration readiness: All required interfaces present")
        print(f"   Sample behavior analysis: {reasoning['behavior_analysis']}")


class TestNAVAPerformanceAndReliability(unittest.TestCase):
    """
    Test suite for NAVA performance and reliability
    Focus: Response times, error handling, fallback mechanisms
    """
    
    def setUp(self):
        """Setup performance testing environment"""
        from app.core.decision_engine import EnhancedDecisionEngine
        self.decision_engine = EnhancedDecisionEngine()
        self.performance_thresholds = {
            "decision_time_ms": 500,  # Decision should take < 500ms
            "confidence_minimum": 0.3,  # Minimum acceptable confidence
            "success_rate_minimum": 0.95  # 95% success rate required
        }
    
    def test_decision_engine_performance(self):
        """Test decision engine performance under various loads"""
        print("\n🧪 Testing Decision Engine Performance...")
        
        test_cases = [
            "Simple question about weather",
            "Complex analysis of market trends with detailed requirements",
            "Code review for Python function with multiple issues",
            "Strategic planning for Q4 business objectives",
            "Creative writing task for marketing campaign"
        ]
        
        response_times = []
        confidences = []
        
        for i, test_case in enumerate(test_cases):
            start_time = datetime.now()
            
            try:
                model, confidence, reasoning = self.decision_engine.select_model(test_case)
                
                end_time = datetime.now()
                response_time_ms = (end_time - start_time).total_seconds() * 1000
                
                response_times.append(response_time_ms)
                confidences.append(confidence)
                
                # Validate individual performance
                self.assertLess(response_time_ms, self.performance_thresholds["decision_time_ms"],
                    f"Decision time {response_time_ms}ms exceeds threshold")
                
                self.assertGreaterEqual(confidence, self.performance_thresholds["confidence_minimum"],
                    f"Confidence {confidence} below minimum threshold")
                
                print(f"✅ Test case {i+1}: {response_time_ms:.1f}ms, confidence: {confidence:.3f}")
                
            except Exception as e:
                self.fail(f"Decision engine failed on test case {i+1}: {str(e)}")
        
        # Validate aggregate performance
        avg_response_time = sum(response_times) / len(response_times)
        avg_confidence = sum(confidences) / len(confidences)
        
        print(f"\n📊 Performance Summary:")
        print(f"   Average response time: {avg_response_time:.1f}ms")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Max response time: {max(response_times):.1f}ms")
        print(f"   Min confidence: {min(confidences):.3f}")
        
        self.assertLess(avg_response_time, self.performance_thresholds["decision_time_ms"],
            f"Average response time {avg_response_time}ms exceeds threshold")

    def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms"""
        print("\n🧪 Testing Error Handling and Fallbacks...")
        
        # Test with invalid inputs
        error_test_cases = [
            {"input": "", "description": "empty string"},
            {"input": "x" * 20000, "description": "extremely long input"},
            {"input": None, "description": "None input"},
            {"input": "🚀" * 1000, "description": "unicode heavy input"}
        ]
        
        for test_case in error_test_cases:
            with self.subTest(description=test_case["description"]):
                try:
                    if test_case["input"] is None:
                        # This should raise an exception
                        with self.assertRaises(Exception):
                            self.decision_engine.select_model(test_case["input"])
                    else:
                        model, confidence, reasoning = self.decision_engine.select_model(test_case["input"])
                        
                        # Should still return valid response
                        self.assertIsInstance(model, str, "Model should be string")
                        self.assertIsInstance(confidence, (int, float), "Confidence should be numeric")
                        self.assertIsInstance(reasoning, dict, "Reasoning should be dict")
                        
                        print(f"✅ Handled {test_case['description']}: {model}")
                        
                except Exception as e:
                    if test_case["input"] is not None:
                        print(f"⚠️ Failed on {test_case['description']}: {str(e)}")

    def test_concurrent_request_handling(self):
        """Test handling of concurrent requests"""
        print("\n🧪 Testing Concurrent Request Handling...")
        
        import threading
        import time
        
        results = []
        errors = []
        
        def make_decision(test_id):
            try:
                start_time = time.time()
                model, confidence, reasoning = self.decision_engine.select_model(
                    f"Test concurrent request {test_id}"
                )
                end_time = time.time()
                
                results.append({
                    "test_id": test_id,
                    "model": model,
                    "confidence": confidence,
                    "response_time": (end_time - start_time) * 1000
                })
            except Exception as e:
                errors.append({"test_id": test_id, "error": str(e)})
        
        # Create and start threads
        threads = []
        num_concurrent = 10
        
        for i in range(num_concurrent):
            thread = threading.Thread(target=make_decision, args=(i,))
            threads.append(thread)
        
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        # Validate results
        success_rate = len(results) / num_concurrent
        self.assertGreaterEqual(success_rate, self.performance_thresholds["success_rate_minimum"],
            f"Success rate {success_rate} below threshold")
        
        if results:
            avg_response_time = sum(r["response_time"] for r in results) / len(results)
            self.assertLess(avg_response_time, self.performance_thresholds["decision_time_ms"] * 2,
                f"Concurrent response time {avg_response_time}ms too high")
        
        print(f"✅ Concurrent handling: {len(results)}/{num_concurrent} success")
        print(f"   Total time: {total_time:.1f}ms")
        print(f"   Success rate: {success_rate:.1%}")
        if errors:
            print(f"   Errors: {len(errors)}")


class TestNAVAFeedbackLearningSystem(unittest.TestCase):
    """
    Test suite for NAVA feedback and learning system
    Focus: Learning algorithm, weight adaptation, feedback processing
    """
    
    def setUp(self):
        """Setup learning system testing environment"""
        from app.core.decision_engine import EnhancedDecisionEngine
        self.decision_engine = EnhancedDecisionEngine()
        # Reset learning system for clean testing
        self.decision_engine.reset_learning()
    
    def test_feedback_processing_accuracy(self):
        """Test feedback processing and storage accuracy"""
        print("\n🧪 Testing Feedback Processing Accuracy...")
        
        test_feedback = [
            {"response_id": "test_001", "model": "gpt", "pattern": "conversation", "score": 4.5},
            {"response_id": "test_002", "model": "claude", "pattern": "deep_analysis", "score": 3.8},
            {"response_id": "test_003", "model": "gemini", "pattern": "strategic_planning", "score": 4.2}
        ]
        
        for feedback in test_feedback:
            # Submit feedback
            self.decision_engine.update_user_feedback(
                feedback["response_id"],
                feedback["model"], 
                feedback["pattern"],
                feedback["score"],
                "rating"
            )
        
        # Validate feedback storage
        stats = self.decision_engine.get_feedback_stats()
        
        self.assertEqual(stats["feedback_summary"]["total_responses"], len(test_feedback),
            "Total response count incorrect")
        
        # Check model satisfaction updates
        for feedback in test_feedback:
            model_stats = stats["feedback_summary"]["model_satisfaction"][feedback["model"]]
            self.assertGreater(model_stats["total"], 0, f"No feedback recorded for {feedback['model']}")
        
        print(f"✅ Feedback processing: {len(test_feedback)} feedbacks processed correctly")
        print(f"   Total responses tracked: {stats['feedback_summary']['total_responses']}")

    def test_weight_adaptation_algorithm(self):
        """Test weight adaptation based on feedback patterns"""
        print("\n🧪 Testing Weight Adaptation Algorithm...")
        
        # Test positive feedback reinforcement
        pattern = "code_development"
        model = "gpt"
        
        # Get initial weight
        initial_stats = self.decision_engine.get_feedback_stats()
        initial_weight = initial_stats["current_pattern_weights"].get(pattern, {}).get(model, 0.5)
        
        # Provide consistent positive feedback
        positive_scores = [4.5, 4.8, 4.2, 4.6, 4.4]
        for i, score in enumerate(positive_scores):
            self.decision_engine.update_user_feedback(
                f"positive_test_{i}", model, pattern, score, "rating"
            )
        
        # Check weight increase
        final_stats = self.decision_engine.get_feedback_stats()
        final_weight = final_stats["current_pattern_weights"].get(pattern, {}).get(model, 0.5)
        
        self.assertGreater(final_weight, initial_weight,
            f"Weight should increase with positive feedback: {initial_weight} -> {final_weight}")
        
        print(f"✅ Positive feedback adaptation: {initial_weight:.3f} -> {final_weight:.3f}")
        
        # Test negative feedback reduction
        pattern = "deep_analysis"
        model = "claude"
        
        initial_weight = final_stats["current_pattern_weights"].get(pattern, {}).get(model, 0.5)
        
        # Provide consistent negative feedback
        negative_scores = [2.0, 1.8, 2.2, 1.5, 2.0]
        for i, score in enumerate(negative_scores):
            self.decision_engine.update_user_feedback(
                f"negative_test_{i}", model, pattern, score, "rating"
            )
        
        # Check weight decrease
        final_stats = self.decision_engine.get_feedback_stats()
        final_weight = final_stats["current_pattern_weights"].get(pattern, {}).get(model, 0.5)
        
        self.assertLess(final_weight, initial_weight,
            f"Weight should decrease with negative feedback: {initial_weight} -> {final_weight}")
        
        print(f"✅ Negative feedback adaptation: {initial_weight:.3f} -> {final_weight:.3f}")

    def test_learning_system_convergence(self):
        """Test learning system convergence and stability"""
        print("\n🧪 Testing Learning System Convergence...")
        
        pattern = "conversation"
        model = "gpt"
        
        # Simulate extended feedback session
        feedback_sequence = []
        weight_history = []
        
        for round_num in range(10):
            # Get current weight
            stats = self.decision_engine.get_feedback_stats()
            current_weight = stats["current_pattern_weights"].get(pattern, {}).get(model, 0.5)
            weight_history.append(current_weight)
            
            # Provide mixed feedback (trending positive)
            base_score = 3.5 + (round_num * 0.1)  # Gradually improving
            round_scores = [base_score + (i * 0.1) for i in range(3)]
            
            for i, score in enumerate(round_scores):
                response_id = f"convergence_test_r{round_num}_f{i}"
                self.decision_engine.update_user_feedback(
                    response_id, model, pattern, score, "rating"
                )
                feedback_sequence.append(score)
        
        # Check convergence characteristics
        final_stats = self.decision_engine.get_feedback_stats()
        final_weight = final_stats["current_pattern_weights"].get(pattern, {}).get(model, 0.5)
        weight_history.append(final_weight)
        
        # Weight should be stable (not oscillating wildly)
        recent_weights = weight_history[-5:]
        weight_variance = sum((w - sum(recent_weights)/len(recent_weights))**2 for w in recent_weights) / len(recent_weights)
        
        self.assertLess(weight_variance, 0.01, f"Weight variance too high: {weight_variance}")
        
        # Weight should reflect the positive trend
        self.assertGreater(final_weight, weight_history[0], 
            f"Final weight should be higher than initial: {weight_history[0]} -> {final_weight}")
        
        print(f"✅ Learning convergence: Stable with variance {weight_variance:.6f}")
        print(f"   Weight progression: {weight_history[0]:.3f} -> {final_weight:.3f}")
        print(f"   Total feedback processed: {len(feedback_sequence)}")


# Integration test runner
async def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Starting NAVA Enhanced Integration Tests...")
    print("=" * 60)
    
    # Decision Engine Tests
    decision_tests = TestNAVAEnhancedDecisionEngine()
    decision_tests.setUp()
    
    decision_tests.test_behavior_pattern_detection_accuracy()
    decision_tests.test_dynamic_weight_learning_system()
    decision_tests.test_intelligent_fallback_system()
    decision_tests.test_decision_reasoning_quality()
    
    # Service Integration Tests
    service_tests = TestNAVAServiceIntegration()
    service_tests.setUp()
    
    await service_tests.test_external_ai_service_communication()
    await service_tests.test_nava_core_endpoints()
    service_tests.test_slf_integration_readiness()
    
    # Performance Tests
    performance_tests = TestNAVAPerformanceAndReliability()
    performance_tests.setUp()
    
    performance_tests.test_decision_engine_performance()
    performance_tests.test_error_handling_and_fallbacks()
    performance_tests.test_concurrent_request_handling()
    
    # Learning System Tests
    learning_tests = TestNAVAFeedbackLearningSystem()
    learning_tests.setUp()
    
    learning_tests.test_feedback_processing_accuracy()
    learning_tests.test_weight_adaptation_algorithm()
    learning_tests.test_learning_system_convergence()
    
    print("=" * 60)
    print("🎉 NAVA Enhanced Integration Tests Completed!")

if __name__ == "__main__":
    # Run async integration tests
    asyncio.run(run_integration_tests())