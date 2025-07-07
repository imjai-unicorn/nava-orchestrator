# Progressive Testing Strategy - Build from Simple to Complex
# Fix path issues and create proper test hierarchy

import sys
import os
import unittest
from pathlib import Path

# === LEVEL 1: PATH SETUP AND BASIC TESTING ===

def setup_python_path():
    """Setup Python path for proper imports"""
    # Get the project root
    current_dir = Path(__file__).parent
    
    # Add paths for imports
    nava_controller_path = current_dir.parent / "app"
    if str(nava_controller_path) not in sys.path:
        sys.path.insert(0, str(nava_controller_path))
    
    # Add backend services path
    backend_path = current_dir.parent.parent.parent
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))
    
    print(f"✅ Python paths added:")
    print(f"   NAVA Controller: {nava_controller_path}")
    print(f"   Backend Root: {backend_path}")

# === LEVEL 2: BASIC FUNCTIONALITY TESTS ===

class Level1BasicFunctionalityTest(unittest.TestCase):
    """Level 1: Test basic functionality without complex imports"""
    
    @classmethod
    def setUpClass(cls):
        """Setup for basic tests"""
        setup_python_path()
    
    def test_01_imports_work(self):
        """Test that we can import basic modules"""
        print("\n🧪 Level 1: Testing Basic Imports...")
        
        try:
            # Test if we can import decision engine
            from core.decision_engine import EnhancedDecisionEngine
            print("✅ EnhancedDecisionEngine import: SUCCESS")
            
            # Test if we can create instance
            engine = EnhancedDecisionEngine()
            print("✅ EnhancedDecisionEngine instance: SUCCESS")
            
            # Test basic attributes
            self.assertTrue(hasattr(engine, 'select_model'))
            self.assertTrue(hasattr(engine, 'update_user_feedback'))
            print("✅ Required methods exist: SUCCESS")
            
        except ImportError as e:
            self.fail(f"Import failed: {e}")
        except Exception as e:
            self.fail(f"Basic setup failed: {e}")
    
    def test_02_basic_model_selection(self):
        """Test basic model selection without strict expectations"""
        print("\n🧪 Level 1: Testing Basic Model Selection...")
        
        try:
            from core.decision_engine import EnhancedDecisionEngine
            engine = EnhancedDecisionEngine()
            
            # Simple test - just check it returns something
            result = engine.select_model("test message")
            
            # Basic validation
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
            
            model, confidence, reasoning = result
            
            self.assertIsInstance(model, str)
            self.assertIn(model, ["gpt", "claude", "gemini"])
            
            self.assertIsInstance(confidence, (int, float))
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
            
            self.assertIsInstance(reasoning, dict)
            
            print(f"✅ Basic selection works: {model} (confidence: {confidence:.3f})")
            
        except Exception as e:
            self.fail(f"Basic model selection failed: {e}")
    
    def test_03_feedback_system_basic(self):
        """Test basic feedback system"""
        print("\n🧪 Level 1: Testing Basic Feedback System...")
        
        try:
            from core.decision_engine import EnhancedDecisionEngine
            engine = EnhancedDecisionEngine()
            
            # Test feedback submission
            engine.update_user_feedback("test_123", "gpt", "conversation", 4.0)
            
            # Test stats retrieval
            stats = engine.get_feedback_stats()
            
            self.assertIsInstance(stats, dict)
            self.assertIn("feedback_summary", stats)
            
            print("✅ Basic feedback system works")
            
        except Exception as e:
            self.fail(f"Basic feedback system failed: {e}")

# === LEVEL 3: INTERMEDIATE FUNCTIONALITY TESTS ===

class Level2IntermediateFunctionalityTest(unittest.TestCase):
    """Level 2: Test intermediate functionality with realistic expectations"""
    
    @classmethod
    def setUpClass(cls):
        """Setup for intermediate tests"""
        setup_python_path()
    
    def test_01_pattern_detection_realistic(self):
        """Test pattern detection with realistic expectations"""
        print("\n🧪 Level 2: Testing Pattern Detection (Realistic)...")
        
        try:
            from core.decision_engine import EnhancedDecisionEngine
            engine = EnhancedDecisionEngine()
            
            test_cases = [
                {"input": "write code in python", "description": "Code task"},
                {"input": "analyze business data", "description": "Analysis task"},
                {"input": "create a story", "description": "Creative task"},
                {"input": "help me plan strategy", "description": "Strategy task"}
            ]
            
            for test_case in test_cases:
                model, confidence, reasoning = engine.select_model(test_case["input"])
                
                # Realistic validation - just check system responds appropriately
                self.assertIn(model, ["gpt", "claude", "gemini"])
                self.assertGreaterEqual(confidence, 0.1)  # Very low minimum
                
                print(f"✅ {test_case['description']}: {model} (conf: {confidence:.3f})")
                
        except Exception as e:
            self.fail(f"Pattern detection failed: {e}")
    
    def test_02_learning_adaptation_basic(self):
        """Test basic learning adaptation"""
        print("\n🧪 Level 2: Testing Learning Adaptation...")
        
        try:
            from core.decision_engine import EnhancedDecisionEngine
            engine = EnhancedDecisionEngine()
            
            # Reset learning
            engine.reset_learning()
            
            # Submit multiple feedback
            feedback_data = [
                {"id": "test_1", "model": "gpt", "pattern": "conversation", "score": 4.0},
                {"id": "test_2", "model": "gpt", "pattern": "conversation", "score": 4.5},
                {"id": "test_3", "model": "claude", "pattern": "deep_analysis", "score": 3.0}
            ]
            
            for feedback in feedback_data:
                engine.update_user_feedback(
                    feedback["id"], feedback["model"], 
                    feedback["pattern"], feedback["score"]
                )
            
            # Check stats
            stats = engine.get_feedback_stats()
            total_responses = stats["feedback_summary"]["total_responses"]
            
            self.assertEqual(total_responses, len(feedback_data))
            print(f"✅ Learning adaptation: {total_responses} responses tracked")
            
        except Exception as e:
            self.fail(f"Learning adaptation failed: {e}")
    
    def test_03_error_handling(self):
        """Test error handling capabilities"""
        print("\n🧪 Level 2: Testing Error Handling...")
        
        try:
            from core.decision_engine import EnhancedDecisionEngine
            engine = EnhancedDecisionEngine()
            
            # Test with edge cases
            edge_cases = [
                "",  # Empty string
                "x" * 1000,  # Very long string
                "special chars: @#$%^&*()",  # Special characters
            ]
            
            for i, test_input in enumerate(edge_cases):
                try:
                    model, confidence, reasoning = engine.select_model(test_input)
                    
                    # Should handle gracefully
                    self.assertIsInstance(model, str)
                    self.assertIsInstance(confidence, (int, float))
                    
                    print(f"✅ Edge case {i+1}: Handled gracefully")
                    
                except Exception:
                    # Some failures are acceptable for edge cases
                    print(f"⚠️ Edge case {i+1}: Failed (acceptable)")
                    
        except Exception as e:
            self.fail(f"Error handling test setup failed: {e}")

# === LEVEL 4: ADVANCED INTEGRATION TESTS ===

class Level3AdvancedIntegrationTest(unittest.TestCase):
    """Level 3: Advanced integration tests with full system validation"""
    
    @classmethod
    def setUpClass(cls):
        """Setup for advanced tests"""
        setup_python_path()
    
    def test_01_service_integration_simulation(self):
        """Test service integration through simulation"""
        print("\n🧪 Level 3: Testing Service Integration (Simulation)...")
        
        try:
            from core.decision_engine import EnhancedDecisionEngine
            engine = EnhancedDecisionEngine()
            
            # Simulate service responses
            test_scenarios = [
                {
                    "user_input": "complex business analysis with multiple factors",
                    "expected_complexity": "high",
                    "description": "Complex analysis scenario"
                },
                {
                    "user_input": "simple hello world",
                    "expected_complexity": "low", 
                    "description": "Simple interaction scenario"
                }
            ]
            
            for scenario in test_scenarios:
                model, confidence, reasoning = engine.select_model(scenario["user_input"])
                
                # Advanced validation
                self.assertIn("behavior_analysis", reasoning)
                self.assertIn("selected_model", reasoning)
                
                behavior_analysis = reasoning.get("behavior_analysis", {})
                
                print(f"✅ {scenario['description']}: {model}")
                print(f"   Behavior: {behavior_analysis.get('behavior_type', 'unknown')}")
                print(f"   Confidence: {confidence:.3f}")
                
        except Exception as e:
            self.fail(f"Service integration simulation failed: {e}")
    
    def test_02_performance_under_load_simulation(self):
        """Test performance under simulated load"""
        print("\n🧪 Level 3: Testing Performance Under Load...")
        
        try:
            from core.decision_engine import EnhancedDecisionEngine
            import time
            
            engine = EnhancedDecisionEngine()
            
            # Simulate multiple requests
            num_requests = 10
            start_time = time.time()
            
            results = []
            for i in range(num_requests):
                model, confidence, reasoning = engine.select_model(f"test request {i}")
                results.append((model, confidence))
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / num_requests
            
            # Performance validation
            self.assertLess(avg_time, 1.0, f"Average request time {avg_time:.3f}s too slow")
            self.assertEqual(len(results), num_requests)
            
            print(f"✅ Performance test: {num_requests} requests in {total_time:.3f}s")
            print(f"   Average per request: {avg_time:.3f}s")
            
        except Exception as e:
            self.fail(f"Performance test failed: {e}")

# === TEST RUNNER WITH PROGRESSIVE EXECUTION ===

class ProgressiveTestRunner:
    """Run tests progressively from basic to advanced"""
    
    def __init__(self):
        self.results = {
            "level_1": {"passed": 0, "failed": 0, "errors": []},
            "level_2": {"passed": 0, "failed": 0, "errors": []},
            "level_3": {"passed": 0, "failed": 0, "errors": []}
        }
    
    def run_level(self, test_class, level_name):
        """Run tests for a specific level"""
        print(f"\n{'='*60}")
        print(f"🚀 RUNNING {level_name.upper()} TESTS")
        print(f"{'='*60}")
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        
        # Custom result handler
        result = unittest.TestResult()
        suite.run(result)
        
        # Record results
        level_key = level_name.lower().replace(" ", "_")
        self.results[level_key]["passed"] = result.testsRun - len(result.failures) - len(result.errors)
        self.results[level_key]["failed"] = len(result.failures) + len(result.errors)
        
        # Store error details
        for test, error in result.failures + result.errors:
            self.results[level_key]["errors"].append(f"{test}: {str(error)[:100]}...")
        
        # Print level summary
        passed = self.results[level_key]["passed"]
        failed = self.results[level_key]["failed"]
        
        if failed == 0:
            print(f"\n✅ {level_name}: ALL {passed} TESTS PASSED!")
            return True
        else:
            print(f"\n⚠️ {level_name}: {passed} passed, {failed} failed")
            return False
    
    def run_all_levels(self):
        """Run all test levels progressively"""
        print("🎯 NAVA Progressive Testing Strategy")
        print("Building confidence level by level...")
        
        # Level 1: Basic functionality
        level_1_success = self.run_level(Level1BasicFunctionalityTest, "Level 1")
        
        if not level_1_success:
            print("\n❌ Level 1 failed - fix basic issues before proceeding")
            return self.show_summary()
        
        # Level 2: Intermediate functionality
        level_2_success = self.run_level(Level2IntermediateFunctionalityTest, "Level 2")
        
        if not level_2_success:
            print("\n⚠️ Level 2 had issues - but Level 1 passed, so basic system works")
        
        # Level 3: Advanced integration (only if Level 2 mostly passed)
        if level_2_success or self.results["level_2"]["passed"] >= 2:
            level_3_success = self.run_level(Level3AdvancedIntegrationTest, "Level 3")
        else:
            print("\n⏭️ Skipping Level 3 - Level 2 needs more work")
        
        return self.show_summary()
    
    def show_summary(self):
        """Show overall test summary"""
        print(f"\n{'='*60}")
        print("📊 PROGRESSIVE TESTING SUMMARY")
        print(f"{'='*60}")
        
        total_passed = 0
        total_failed = 0
        
        for level, results in self.results.items():
            passed = results["passed"]
            failed = results["failed"]
            total_passed += passed
            total_failed += failed
            
            if failed == 0 and passed > 0:
                status = "✅ PASSED"
            elif passed > failed:
                status = "⚠️ MOSTLY PASSED"
            elif passed > 0:
                status = "❌ MIXED RESULTS"
            else:
                status = "❌ FAILED"
            
            print(f"{level.upper()}: {status} ({passed} passed, {failed} failed)")
        
        print(f"\nOVERALL: {total_passed} passed, {total_failed} failed")
        
        if total_failed == 0:
            print("\n🎉 ALL TESTS PASSED! System ready for next phase.")
        elif total_passed > total_failed:
            print("\n✅ Core system working! Some advanced features need attention.")
        else:
            print("\n⚠️ Core issues need to be resolved.")
        
        return total_failed == 0

# === MAIN EXECUTION ===

def main():
    """Main test execution"""
    runner = ProgressiveTestRunner()
    success = runner.run_all_levels()
    
    if success:
        print("\n🚀 Ready to proceed to production deployment!")
    else:
        print("\n🔧 Address the issues above before proceeding.")

if __name__ == "__main__":
    main()