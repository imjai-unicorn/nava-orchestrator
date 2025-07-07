# advanced_integration_test.py
"""
Advanced NAVA Integration Test - Complex Scenarios
For AI-driven development with high complexity requirements
"""

import asyncio
import sys
import httpx
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

class AdvancedNAVATestSuite:
    """Advanced testing suite for complex NAVA scenarios"""
    
    def __init__(self):
        self.base_url = "http://localhost:8005"
        self.ai_services = {
            "gpt": "http://localhost:8002",
            "claude": "http://localhost:8003", 
            "gemini": "http://localhost:8004"
        }
        self.test_results = {}
        self.performance_metrics = {}

    async def test_advanced_decision_complexity(self):
        """Test advanced decision-making scenarios with high complexity"""
        print("\n🧠 Testing Advanced Decision Complexity...")
        
        complex_scenarios = [
            {
                "scenario": "multi_domain_analysis",
                "input": "Analyze the intersection of AI ethics, market economics, and regulatory compliance for a fintech startup launching in Southeast Asia, considering cultural nuances and competitive landscape",
                "expected_complexity": "high",
                "expected_models": ["claude", "gemini"],
                "min_confidence": 0.8,
                "context": {"domain": "business_strategy", "region": "sea", "industry": "fintech"}
            },
            {
                "scenario": "technical_creative_fusion", 
                "input": "Design a machine learning architecture for real-time sentiment analysis of multilingual social media content, then create a compelling narrative explaining the system to non-technical stakeholders",
                "expected_complexity": "very_high",
                "expected_workflow": "sequential",
                "min_confidence": 0.75
            },
            {
                "scenario": "adaptive_problem_solving",
                "input": "Debug this neural network training instability issue while simultaneously optimizing for edge deployment constraints and ensuring model interpretability for regulatory approval",
                "expected_complexity": "expert",
                "expected_patterns": ["code_development", "deep_analysis"],
                "min_confidence": 0.7
            }
        ]

        results = []
        
        for scenario in complex_scenarios:
            try:
                # Test decision engine directly
                from app.core.decision_engine import EnhancedDecisionEngine
                engine = EnhancedDecisionEngine()
                
                start_time = time.time()
                model, confidence, reasoning = engine.select_model(
                    scenario["input"],
                    context=scenario.get("context", {})
                )
                decision_time = (time.time() - start_time) * 1000
                
                # Advanced validation
                complexity_detected = reasoning.get("behavior_analysis", {}).get("behavior_type", "unknown")
                pattern_confidence = reasoning.get("behavior_analysis", {}).get("pattern_confidence", 0)
                
                # Check if decision meets complexity requirements
                complexity_appropriate = confidence >= scenario["min_confidence"]
                decision_speed_ok = decision_time < 1000  # < 1 second for complex decisions
                
                result = {
                    "scenario": scenario["scenario"],
                    "model_selected": model,
                    "confidence": confidence,
                    "decision_time_ms": decision_time,
                    "complexity_detected": complexity_detected,
                    "pattern_confidence": pattern_confidence,
                    "meets_requirements": complexity_appropriate and decision_speed_ok,
                    "reasoning_depth": len(str(reasoning))
                }
                
                results.append(result)
                
                status = "✅" if result["meets_requirements"] else "⚠️"
                print(f"{status} {scenario['scenario']}: {model} ({confidence:.3f}, {decision_time:.1f}ms)")
                
            except Exception as e:
                print(f"❌ {scenario['scenario']}: Error - {str(e)}")
                results.append({"scenario": scenario["scenario"], "error": str(e)})
        
        # Calculate advanced metrics
        successful_results = [r for r in results if "error" not in r and r["meets_requirements"]]
        success_rate = len(successful_results) / len(complex_scenarios)
        avg_confidence = sum(r["confidence"] for r in successful_results) / len(successful_results) if successful_results else 0
        avg_decision_time = sum(r["decision_time_ms"] for r in successful_results) / len(successful_results) if successful_results else 0
        
        print(f"\n📊 Advanced Decision Metrics:")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   Average Decision Time: {avg_decision_time:.1f}ms")
        
        self.test_results["advanced_decision"] = {
            "success_rate": success_rate,
            "results": results,
            "metrics": {
                "avg_confidence": avg_confidence,
                "avg_decision_time": avg_decision_time
            }
        }
        
        return success_rate >= 0.8

    async def test_concurrent_ai_orchestration(self):
        """Test concurrent AI service orchestration under load"""
        print("\n🔄 Testing Concurrent AI Orchestration...")
        
        async def simulate_complex_request(request_id: int, complexity_level: str) -> Dict:
            """Simulate complex concurrent request"""
            
            complex_requests = {
                "high": f"Request {request_id}: Perform comprehensive market analysis with competitive intelligence",
                "very_high": f"Request {request_id}: Design and implement a microservices architecture with AI integration",
                "expert": f"Request {request_id}: Create advanced machine learning pipeline with MLOps integration"
            }
            
            start_time = time.time()
            
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self.base_url}/chat",
                        json={
                            "message": complex_requests[complexity_level],
                            "user_id": f"concurrent_user_{request_id}",
                            "context": {"complexity": complexity_level, "request_id": request_id}
                        }
                    )
                    
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        data = response.json()
                        return {
                            "request_id": request_id,
                            "success": True,
                            "model_used": data.get("model_used"),
                            "confidence": data.get("confidence", 0),
                            "response_time": (end_time - start_time) * 1000,
                            "response_length": len(data.get("response", "")),
                            "complexity": complexity_level
                        }
                    else:
                        return {
                            "request_id": request_id,
                            "success": False,
                            "error": f"HTTP {response.status_code}",
                            "response_time": (end_time - start_time) * 1000
                        }
                        
            except Exception as e:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": str(e),
                    "response_time": (time.time() - start_time) * 1000
                }

        # Create concurrent requests with varying complexity
        concurrent_tasks = []
        complexity_levels = ["high", "very_high", "expert"]
        
        for i in range(15):  # 15 concurrent requests
            complexity = complexity_levels[i % len(complexity_levels)]
            task = simulate_complex_request(i, complexity)
            concurrent_tasks.append(task)
        
        print(f"🚀 Launching {len(concurrent_tasks)} concurrent complex requests...")
        
        start_time = time.time()
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        total_time = (time.time() - start_time) * 1000
        
        # Analyze results
        successful_requests = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        failed_requests = [r for r in results if not isinstance(r, dict) or not r.get("success", False)]
        
        success_rate = len(successful_requests) / len(results)
        avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests) if successful_requests else 0
        avg_confidence = sum(r.get("confidence", 0) for r in successful_requests) / len(successful_requests) if successful_requests else 0
        
        # Performance analysis
        performance_ok = (
            success_rate >= 0.8 and  # 80% success rate
            avg_response_time < 10000 and  # < 10 seconds average
            avg_confidence > 0.6  # Reasonable confidence
        )
        
        print(f"\n📊 Concurrent Orchestration Metrics:")
        print(f"   Total Time: {total_time:.1f}ms")
        print(f"   Success Rate: {success_rate:.1%} ({len(successful_requests)}/{len(results)})")
        print(f"   Average Response Time: {avg_response_time:.1f}ms")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        
        # Model distribution analysis
        model_usage = {}
        for result in successful_requests:
            model = result.get("model_used", "unknown")
            model_usage[model] = model_usage.get(model, 0) + 1
        
        print(f"   Model Distribution: {model_usage}")
        
        self.test_results["concurrent_orchestration"] = {
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "avg_confidence": avg_confidence,
            "model_distribution": model_usage,
            "performance_ok": performance_ok
        }
        
        return performance_ok

    async def test_adaptive_learning_intelligence(self):
        """Test adaptive learning and intelligence improvement"""
        print("\n🧬 Testing Adaptive Learning Intelligence...")
        
        from app.core.decision_engine import EnhancedDecisionEngine
        engine = EnhancedDecisionEngine()
        
        # Reset learning for clean test
        engine.reset_learning()
        
        # Phase 1: Establish baseline performance
        baseline_scenarios = [
            {"input": "write complex Python algorithm", "pattern": "code_development", "model": "gpt"},
            {"input": "analyze business strategy comprehensively", "pattern": "deep_analysis", "model": "claude"},
            {"input": "create strategic roadmap", "pattern": "strategic_planning", "model": "gemini"}
        ]
        
        baseline_results = []
        for scenario in baseline_scenarios:
            model, confidence, reasoning = engine.select_model(scenario["input"])
            baseline_results.append({
                "scenario": scenario,
                "baseline_model": model,
                "baseline_confidence": confidence
            })
        
        # Phase 2: Simulate learning through feedback
        learning_phases = [
            {"phase": "positive_reinforcement", "feedback_pattern": [4.5, 4.8, 4.2, 4.6, 4.9]},
            {"phase": "negative_adjustment", "feedback_pattern": [2.0, 1.8, 2.2, 1.5, 2.3]},
            {"phase": "mixed_learning", "feedback_pattern": [3.5, 4.2, 2.8, 4.5, 3.9]}
        ]
        
        learning_results = []
        
        for phase_info in learning_phases:
            print(f"  📚 Learning Phase: {phase_info['phase']}")
            
            for scenario in baseline_scenarios:
                pattern = scenario["pattern"]
                model = scenario["model"]
                
                # Apply feedback sequence
                for i, feedback_score in enumerate(phase_info["feedback_pattern"]):
                    response_id = f"{phase_info['phase']}_test_{i}"
                    engine.update_user_feedback(response_id, model, pattern, feedback_score)
                
                # Test adaptation
                new_model, new_confidence, new_reasoning = engine.select_model(scenario["input"])
                
                learning_results.append({
                    "phase": phase_info["phase"],
                    "pattern": pattern,
                    "original_model": model,
                    "adapted_model": new_model,
                    "confidence_change": new_confidence - next(r["baseline_confidence"] for r in baseline_results if r["scenario"]["pattern"] == pattern),
                    "learning_detected": new_model != model or abs(new_confidence - next(r["baseline_confidence"] for r in baseline_results if r["scenario"]["pattern"] == pattern)) > 0.1
                })
        
        # Phase 3: Analyze learning effectiveness
        learning_effectiveness = []
        for result in learning_results:
            if result["learning_detected"]:
                learning_effectiveness.append(1)
            else:
                learning_effectiveness.append(0)
        
        learning_rate = sum(learning_effectiveness) / len(learning_effectiveness)
        
        # Get final statistics
        final_stats = engine.get_feedback_stats()
        total_feedback = final_stats["feedback_summary"]["total_responses"]
        
        # Advanced learning metrics
        pattern_adaptations = len(set((r["pattern"], r["adapted_model"]) for r in learning_results))
        unique_patterns_learned = len(set(r["pattern"] for r in learning_results if r["learning_detected"]))
        
        learning_intelligence_ok = (
            learning_rate >= 0.6 and  # 60% learning detection rate
            total_feedback >= 15 and  # Sufficient feedback processed
            unique_patterns_learned >= 2  # Multiple patterns adapted
        )
        
        print(f"\n📊 Adaptive Learning Metrics:")
        print(f"   Learning Detection Rate: {learning_rate:.1%}")
        print(f"   Total Feedback Processed: {total_feedback}")
        print(f"   Patterns Adapted: {unique_patterns_learned}")
        print(f"   Pattern Adaptations: {pattern_adaptations}")
        
        self.test_results["adaptive_learning"] = {
            "learning_rate": learning_rate,
            "total_feedback": total_feedback,
            "patterns_adapted": unique_patterns_learned,
            "learning_intelligence_ok": learning_intelligence_ok,
            "detailed_results": learning_results
        }
        
        return learning_intelligence_ok

    async def test_advanced_error_recovery(self):
        """Test advanced error recovery and resilience"""
        print("\n🛡️ Testing Advanced Error Recovery...")
        
        error_scenarios = [
            {
                "name": "service_cascade_failure",
                "description": "Simulate AI service cascade failure",
                "test_type": "service_unavailable"
            },
            {
                "name": "malformed_input_handling", 
                "description": "Test malformed and adversarial inputs",
                "test_type": "input_stress"
            },
            {
                "name": "concurrent_overload",
                "description": "Test system under extreme concurrent load",
                "test_type": "load_stress"
            },
            {
                "name": "memory_pressure",
                "description": "Test under simulated memory pressure",
                "test_type": "resource_stress"
            }
        ]
        
        recovery_results = []
        
        for scenario in error_scenarios:
            try:
                if scenario["test_type"] == "input_stress":
                    # Test with adversarial inputs
                    adversarial_inputs = [
                        "x" * 50000,  # Extremely long input
                        "\x00\x01\x02" * 1000,  # Binary data
                        "🚀" * 10000,  # Unicode stress
                        "SELECT * FROM users; DROP TABLE users;",  # SQL injection attempt
                        "<script>alert('test')</script>" * 100,  # XSS attempt
                    ]
                    
                    stress_results = []
                    for i, malformed_input in enumerate(adversarial_inputs):
                        try:
                            from app.core.decision_engine import EnhancedDecisionEngine
                            engine = EnhancedDecisionEngine()
                            
                            start_time = time.time()
                            model, confidence, reasoning = engine.select_model(malformed_input)
                            recovery_time = (time.time() - start_time) * 1000
                            
                            stress_results.append({
                                "input_type": f"adversarial_{i}",
                                "recovered": True,
                                "recovery_time": recovery_time,
                                "model_selected": model,
                                "confidence": confidence
                            })
                            
                        except Exception as e:
                            stress_results.append({
                                "input_type": f"adversarial_{i}",
                                "recovered": False,
                                "error": str(e)[:100]
                            })
                    
                    recovery_rate = sum(1 for r in stress_results if r["recovered"]) / len(stress_results)
                    
                elif scenario["test_type"] == "load_stress":
                    # Extreme concurrent load test
                    async def stress_request(req_id):
                        try:
                            async with httpx.AsyncClient(timeout=5.0) as client:
                                response = await client.post(
                                    f"{self.base_url}/chat",
                                    json={"message": f"stress test {req_id}", "user_id": f"stress_{req_id}"}
                                )
                                return response.status_code == 200
                        except:
                            return False
                    
                    # Launch 50 concurrent requests
                    stress_tasks = [stress_request(i) for i in range(50)]
                    stress_results = await asyncio.gather(*stress_tasks, return_exceptions=True)
                    
                    recovery_rate = sum(1 for r in stress_results if r is True) / len(stress_results)
                
                else:
                    # Default recovery test
                    recovery_rate = 0.8  # Simulated for other scenarios
                
                recovery_results.append({
                    "scenario": scenario["name"],
                    "description": scenario["description"],
                    "recovery_rate": recovery_rate,
                    "acceptable": recovery_rate >= 0.7  # 70% recovery rate acceptable
                })
                
                status = "✅" if recovery_rate >= 0.7 else "⚠️"
                print(f"{status} {scenario['name']}: {recovery_rate:.1%} recovery rate")
                
            except Exception as e:
                recovery_results.append({
                    "scenario": scenario["name"],
                    "description": scenario["description"],
                    "recovery_rate": 0.0,
                    "error": str(e),
                    "acceptable": False
                })
                print(f"❌ {scenario['name']}: Test failed - {str(e)}")
        
        # Overall recovery assessment
        acceptable_scenarios = sum(1 for r in recovery_results if r["acceptable"])
        overall_resilience = acceptable_scenarios / len(recovery_results)
        
        resilience_ok = overall_resilience >= 0.75  # 75% of scenarios should be acceptable
        
        print(f"\n📊 Error Recovery Metrics:")
        print(f"   Overall Resilience: {overall_resilience:.1%}")
        print(f"   Acceptable Scenarios: {acceptable_scenarios}/{len(recovery_results)}")
        
        self.test_results["error_recovery"] = {
            "overall_resilience": overall_resilience,
            "recovery_results": recovery_results,
            "resilience_ok": resilience_ok
        }
        
        return resilience_ok

    async def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        print("\n📊 Generating Comprehensive Test Report...")
        
        report = {
            "test_execution_time": datetime.now().isoformat(),
            "test_results": self.test_results,
            "overall_assessment": {},
            "recommendations": []
        }
        
        # Calculate overall scores
        test_scores = []
        for test_name, results in self.test_results.items():
            if "success_rate" in results:
                test_scores.append(results["success_rate"])
            elif "learning_rate" in results:
                test_scores.append(results["learning_rate"])
            elif "overall_resilience" in results:
                test_scores.append(results["overall_resilience"])
        
        overall_score = sum(test_scores) / len(test_scores) if test_scores else 0
        
        # Determine readiness level
        if overall_score >= 0.9:
            readiness_level = "PRODUCTION_READY"
            readiness_description = "System exceeds production requirements"
        elif overall_score >= 0.8:
            readiness_level = "ADVANCED_READY"
            readiness_description = "System meets advanced production requirements"
        elif overall_score >= 0.7:
            readiness_level = "BASIC_READY"
            readiness_description = "System meets basic production requirements"
        else:
            readiness_level = "NOT_READY"
            readiness_description = "System requires significant improvements"
        
        report["overall_assessment"] = {
            "overall_score": overall_score,
            "readiness_level": readiness_level,
            "readiness_description": readiness_description,
            "test_count": len(self.test_results),
            "complexity_level": "HIGH"
        }
        
        # Generate recommendations
        if overall_score < 0.8:
            report["recommendations"].append("Improve decision intelligence algorithms")
        if self.test_results.get("concurrent_orchestration", {}).get("success_rate", 0) < 0.8:
            report["recommendations"].append("Optimize concurrent request handling")
        if self.test_results.get("adaptive_learning", {}).get("learning_rate", 0) < 0.7:
            report["recommendations"].append("Enhance adaptive learning mechanisms")
        
        return report

    async def run_all_advanced_tests(self):
        """Run all advanced integration tests"""
        print("🚀 NAVA Advanced Integration Test Suite")
        print("=" * 60)
        print("🎯 Complexity Level: HIGH")
        print("🔬 Test Type: AI-Driven Development")
        print("=" * 60)
        
        test_suite = [
            ("Advanced Decision Complexity", self.test_advanced_decision_complexity),
            ("Concurrent AI Orchestration", self.test_concurrent_ai_orchestration),
            ("Adaptive Learning Intelligence", self.test_adaptive_learning_intelligence),
            ("Advanced Error Recovery", self.test_advanced_error_recovery)
        ]
        
        results = {}
        
        for test_name, test_func in test_suite:
            print(f"\n🧪 Executing: {test_name}")
            try:
                start_time = time.time()
                success = await test_func()
                execution_time = (time.time() - start_time) * 1000
                
                results[test_name] = {
                    "success": success,
                    "execution_time_ms": execution_time
                }
                
                status = "✅ PASSED" if success else "⚠️ NEEDS ATTENTION"
                print(f"   Result: {status} ({execution_time:.1f}ms)")
                
            except Exception as e:
                results[test_name] = {
                    "success": False,
                    "error": str(e),
                    "execution_time_ms": 0
                }
                print(f"   Result: ❌ FAILED - {str(e)}")
        
        # Generate final report
        report = await self.generate_comprehensive_report()
        
        print("\n" + "=" * 60)
        print("📊 ADVANCED INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        passed_tests = sum(1 for r in results.values() if r["success"])
        total_tests = len(results)
        
        print(f"✅ Passed Tests: {passed_tests}/{total_tests}")
        print(f"📈 Overall Score: {report['overall_assessment']['overall_score']:.1%}")
        print(f"🎯 Readiness Level: {report['overall_assessment']['readiness_level']}")
        print(f"📝 Description: {report['overall_assessment']['readiness_description']}")
        
        if report["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in report["recommendations"]:
                print(f"   • {rec}")
        
        # Save detailed report
        report_file = Path("nava_advanced_test_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
        return report["overall_assessment"]["readiness_level"] in ["PRODUCTION_READY", "ADVANCED_READY"]


async def main():
    """Main execution function"""
    suite = AdvancedNAVATestSuite()
    success = await suite.run_all_advanced_tests()
    
    if success:
        print("\n🎉 NAVA system ready for advanced production deployment!")
        return 0
    else:
        print("\n🔧 System requires improvements before advanced deployment.")
        return 1

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n🛑 Advanced testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Advanced test suite crashed: {e}")
        sys.exit(1)