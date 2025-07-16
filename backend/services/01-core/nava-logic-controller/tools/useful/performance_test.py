# backend/tests/performance_test.py
"""
Performance Test Script - Week 1 Validation
วัดผล performance improvement หลังจาก timeout fixes
เป้าหมาย: Response time จาก >5s → <3s P95
"""

import asyncio
import time
import statistics
import json
from typing import List, Dict, Any
import httpx
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceResult:
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    response_times: List[float]
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    average_response_time: float
    requests_per_second: float
    success_rate: float
    errors: List[str]

class NAVAPerformanceTester:
    """Comprehensive performance testing for NAVA system"""
    
    def __init__(self, base_url: str = "http://localhost:8005"):
        self.base_url = base_url
        self.test_results = []
        
        # Test scenarios
        self.test_scenarios = [
            {
                "name": "simple_chat",
                "message": "Hello, how are you?",
                "expected_model": "gpt",
                "complexity": "simple"
            },
            {
                "name": "code_generation", 
                "message": "Write a Python function to calculate factorial",
                "expected_model": "gpt",
                "complexity": "medium"
            },
            {
                "name": "business_analysis",
                "message": "Analyze the market trends for electric vehicles in 2024",
                "expected_model": "claude", 
                "complexity": "complex"
            },
            {
                "name": "research_query",
                "message": "Find recent developments in artificial intelligence research",
                "expected_model": "gemini",
                "complexity": "medium"
            },
            {
                "name": "strategic_planning",
                "message": "Create a 5-year strategic plan for a tech startup",
                "expected_model": "claude",
                "complexity": "critical"
            }
        ]
    
    async def test_single_request_performance(self) -> PerformanceResult:
        """Test individual request performance"""
        logger.info("🧪 Testing single request performance...")
        
        response_times = []
        errors = []
        successful = 0
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for scenario in self.test_scenarios:
                for attempt in range(3):  # 3 attempts per scenario
                    try:
                        start_time = time.time()
                        
                        response = await client.post(
                            f"{self.base_url}/chat",
                            json={
                                "message": scenario["message"],
                                "user_id": f"perf_test_{scenario['name']}_{attempt}",
                                "complexity": scenario["complexity"]
                            }
                        )
                        
                        end_time = time.time()
                        response_time = end_time - start_time
                        response_times.append(response_time)
                        
                        if response.status_code == 200:
                            successful += 1
                            data = response.json()
                            logger.info(f"✅ {scenario['name']} attempt {attempt+1}: {response_time:.2f}s")
                        else:
                            errors.append(f"{scenario['name']}: HTTP {response.status_code}")
                            logger.warning(f"❌ {scenario['name']} attempt {attempt+1}: HTTP {response.status_code}")
                        
                    except Exception as e:
                        errors.append(f"{scenario['name']}: {str(e)}")
                        logger.error(f"💥 {scenario['name']} attempt {attempt+1}: {str(e)}")
        
        return self._calculate_performance_result(
            "single_request_performance",
            len(self.test_scenarios) * 3,
            successful,
            response_times,
            errors
        )
    
    async def test_concurrent_requests(self, concurrent_users: int = 10) -> PerformanceResult:
        """Test concurrent request handling"""
        logger.info(f"🧪 Testing concurrent requests ({concurrent_users} users)...")
        
        response_times = []
        errors = []
        successful = 0
        
        async def single_user_test(user_id: int):
            """Single user making requests"""
            user_times = []
            user_errors = []
            user_successful = 0
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Each user makes 3 requests
                for i in range(3):
                    scenario = self.test_scenarios[i % len(self.test_scenarios)]
                    
                    try:
                        start_time = time.time()
                        
                        response = await client.post(
                            f"{self.base_url}/chat",
                            json={
                                "message": scenario["message"],
                                "user_id": f"concurrent_user_{user_id}_{i}",
                                "complexity": scenario.get("complexity", "medium")
                            }
                        )
                        
                        end_time = time.time()
                        response_time = end_time - start_time
                        user_times.append(response_time)
                        
                        if response.status_code == 200:
                            user_successful += 1
                        else:
                            user_errors.append(f"User {user_id}: HTTP {response.status_code}")
                        
                    except Exception as e:
                        user_errors.append(f"User {user_id}: {str(e)}")
            
            return user_times, user_errors, user_successful
        
        # Run concurrent users
        start_time = time.time()
        tasks = [single_user_test(user_id) for user_id in range(concurrent_users)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Aggregate results
        for result in results:
            if isinstance(result, Exception):
                errors.append(f"Task failed: {str(result)}")
            else:
                user_times, user_errors, user_successful = result
                response_times.extend(user_times)
                errors.extend(user_errors)
                successful += user_successful
        
        perf_result = self._calculate_performance_result(
            f"concurrent_{concurrent_users}_users",
            concurrent_users * 3,
            successful,
            response_times,
            errors
        )
        
        # Override RPS calculation for concurrent test
        perf_result.requests_per_second = len(response_times) / total_time if total_time > 0 else 0
        
        return perf_result
    
    async def test_ai_service_failover(self) -> PerformanceResult:
        """Test performance during AI service failover"""
        logger.info("🧪 Testing AI service failover performance...")
        
        response_times = []
        errors = []
        successful = 0
        
        # Test requests that should trigger different AI services
        failover_scenarios = [
            {"message": "Write Python code", "expected": "gpt"},
            {"message": "Analyze business data", "expected": "claude"},
            {"message": "Search for information", "expected": "gemini"},
            {"message": "General conversation", "expected": "any"}
        ]
        
        async with httpx.AsyncClient(timeout=45.0) as client:  # Longer timeout for failover
            for scenario in failover_scenarios:
                for attempt in range(5):  # More attempts to test failover
                    try:
                        start_time = time.time()
                        
                        response = await client.post(
                            f"{self.base_url}/chat",
                            json={
                                "message": scenario["message"],
                                "user_id": f"failover_test_{attempt}",
                                "test_failover": True
                            }
                        )
                        
                        end_time = time.time()
                        response_time = end_time - start_time
                        response_times.append(response_time)
                        
                        if response.status_code == 200:
                            successful += 1
                            data = response.json()
                            model_used = data.get("model_used", "unknown")
                            logger.info(f"✅ Failover test {attempt+1}: {response_time:.2f}s, model: {model_used}")
                        else:
                            errors.append(f"Failover test: HTTP {response.status_code}")
                        
                    except Exception as e:
                        errors.append(f"Failover test: {str(e)}")
                        logger.error(f"💥 Failover test {attempt+1}: {str(e)}")
        
        return self._calculate_performance_result(
            "ai_service_failover",
            len(failover_scenarios) * 5,
            successful,
            response_times,
            errors
        )
    
    async def test_system_health_endpoints(self) -> PerformanceResult:
        """Test system health and monitoring endpoints"""
        logger.info("🧪 Testing system health endpoints...")
        
        response_times = []
        errors = []
        successful = 0
        
        health_endpoints = [
            "/health",
            "/health/ai",
            "/health/system", 
            "/health/detailed"
        ]
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for endpoint in health_endpoints:
                for attempt in range(10):  # 10 attempts per endpoint
                    try:
                        start_time = time.time()
                        
                        response = await client.get(f"{self.base_url}{endpoint}")
                        
                        end_time = time.time()
                        response_time = end_time - start_time
                        response_times.append(response_time)
                        
                        if response.status_code == 200:
                            successful += 1
                        else:
                            errors.append(f"{endpoint}: HTTP {response.status_code}")
                        
                    except Exception as e:
                        errors.append(f"{endpoint}: {str(e)}")
        
        return self._calculate_performance_result(
            "health_endpoints",
            len(health_endpoints) * 10,
            successful,
            response_times,
            errors
        )
    
    def _calculate_performance_result(
        self, 
        test_name: str, 
        total_requests: int, 
        successful: int, 
        response_times: List[float], 
        errors: List[str]
    ) -> PerformanceResult:
        """Calculate performance metrics"""
        
        if not response_times:
            return PerformanceResult(
                test_name=test_name,
                total_requests=total_requests,
                successful_requests=successful,
                failed_requests=total_requests - successful,
                response_times=[],
                p50_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                average_response_time=0,
                requests_per_second=0,
                success_rate=0,
                errors=errors
            )
        
        # Sort response times for percentile calculations
        sorted_times = sorted(response_times)
        
        p50 = statistics.median(sorted_times)
        p95 = sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 0 else 0
        p99 = sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 0 else 0
        average = statistics.mean(sorted_times)
        
        # Calculate requests per second (assuming tests ran in sequence)
        total_time = sum(response_times)
        rps = len(response_times) / total_time if total_time > 0 else 0
        
        success_rate = (successful / total_requests * 100) if total_requests > 0 else 0
        
        return PerformanceResult(
            test_name=test_name,
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=total_requests - successful,
            response_times=response_times,
            p50_response_time=round(p50, 3),
            p95_response_time=round(p95, 3),
            p99_response_time=round(p99, 3),
            average_response_time=round(average, 3),
            requests_per_second=round(rps, 2),
            success_rate=round(success_rate, 2),
            errors=errors
        )
    
    async def run_all_performance_tests(self) -> Dict[str, Any]:
        """Run complete performance test suite"""
        logger.info("🚀 Starting NAVA Performance Test Suite...")
        
        # Check if NAVA is running
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code != 200:
                    raise Exception(f"NAVA health check failed: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"❌ NAVA is not running or not responding: {e}")
            return {"error": "NAVA service not available", "details": str(e)}
        
        # Run all tests
        tests = [
            ("Single Request Performance", self.test_single_request_performance),
            ("Concurrent 10 Users", lambda: self.test_concurrent_requests(10)),
            ("Concurrent 25 Users", lambda: self.test_concurrent_requests(25)),
            ("AI Service Failover", self.test_ai_service_failover),
            ("Health Endpoints", self.test_system_health_endpoints)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n📊 Running: {test_name}")
            try:
                result = await test_func()
                results[result.test_name] = result
                
                # Log key metrics
                logger.info(f"✅ {test_name} completed:")
                logger.info(f"   Success Rate: {result.success_rate}%")
                logger.info(f"   P95 Response Time: {result.p95_response_time}s")
                logger.info(f"   Average Response Time: {result.average_response_time}s")
                
            except Exception as e:
                logger.error(f"❌ {test_name} failed: {e}")
                results[test_name.lower().replace(" ", "_")] = {"error": str(e)}
        
        # Generate summary
        summary = self._generate_performance_summary(results)
        
        return {
            "summary": summary,
            "detailed_results": results,
            "timestamp": time.time()
        }
    
    def _generate_performance_summary(self, results: Dict[str, PerformanceResult]) -> Dict[str, Any]:
        """Generate performance test summary"""
        
        # Week 1 Success Criteria (from immediate_fixed.docx)
        success_criteria = {
            "response_time_p95": 3.0,  # <3s P95
            "success_rate": 95.0,      # >95%
            "concurrent_users": 25     # Handle 25+ concurrent users
        }
        
        # Calculate overall metrics
        all_response_times = []
        total_requests = 0
        total_successful = 0
        
        for result in results.values():
            if isinstance(result, PerformanceResult):
                all_response_times.extend(result.response_times)
                total_requests += result.total_requests
                total_successful += result.successful_requests
        
        overall_success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0
        overall_p95 = statistics.quantiles(all_response_times, n=20)[18] if all_response_times else 0
        
        # Check success criteria
        criteria_met = {
            "response_time_p95": overall_p95 <= success_criteria["response_time_p95"],
            "success_rate": overall_success_rate >= success_criteria["success_rate"],
            "concurrent_handling": "concurrent_25_users" in results and 
                                   isinstance(results["concurrent_25_users"], PerformanceResult) and
                                   results["concurrent_25_users"].success_rate >= 90.0
        }
        
        week_1_success = all(criteria_met.values())
        
        return {
            "week_1_success_criteria_met": week_1_success,
            "criteria_details": criteria_met,
            "overall_metrics": {
                "total_requests": total_requests,
                "total_successful": total_successful,
                "overall_success_rate": round(overall_success_rate, 2),
                "overall_p95_response_time": round(overall_p95, 3),
                "performance_improvement": "PASS" if overall_p95 <= 3.0 else "FAIL"
            },
            "recommendations": self._generate_recommendations(results, criteria_met)
        }
    
    def _generate_recommendations(
        self, 
        results: Dict[str, PerformanceResult], 
        criteria_met: Dict[str, bool]
    ) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        
        if not criteria_met["response_time_p95"]:
            recommendations.append("⚠️ Response time P95 still >3s. Check timeout handler configurations.")
        
        if not criteria_met["success_rate"]:
            recommendations.append("⚠️ Success rate <95%. Investigate error patterns and improve circuit breakers.")
        
        if not criteria_met["concurrent_handling"]:
            recommendations.append("⚠️ Concurrent user handling needs improvement. Consider load balancing.")
        
        # Check for specific issues
        for test_name, result in results.items():
            if isinstance(result, PerformanceResult):
                if result.success_rate < 90:
                    recommendations.append(f"🔧 {test_name} has low success rate ({result.success_rate}%). Investigate specific issues.")
                
                if result.p95_response_time > 5.0:
                    recommendations.append(f"🐌 {test_name} is slow (P95: {result.p95_response_time}s). Optimize timeout settings.")
        
        if not recommendations:
            recommendations.append("🎉 All performance targets met! System is ready for Week 2 features.")
        
        return recommendations

async def main():
    """Main performance test runner"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="NAVA Performance Test Suite")
    parser.add_argument("--url", default="http://localhost:8005", help="NAVA base URL")
    parser.add_argument("--output", default="performance_results.json", help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run performance tests
    tester = NAVAPerformanceTester(args.url)
    results = await tester.run_all_performance_tests()
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 NAVA PERFORMANCE TEST RESULTS")
    print("="*60)
    
    if "error" in results:
        print(f"❌ Test failed: {results['error']}")
        return 1
    
    summary = results["summary"]
    
    print(f"🎯 Week 1 Success Criteria: {'✅ PASSED' if summary['week_1_success_criteria_met'] else '❌ FAILED'}")
    print(f"📈 Overall Success Rate: {summary['overall_metrics']['overall_success_rate']}%")
    print(f"⚡ P95 Response Time: {summary['overall_metrics']['overall_p95_response_time']}s")
    print(f"📊 Total Requests: {summary['overall_metrics']['total_requests']}")
    
    print("\n🔍 Recommendations:")
    for rec in summary["recommendations"]:
        print(f"  {rec}")
    
    print(f"\n💾 Detailed results saved to: {args.output}")
    
    return 0 if summary['week_1_success_criteria_met'] else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))