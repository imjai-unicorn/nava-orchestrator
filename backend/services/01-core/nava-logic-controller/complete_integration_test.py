# complete_integration_test.py
"""
Complete NAVA Integration Test - ตรวจสอบทุกอย่างก่อน Production
วางไฟล์นี้ใน E:\nava-projects\backend\services\01-core\nava-logic-controller\
"""

import asyncio
import sys
import httpx
import json
from datetime import datetime
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def print_result(test_name, success, message, details=""):
    icon = "✅" if success else "❌"
    print(f"{icon} {test_name}: {message}")
    if details:
        print(f"   └─ {details}")

async def test_1_file_structure():
    """ตรวจสอบไฟล์ทั้งหมดมีครบไหม"""
    print_header("File Structure Check")
    
    required_files = {
        "Enhanced Decision Engine": "app/core/decision_engine.py",
        "Enhanced Orchestrator": "app/service/logic_orchestrator.py",
        "Service Discovery": "app/service/service_discovery.py", 
        "Real AI Client": "app/service/real_ai_client.py",
        "Main Controller": "app/core/controller.py",
        "Core __init__": "app/core/__init__.py",
        "Service __init__": "app/service/__init__.py",
        "Main API": "main.py"
    }
    
    results = {}
    for name, filepath in required_files.items():
        exists = Path(filepath).exists()
        results[name] = exists
        print_result(name, exists, f"{filepath} {'EXISTS' if exists else 'MISSING'}")
    
    all_exist = all(results.values())
    print_result("All Files Check", all_exist, f"{sum(results.values())}/{len(results)} files present")
    
    return all_exist, results

async def test_2_imports():
    """ตรวจสอบ Enhanced Imports"""
    print_header("Enhanced Imports Test")
    
    import_tests = []
    
    # Test Enhanced Decision Engine
    try:
        from app.core.decision_engine import EnhancedDecisionEngine
        engine = EnhancedDecisionEngine()
        print_result("Enhanced Decision Engine", True, "Import successful")
        
        # Test behavior patterns
        patterns = engine.get_behavior_patterns()
        pattern_count = len(patterns["patterns"])
        print_result("Behavior Patterns", pattern_count > 5, f"{pattern_count} patterns loaded")
        import_tests.append(True)
        
    except Exception as e:
        print_result("Enhanced Decision Engine", False, f"Import failed: {e}")
        import_tests.append(False)
    
    # Test Enhanced Orchestrator  
    try:
        from app.service.logic_orchestrator import LogicOrchestrator
        orchestrator = LogicOrchestrator()
        print_result("Enhanced Orchestrator", True, "Import successful")
        
        # Test version
        version = getattr(orchestrator, 'version', 'unknown')
        print_result("Orchestrator Version", version.startswith('2.'), f"Version: {version}")
        import_tests.append(True)
        
    except Exception as e:
        print_result("Enhanced Orchestrator", False, f"Import failed: {e}")
        import_tests.append(False)
    
    # Test Service Components
    try:
        from app.service.service_discovery import ServiceDiscovery
        from app.service.real_ai_client import RealAIClient
        print_result("Service Components", True, "ServiceDiscovery + RealAIClient")
        import_tests.append(True)
        
    except Exception as e:
        print_result("Service Components", False, f"Import failed: {e}")
        import_tests.append(False)
    
    success_rate = sum(import_tests) / len(import_tests) if import_tests else 0
    overall_success = success_rate >= 0.8
    print_result("Import Success Rate", overall_success, f"{success_rate:.1%} imports successful")
    
    return overall_success, import_tests

async def test_3_enhanced_decision_logic():
    """ตรวจสอบ Enhanced Decision Logic"""
    print_header("Enhanced Decision Logic Test")
    
    try:
        from app.core.decision_engine import EnhancedDecisionEngine
        engine = EnhancedDecisionEngine()
        
        test_cases = [
            ("help me write Python code", "gpt", "code_development"),
            ("analyze this business in detail", "claude", "deep_analysis"),
            ("สร้างแผนกลยุทธ์สำหรับบริษัท", "gemini", "strategic_planning"),
            ("คุยเรื่องทั่วไป สนทนาปกติ", "gpt", "conversation"),
            ("วิจัยเรื่องนี้ให้ครบถ้วน", "claude", "research_workflow")
        ]
        
        results = []
        for message, expected_model, expected_pattern in test_cases:
            try:
                selected_model, confidence, reasoning = engine.select_model(message)
                
                model_correct = selected_model == expected_model
                pattern_detected = reasoning.get("behavior_analysis", {}).get("detected_pattern")
                success = model_correct and (confidence > 0.6)
                
                success = model_correct and confidence_good
                results.append(success)
                
                details = f"Model: {selected_model} (exp: {expected_model}), Confidence: {confidence:.2f}, Pattern: {pattern_detected}"
                print_result(f"Test: '{message[:30]}...'", success, "PASS" if success else "FAIL", details)
                
            except Exception as e:
                print_result(f"Test: '{message[:30]}...'", False, f"ERROR: {e}")
                results.append(False)
        
        success_rate = sum(results) / len(results) if results else 0
        overall_success = success_rate >= 0.8
        print_result("Decision Logic Success", overall_success, f"{success_rate:.1%} tests passed")
        
        return overall_success, results
        
    except Exception as e:
        print_result("Decision Logic Test", False, f"Setup failed: {e}")
        return False, []

async def test_4_ai_services_connectivity():
    """ตรวจสอบการเชื่อมต่อ AI Services"""
    print_header("AI Services Connectivity Test")
    
    services = {
        "GPT Service": "http://localhost:8002",
        "Claude Service": "http://localhost:8003", 
        "Gemini Service": "http://localhost:8004"
    }
    
    results = {}
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for service_name, url in services.items():
            try:
                # Test health endpoint
                health_response = await client.get(f"{url}/health")
                health_ok = health_response.status_code == 200
                
                if health_ok:
                    # Test chat endpoint
                    chat_payload = {"message": "test", "user_id": "test"}
                    chat_response = await client.post(f"{url}/chat", json=chat_payload)
                    chat_ok = chat_response.status_code == 200
                    
                    if chat_ok:
                        response_data = chat_response.json()
                        has_response = "response" in response_data
                        results[service_name] = "WORKING"
                        print_result(service_name, True, f"WORKING - {url}")
                    else:
                        results[service_name] = "HEALTH_ONLY"
                        print_result(service_name, False, f"Health OK, Chat Failed - {chat_response.status_code}")
                else:
                    results[service_name] = "DOWN"
                    print_result(service_name, False, f"DOWN - {health_response.status_code}")
                    
            except Exception as e:
                results[service_name] = "ERROR"
                print_result(service_name, False, f"CONNECTION ERROR - {str(e)[:50]}")
    
    working_services = [k for k, v in results.items() if v == "WORKING"]
    all_working = len(working_services) == len(services)
    
    print_result("Services Summary", all_working, f"{len(working_services)}/{len(services)} services working")
    
    return all_working, results

async def test_5_nava_orchestrator():
    """ตรวจสอบ NAVA Orchestrator รวม"""
    print_header("NAVA Orchestrator Integration Test")
    
    try:
        from app.service.logic_orchestrator import LogicOrchestrator
        
        # Initialize orchestrator
        orchestrator = LogicOrchestrator()
        await orchestrator.initialize()
        print_result("Orchestrator Init", True, "Initialization successful")
        
        # Test single model workflow
        result1 = await orchestrator.process_request(
            message="write a simple Python function",
            user_id="test_user"
        )
        
        single_success = "response" in result1 and not result1.get("error")
        model_used = result1.get("model_used", "unknown")
        print_result("Single Workflow", single_success, f"Model: {model_used}")
        
        # Test behavior detection
        result2 = await orchestrator.process_request(
            message="วิเคราะห์ธุรกิจนี้อย่างละเอียด",
            user_id="test_user"
        )
        
        behavior_success = "response" in result2 and not result2.get("error")
        behavior_model = result2.get("model_used", "unknown")
        print_result("Behavior Detection", behavior_success, f"Model: {behavior_model}")
        
        # Test system status
        status = await orchestrator.get_system_status()
        status_ok = status.get("nava_initialized", False)
        available_models = status.get("available_models", [])
        print_result("System Status", status_ok, f"Available: {available_models}")
        
        overall_success = single_success and behavior_success and status_ok
        return overall_success, {
            "single_workflow": single_success,
            "behavior_detection": behavior_success, 
            "system_status": status_ok
        }
        
    except Exception as e:
        print_result("Orchestrator Test", False, f"Failed: {e}")
        return False, {"error": str(e)}

async def test_6_nava_api_integration():
    """ตรวจสอบ NAVA API รวม (ถ้า server รันอยู่)"""
    print_header("NAVA API Integration Test")
    
    nava_url = "http://localhost:8005"  # หรือ port ที่ใช้
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test health
            health_response = await client.get(f"{nava_url}/health")
            health_ok = health_response.status_code == 200
            
            if not health_ok:
                print_result("NAVA API", False, f"Server not running on {nava_url}")
                return False, {"server_status": "down"}
            
            health_data = health_response.json()
            print_result("NAVA Health", True, f"Status: {health_data.get('status', 'unknown')}")
            
            # Test enhanced chat
            chat_tests = [
                {"message": "write Python code", "expected_model": "gpt"},
                {"message": "analyze business data", "expected_model": "claude"},
                {"message": "create strategic plan", "expected_model": "gemini"}
            ]
            
            test_results = []
            for test_case in chat_tests:
                try:
                    chat_response = await client.post(
                        f"{nava_url}/chat",
                        json=test_case,
                        timeout=30.0
                    )
                    
                    if chat_response.status_code == 200:
                        data = chat_response.json()
                        model_used = data.get("model_used", "unknown")
                        has_response = bool(data.get("response"))
                        
                        success = has_response and not data.get("error")
                        test_results.append(success)
                        
                        print_result(
                            f"Chat: {test_case['message'][:20]}...", 
                            success, 
                            f"Model: {model_used}",
                            f"Response length: {len(data.get('response', ''))}"
                        )
                    else:
                        test_results.append(False)
                        print_result(
                            f"Chat: {test_case['message'][:20]}...",
                            False,
                            f"HTTP {chat_response.status_code}"
                        )
                        
                except Exception as e:
                    test_results.append(False)
                    print_result(
                        f"Chat: {test_case['message'][:20]}...",
                        False,
                        f"Error: {str(e)[:30]}"
                    )
            
            success_rate = sum(test_results) / len(test_results) if test_results else 0
            overall_success = success_rate >= 0.7
            
            print_result("API Integration", overall_success, f"{success_rate:.1%} API tests passed")
            
            return overall_success, {
                "health_check": health_ok,
                "chat_tests": test_results,
                "success_rate": success_rate
            }
            
    except Exception as e:
        print_result("NAVA API Test", False, f"Connection failed: {e}")
        return False, {"connection_error": str(e)}

async def test_7_advanced_complexity_analysis():
    """ทดสอบ Advanced Complexity Analysis"""
    print_header("Advanced Complexity Analysis Test")
    
    try:
        from app.core.decision_engine import EnhancedDecisionEngine
        engine = EnhancedDecisionEngine()
        
        test_cases = [
            {"input": "Create machine learning pipeline", "description": "Complex ML task"},
            {"input": "Hello world", "description": "Simple task"},
            {"input": "Analyze business strategy", "description": "Business analysis"}
        ]
        
        results = []
        for test_case in test_cases:
            try:
                complexity_result = engine.analyze_task_complexity_advanced(test_case["input"])
                detected_tier = complexity_result.get("complexity_tier", "unknown")
                overall_complexity = complexity_result.get("overall_complexity", 0)
                
                results.append({
                    "input": test_case["input"][:30] + "...",
                    "detected": detected_tier,
                    "complexity_score": overall_complexity
                })
                
                print_result(test_case["description"], True, f"Tier: {detected_tier}, Score: {overall_complexity:.3f}")
                
            except Exception as e:
                print_result(test_case["description"], False, f"Error: {e}")
                results.append({"error": str(e)})
        
        success_rate = len([r for r in results if "error" not in r]) / len(results)
        overall_success = success_rate >= 0.7
        
        print_result("Complexity Analysis Overall", overall_success, f"{success_rate:.1%} success rate")
        return overall_success, results
        
    except Exception as e:
        print_result("Complexity Analysis Test", False, f"Setup failed: {e}")
        return False, {"error": str(e)}

async def test_8_advanced_orchestration():
    """ทดสอบ Advanced Orchestration"""
    print_header("Advanced Orchestration Test")
    
    try:
        from app.service.logic_orchestrator import LogicOrchestrator
        
        orchestrator = LogicOrchestrator()
        await orchestrator.initialize()
        
        # Test complex request processing
        if hasattr(orchestrator, 'process_complex_request'):
            result = await orchestrator.process_complex_request(
                "Create comprehensive business analysis",
                "test_user",
                complexity_level="advanced_professional"
            )
            
            has_response = "response" in result and len(result["response"]) > 10
            has_workflow_info = "workflow_type" in result
            
            success = has_response and has_workflow_info
            print_result("Complex Request", success, f"Workflow: {result.get('workflow_type', 'unknown')}")
            
            return success, {"complex_request": success}
        else:
            print_result("Complex Request", False, "Method not available")
            return False, {"error": "process_complex_request not available"}
        
    except Exception as e:
        print_result("Advanced Orchestration", False, f"Error: {e}")
        return False, {"error": str(e)}

async def main():
    """รัน Integration Test ครบชุด"""
    print(f"🚀 NAVA Complete Integration Test")
    print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Location: {Path.cwd()}")
    
    # รัน tests ทั้งหมด
    tests = [
        ("File Structure", test_1_file_structure),
        ("Enhanced Imports", test_2_imports),
        ("Decision Logic", test_3_enhanced_decision_logic),
        ("AI Services", test_4_ai_services_connectivity),
        ("NAVA Orchestrator", test_5_nava_orchestrator),
        ("NAVA API", test_6_nava_api_integration),
        ("Advanced Complexity", test_7_advanced_complexity_analysis),
        ("Advanced Orchestration", test_8_advanced_orchestration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success, details = await test_func()
            results[test_name] = {"success": success, "details": details}
        except Exception as e:
            results[test_name] = {"success": False, "details": {"error": str(e)}}
            print_result(test_name, False, f"Test crashed: {e}")
    
    # สรุปผลรวม
    print_header("COMPLETE INTEGRATION TEST SUMMARY")
    
    passed_tests = []
    failed_tests = []
    
    for test_name, result in results.items():
        success = result["success"]
        if success:
            passed_tests.append(test_name)
        else:
            failed_tests.append(test_name)
        
        print_result(test_name, success, "PASS" if success else "FAIL")
    
    total_tests = len(results)
    passed_count = len(passed_tests)
    success_rate = passed_count / total_tests * 100
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"✅ Passed: {passed_count}/{total_tests} ({success_rate:.1f}%)")
    print(f"❌ Failed: {len(failed_tests)}/{total_tests}")
    
    if failed_tests:
        print(f"\n⚠️  Failed Tests: {', '.join(failed_tests)}")
    
    # Production Readiness Assessment
    production_ready = passed_count >= 4 and "AI Services" in passed_tests
    
    print(f"\n🎯 PRODUCTION READINESS: {'✅ READY' if production_ready else '❌ NOT READY'}")
    
    if production_ready:
        print("🚀 NAVA is ready for production deployment!")
    else:
        print("🔧 Fix failing tests before production deployment.")
        
    return production_ready

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1)