# start_ai_services.py
"""
🚀 NAVA AI Services Quick Start
วางไฟล์นี้ใน: E:\nava-projects\

Script นี้จะ:
1. เริ่ม AI services ทั้ง 3 ตัว
2. ตรวจสอบ connectivity
3. Test NAVA performance
"""

import subprocess
import time
import asyncio
import httpx
import sys
import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

class NAVAServiceManager:
    """จัดการ NAVA AI Services"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.services = {
            "GPT": {
                "port": 8002,
                "path": "backend/services/03-external-ai/gpt-client",
                "process": None,
                "status": "stopped"
            },
            "Claude": {
                "port": 8003,
                "path": "backend/services/03-external-ai/claude-client",
                "process": None,
                "status": "stopped"
            },
            "Gemini": {
                "port": 8004,
                "path": "backend/services/03-external-ai/gemini-client",
                "process": None,
                "status": "stopped"
            },
            "NAVA": {
                "port": 8005,
                "path": "backend/services/01-core/nava-logic-controller",
                "process": None,
                "status": "stopped"
            }
        }
    
    def log(self, message, level="INFO"):
        icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
        print(f"{icons.get(level, 'ℹ️')} {message}")
    
    def start_service(self, service_name: str):
        """เริ่ม service"""
        service = self.services[service_name]
        service_path = self.project_root / service["path"]
        
        if not service_path.exists():
            self.log(f"Path not found: {service_path}", "ERROR")
            return False
        
        try:
            # Change to service directory and start
            process = subprocess.Popen(
                [sys.executable, "main.py"],
                cwd=service_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            service["process"] = process
            service["status"] = "starting"
            
            self.log(f"Started {service_name} service on port {service['port']}")
            return True
            
        except Exception as e:
            self.log(f"Failed to start {service_name}: {e}", "ERROR")
            return False
    
    async def check_service_health(self, service_name: str, max_attempts: int = 10):
        """ตรวจสอบ service health"""
        service = self.services[service_name]
        port = service["port"]
        
        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"http://localhost:{port}/health")
                    
                    if response.status_code == 200:
                        service["status"] = "healthy"
                        data = response.json()
                        self.log(f"✅ {service_name} is healthy: {data.get('status', 'unknown')}", "SUCCESS")
                        return True
                    else:
                        self.log(f"⚠️ {service_name} responded with HTTP {response.status_code}", "WARNING")
                        
            except Exception as e:
                if attempt < max_attempts - 1:
                    self.log(f"⏳ {service_name} not ready yet (attempt {attempt + 1}/{max_attempts})")
                    await asyncio.sleep(2)
                else:
                    self.log(f"❌ {service_name} health check failed: {str(e)[:50]}", "ERROR")
        
        service["status"] = "unhealthy"
        return False
    
    def start_all_services(self):
        """เริ่ม services ทั้งหมด"""
        self.log("🚀 Starting all NAVA AI services...")
        
        # Start AI services first
        ai_services = ["GPT", "Claude", "Gemini"]
        
        for service_name in ai_services:
            success = self.start_service(service_name)
            if success:
                time.sleep(2)  # Give each service time to start
            else:
                self.log(f"⚠️ Failed to start {service_name} - will continue with others", "WARNING")
        
        # Wait a bit before starting NAVA
        self.log("⏳ Waiting for AI services to initialize...")
        time.sleep(5)
        
        # Start NAVA controller
        nava_success = self.start_service("NAVA")
        if nava_success:
            self.log("✅ All services started", "SUCCESS")
        else:
            self.log("⚠️ NAVA failed to start", "WARNING")
        
        return True
    
    async def verify_all_services(self):
        """ตรวจสอบ services ทั้งหมด"""
        self.log("🔍 Verifying all services...")
        
        results = {}
        
        for service_name in self.services.keys():
            self.log(f"Checking {service_name}...")
            healthy = await self.check_service_health(service_name)
            results[service_name] = healthy
        
        # Summary
        healthy_count = sum(results.values())
        total_count = len(results)
        
        self.log(f"📊 Service Health Summary: {healthy_count}/{total_count} services healthy")
        
        if healthy_count == total_count:
            self.log("🎉 All services are healthy!", "SUCCESS")
        elif healthy_count >= 3:  # At least 3 services (including NAVA)
            self.log("✅ Enough services are healthy to proceed", "SUCCESS")
        else:
            self.log("❌ Too many services are unhealthy", "ERROR")
        
        return results
    
    async def test_nava_performance(self):
        """ทดสอบ NAVA performance"""
        self.log("🧪 Testing NAVA performance...")
        
        nava_url = "http://localhost:8005"
        test_cases = [
            {"message": "Hello world", "expected_time": 3.0},
            {"message": "Write a Python function to sort a list", "expected_time": 5.0},
            {"message": "Analyze this business scenario for me", "expected_time": 8.0}
        ]
        
        results = []
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            for i, test_case in enumerate(test_cases):
                try:
                    start_time = time.time()
                    
                    response = await client.post(
                        f"{nava_url}/chat",
                        json={
                            "message": test_case["message"],
                            "user_id": f"performance_test_{i}"
                        }
                    )
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    if response.status_code == 200:
                        data = response.json()
                        model_used = data.get("model_used", "unknown")
                        success = data.get("success", False)
                        
                        results.append({
                            "test": i + 1,
                            "message": test_case["message"][:30] + "...",
                            "response_time": response_time,
                            "model_used": model_used,
                            "success": success,
                            "target_met": response_time <= test_case["expected_time"]
                        })
                        
                        status = "✅" if response_time <= test_case["expected_time"] else "⚠️"
                        self.log(f"{status} Test {i+1}: {response_time:.2f}s, model: {model_used}")
                        
                    else:
                        self.log(f"❌ Test {i+1}: HTTP {response.status_code}", "ERROR")
                        results.append({
                            "test": i + 1,
                            "message": test_case["message"][:30] + "...",
                            "response_time": response_time,
                            "success": False,
                            "error": f"HTTP {response.status_code}"
                        })
                        
                except Exception as e:
                    self.log(f"❌ Test {i+1} failed: {str(e)[:50]}", "ERROR")
                    results.append({
                        "test": i + 1,
                        "message": test_case["message"][:30] + "...",
                        "success": False,
                        "error": str(e)[:50]
                    })
        
        # Performance summary
        successful_tests = [r for r in results if r.get("success", False)]
        if successful_tests:
            avg_time = sum(r["response_time"] for r in successful_tests) / len(successful_tests)
            max_time = max(r["response_time"] for r in successful_tests)
            targets_met = sum(1 for r in successful_tests if r.get("target_met", False))
            
            self.log("📊 Performance Summary:")
            self.log(f"   Average response time: {avg_time:.2f}s")
            self.log(f"   Maximum response time: {max_time:.2f}s")
            self.log(f"   Targets met: {targets_met}/{len(successful_tests)}")
            
            if max_time <= 8.0 and avg_time <= 5.0:
                self.log("🎉 Performance targets met!", "SUCCESS")
                return True
            else:
                self.log("⚠️ Performance could be improved", "WARNING")
                return False
        else:
            self.log("❌ No successful performance tests", "ERROR")
            return False
    
    async def check_emergency_mode_status(self):
        """ตรวจสอบสถานะ Emergency Mode"""
        self.log("🔍 Checking NAVA emergency mode status...")
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get("http://localhost:8005/health/detailed")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for emergency mode indicators
                    import_tests = data.get("import_tests", {})
                    working_imports = sum(1 for test in import_tests.values() if test.get("success", False))
                    total_imports = len(import_tests)
                    
                    self.log(f"📋 Import Status: {working_imports}/{total_imports} imports working")
                    
                    if working_imports >= total_imports * 0.7:  # 70% working
                        self.log("✅ NAVA is likely out of emergency mode", "SUCCESS")
                        return "normal"
                    else:
                        self.log("⚠️ NAVA may still be in emergency mode", "WARNING")
                        return "emergency"
                else:
                    self.log("⚠️ Cannot check NAVA detailed status", "WARNING")
                    return "unknown"
                    
        except Exception as e:
            self.log(f"❌ Error checking emergency mode: {str(e)[:50]}", "ERROR")
            return "error"
    
    def stop_all_services(self):
        """หยุด services ทั้งหมด"""
        self.log("🛑 Stopping all services...")
        
        for service_name, service in self.services.items():
            if service["process"]:
                try:
                    service["process"].terminate()
                    service["process"] = None
                    service["status"] = "stopped"
                    self.log(f"Stopped {service_name}")
                except:
                    pass
    
    def generate_report(self, service_results, performance_result, emergency_status):
        """สร้างรายงาน"""
        report = {
            "timestamp": time.time(),
            "services": service_results,
            "performance": {
                "test_passed": performance_result,
                "emergency_mode": emergency_status
            },
            "summary": {
                "healthy_services": sum(service_results.values()),
                "total_services": len(service_results),
                "overall_status": "ready" if performance_result and emergency_status == "normal" else "needs_attention"
            }
        }
        
        # Save report
        report_file = self.project_root / "nava_startup_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        self.log(f"📋 Startup report saved: {report_file}")
        return report

async def main():
    """Main startup sequence"""
    manager = NAVAServiceManager()
    
    print("🚀 NAVA AI Services Quick Start")
    print("=" * 50)
    
    try:
        # Step 1: Start all services
        manager.start_all_services()
        
        # Step 2: Wait for services to be ready
        print("\n⏳ Waiting for services to be ready...")
        await asyncio.sleep(10)
        
        # Step 3: Verify services
        service_results = await manager.verify_all_services()
        
        # Step 4: Check emergency mode
        emergency_status = await manager.check_emergency_mode_status()
        
        # Step 5: Test performance (if enough services are ready)
        healthy_count = sum(service_results.values())
        if healthy_count >= 3:
            performance_result = await manager.test_nava_performance()
        else:
            print("⚠️ Too few services healthy - skipping performance test")
            performance_result = False
        
        # Step 6: Generate report
        report = manager.generate_report(service_results, performance_result, emergency_status)
        
        # Step 7: Final summary
        print("\n" + "=" * 50)
        print("🎯 NAVA STARTUP SUMMARY")
        print("=" * 50)
        
        print(f"Services healthy: {report['summary']['healthy_services']}/{report['summary']['total_services']}")
        print(f"Performance test: {'✅ PASSED' if performance_result else '❌ FAILED'}")
        print(f"Emergency mode: {emergency_status}")
        print(f"Overall status: {report['summary']['overall_status']}")
        
        if report['summary']['overall_status'] == 'ready':
            print("\n🎉 NAVA is ready! You can now:")
            print("   1. Test chat: curl -X POST http://localhost:8005/chat \\")
            print("      -H 'Content-Type: application/json' \\")
            print("      -d '{\"message\":\"Hello NAVA\",\"user_id\":\"test\"}'")
            print("   2. Run performance tests: python performance_test.py")
            print("   3. Check health: curl http://localhost:8005/health")
        else:
            print("\n⚠️ NAVA needs attention:")
            if healthy_count < 3:
                print("   - Fix service startup issues")
            if not performance_result:
                print("   - Improve response times")
            if emergency_status == "emergency":
                print("   - Run emergency_mode_fix.py")
        
        return report['summary']['overall_status'] == 'ready'
        
    except KeyboardInterrupt:
        print("\n🛑 Startup interrupted by user")
        manager.stop_all_services()
        return False
    except Exception as e:
        print(f"\n💥 Startup failed: {e}")
        manager.stop_all_services()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)