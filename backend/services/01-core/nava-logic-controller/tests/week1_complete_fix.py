# week1_complete_fix.py
"""
NAVA Week 1 Complete Fix Runner
รัน patch และ test ทั้งหมดในไฟล์เดียว

Steps:
1. Backup main.py
2. Apply import fixes
3. Test all functions
4. Run performance test
5. Report results
"""

import sys
import os
import shutil
import time
import subprocess
import asyncio
import httpx
from datetime import datetime

# ===== CONFIGURATION =====
NAVA_DIR = "E:/nava-projects/backend/services/01-core/nava-logic-controller"
MAIN_PY_PATH = os.path.join(NAVA_DIR, "main.py")

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def print_step(step, description):
    print(f"\n📋 Step {step}: {description}")
    print("-" * 40)

def run_patch():
    """รัน patch สำหรับแก้ไข imports"""
    print_step(1, "Applying Import Fixes")
    
    try:
        # สร้าง backup
        backup_path = f"{MAIN_PY_PATH}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(MAIN_PY_PATH, backup_path)
        print(f"✅ Backup created: {backup_path}")
        
        # อ่านไฟล์ main.py
        with open(MAIN_PY_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # แก้ไข imports
        import re
        
        # 1. เพิ่ม sys และ os imports
        if "import sys" not in content:
            content = re.sub(
                r'(import logging\n)',
                r'\1import sys\nimport os\n',
                content,
                count=1
            )
        
        # 2. เพิ่ม path setup และ shared module imports
        path_setup = '''
# ===== PATH SETUP FOR SHARED MODULES =====
current_dir = os.path.dirname(os.path.abspath(__file__))
shared_path = os.path.join(current_dir, '..', '..', '..', 'shared')
if shared_path not in sys.path:
    sys.path.insert(0, shared_path)

# Try import shared modules with fallback
try:
    from common.cache_manager import cache_manager as shared_cache_manager, global_cache
    from common.circuit_breaker import circuit_breaker as shared_circuit_breaker
    from common.error_handler import handle_error, ErrorCategory
    SHARED_MODULES_AVAILABLE = True
    logger.info("✅ Shared modules imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Shared modules not available: {e}")
    SHARED_MODULES_AVAILABLE = False
'''
        
        if "PATH SETUP FOR SHARED MODULES" not in content:
            content = re.sub(
                r'(logger = logging\.getLogger\(__name__\))',
                r'\1' + path_setup,
                content,
                count=1
            )
        
        # 3. แก้ไข fallback functions
        fallback_enhancement = '''
# ===== ENHANCED FALLBACK FUNCTIONS =====
def get_cached_response(*args, **kwargs):
    if SHARED_MODULES_AVAILABLE:
        try:
            return shared_cache_manager.get_similar_response(args[0] if args else "")
        except Exception as e:
            logger.debug(f"Shared cache error: {e}")
    return None

def cache_response(*args, **kwargs):
    if SHARED_MODULES_AVAILABLE:
        try:
            if len(args) >= 2:
                shared_cache_manager.cache_response(args[0], args[1])
            return
        except Exception as e:
            logger.debug(f"Shared cache storage error: {e}")
    pass

def get_cache_stats():
    if SHARED_MODULES_AVAILABLE:
        try:
            return shared_cache_manager.get_cache_stats()
        except Exception as e:
            logger.debug(f"Shared cache stats error: {e}")
    
    return {
        "error": "Cache manager not available",
        "available": False,
        "hit_rate": 0,
        "total_requests": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "memory_entries": 0
    }'''
        
        # แทนที่ fallback functions เดิม
        if "ENHANCED FALLBACK FUNCTIONS" not in content:
            old_fallback_pattern = r'# ===== FIXED: Create fallback functions first =====.*?(?=def get_service_health)'
            content = re.sub(
                old_fallback_pattern,
                fallback_enhancement + '\n\n',
                content,
                flags=re.DOTALL
            )
        
        # 4. แก้ไข stabilization_wrapper calls
        content = re.sub(
            r'stabilization_wrapper\.get_cached_response\(',
            r'get_cached_response(',
            content
        )
        
        content = re.sub(
            r'stabilization_wrapper\.cache_response\(',
            r'cache_response(',
            content
        )
        
        # เขียนไฟล์ใหม่
        with open(MAIN_PY_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Import fixes applied successfully")
        return True, backup_path
        
    except Exception as e:
        print(f"❌ Patch failed: {e}")
        return False, None

def test_imports():
    """ทดสอบ imports"""
    print_step(2, "Testing Imports")
    
    try:
        original_dir = os.getcwd()
        os.chdir(NAVA_DIR)
        
        # ทดสอบ import shared modules
        sys.path.insert(0, os.path.join(NAVA_DIR, '..', '..', '..', 'shared'))
        
        from common.cache_manager import cache_manager
        from common.circuit_breaker import circuit_breaker
        print("✅ Shared modules import OK")
        
        # ทดสอบ app modules  
        from app.core.controller import NAVAController
        from app.service.logic_orchestrator import LogicOrchestrator
        print("✅ App modules import OK")
        
        os.chdir(original_dir)
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return False

async def test_server():
    """ทดสอบ server startup และ endpoints"""
    print_step(3, "Testing Server")
    
    process = None
    try:
        # เริ่ม server
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=NAVA_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("⏳ Starting server (10 seconds)...")
        await asyncio.sleep(10)
        
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print(f"❌ Server failed to start: {stderr.decode()[:200]}")
            return False
        
        # ทดสอบ health endpoint
        async with httpx.AsyncClient(timeout=10.0) as client:
            print("🏥 Testing health endpoint...")
            health_response = await client.get("http://localhost:8005/health")
            
            if health_response.status_code == 200:
                print("✅ Health endpoint OK")
            else:
                print(f"❌ Health endpoint failed: {health_response.status_code}")
                return False
            
            # ทดสอบ chat endpoint
            print("💬 Testing chat endpoint...")
            start_time = time.time()
            chat_response = await client.post(
                "http://localhost:8005/chat",
                json={"message": "Hello test", "user_id": "test_user"}
            )
            response_time = time.time() - start_time
            
            if chat_response.status_code == 200:
                print(f"✅ Chat endpoint OK ({response_time:.2f}s)")
                
                if response_time < 5.0:
                    print("🚀 Response time improved!")
                else:
                    print("⚠️ Response time still slow")
                
                return True
            else:
                print(f"❌ Chat endpoint failed: {chat_response.status_code}")
                return False
        
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return False
    finally:
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except:
                process.kill()

def run_performance_test():
    """รัน performance test"""
    print_step(4, "Running Performance Test")
    
    try:
        result = subprocess.run(
            [sys.executable, "performance_test.py"],
            cwd=NAVA_DIR,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print("✅ Performance test PASSED")
            
            # แสดงผลลัพธ์
            output = result.stdout
            for line in output.split('\n'):
                if any(keyword in line for keyword in ["P95 Response Time:", "Overall Success Rate:", "Week 1 Success"]):
                    print(f"📊 {line.strip()}")
            
            return True
        else:
            print("❌ Performance test FAILED")
            print(f"Error: {result.stderr[:300]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Performance test timeout (> 2 minutes)")
        return False
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False

def generate_report(results):
    """สร้างรายงานผลลัพธ์"""
    print_header("WEEK 1 FIX RESULTS")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    success_rate = passed / total * 100
    print(f"\n📈 Overall Success Rate: {success_rate:.1f}% ({passed}/{total})")
    
    if success_rate >= 80:
        print("\n🎉 WEEK 1 FIXES SUCCESSFUL!")
        print("🚀 NAVA is ready for production")
        print("\n📋 Next Steps:")
        print("1. ✅ Performance issues resolved")
        print("2. ✅ Advanced features can be activated")
        print("3. 📊 Monitor system in production")
        print("4. 🔧 Proceed to Week 2 optimizations")
        
    elif success_rate >= 60:
        print("\n⚠️ PARTIAL SUCCESS")
        print("🔧 Additional fixes needed")
        print("\n📋 Action Required:")
        failed_tests = [name for name, result in results.items() if not result]
        for test in failed_tests:
            print(f"❌ Fix: {test}")
            
    else:
        print("\n❌ MAJOR ISSUES REMAIN")
        print("🆘 Manual intervention required")
        print("\n📋 Critical Actions:")
        print("1. Check error logs in detail")
        print("2. Verify all service dependencies")
        print("3. Consider rollback if necessary")
    
    return success_rate >= 80

async def main():
    """Main execution"""
    print_header("NAVA WEEK 1 COMPLETE FIX")
    print("🎯 Objective: Fix timeout issues and activate advanced features")
    print("⏰ Estimated time: 5-10 minutes")
    
    results = {}
    
    # Step 1: Apply patch
    patch_success, backup_path = run_patch()
    results["Import Patch"] = patch_success
    
    if not patch_success:
        print("❌ Cannot continue without successful patch")
        return False
    
    # Step 2: Test imports
    import_success = test_imports()
    results["Import Test"] = import_success
    
    # Step 3: Test server
    server_success = await test_server()
    results["Server Test"] = server_success
    
    # Step 4: Performance test
    if server_success:
        perf_success = run_performance_test()
        results["Performance Test"] = perf_success
    else:
        results["Performance Test"] = False
    
    # Step 5: Generate report
    overall_success = generate_report(results)
    
    # Show backup info
    if backup_path:
        print(f"\n💾 Backup saved: {backup_path}")
        print("🔄 Restore command: cp {backup_path} {MAIN_PY_PATH}")
    
    return overall_success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        print(f"\n🏁 Exit code: {0 if success else 1}")
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        exit(1)
