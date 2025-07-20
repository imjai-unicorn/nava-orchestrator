# run_all_tests.py
"""
Test runner script for all NAVA components
Run this script to execute all tests and generate a comprehensive report
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import json

def run_test_file(test_file_path, test_name):
    """Run a specific test file and return results"""
    print(f"\n{'='*60}")
    print(f"🧪 Running {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_file_path, 
            "-v", 
            "--tb=short",
            "--disable-warnings"
        ], capture_output=True, text=True, timeout=300)
        
        execution_time = time.time() - start_time
        
        success = result.returncode == 0
        
        print(f"⏱️  Execution time: {execution_time:.2f} seconds")
        print(f"📊 Exit code: {result.returncode}")
        
        if success:
            print(f"✅ {test_name} - PASSED")
        else:
            print(f"❌ {test_name} - FAILED")
            print("\n🔍 Error output:")
            print(result.stderr)
            print("\n📝 Test output:")
            print(result.stdout)
        
        return {
            "name": test_name,
            "file": test_file_path,
            "success": success,
            "execution_time": execution_time,
            "output": result.stdout,
            "errors": result.stderr,
            "exit_code": result.returncode
        }
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} - TIMEOUT (exceeded 5 minutes)")
        return {
            "name": test_name,
            "file": test_file_path,
            "success": False,
            "execution_time": 300.0,
            "output": "",
            "errors": "Test execution timeout",
            "exit_code": -1
        }
    except Exception as e:
        print(f"💥 {test_name} - ERROR: {str(e)}")
        return {
            "name": test_name,
            "file": test_file_path,
            "success": False,
            "execution_time": time.time() - start_time,
            "output": "",
            "errors": str(e),
            "exit_code": -2
        }

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = ["pytest", "pydantic", "pillow"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - available")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def create_test_directories():
    """Create test directories if they don't exist"""
    test_dirs = [
        "backend/services/01-core/nava-logic-controller/tests",
        "backend/services/03-external-ai/gemini-client/tests", 
        "backend/services/shared/common/tests"
    ]
    
    for test_dir in test_dirs:
        os.makedirs(test_dir, exist_ok=True)
        
        # Create __init__.py files
        init_file = os.path.join(test_dir, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# Test package\n")

def generate_test_report(test_results):
    """Generate a comprehensive test report"""
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result["success"])
    failed_tests = total_tests - passed_tests
    total_time = sum(result["execution_time"] for result in test_results)
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"📋 NAVA TEST EXECUTION REPORT")
    print(f"{'='*80}")
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total execution time: {total_time:.2f} seconds")
    print(f"📊 Tests executed: {total_tests}")
    print(f"✅ Tests passed: {passed_tests}")
    print(f"❌ Tests failed: {failed_tests}")
    print(f"📈 Success rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print(f"🎉 EXCELLENT - Ready for deployment!")
    elif success_rate >= 80:
        print(f"👍 GOOD - Minor issues to address")
    elif success_rate >= 70:
        print(f"⚠️  NEEDS IMPROVEMENT - Several issues found")
    else:
        print(f"🚨 CRITICAL - Major issues require attention")
    
    print(f"\n{'='*80}")
    print(f"📝 DETAILED RESULTS")
    print(f"{'='*80}")
    
    for i, result in enumerate(test_results, 1):
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{i:2d}. {result['name']:<30} {status} ({result['execution_time']:.2f}s)")
        
        if not result["success"]:
            print(f"    ⚠️  Error: {result['errors'][:100]}...")
    
    # Save detailed report to file
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "total_time": total_time
        },
        "detailed_results": test_results
    }
    
    with open("test_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: test_report.json")
    
    return success_rate >= 80  # Return True if ready for deployment

def main():
    """Main test execution function"""
    print("🚀 NAVA Comprehensive Test Suite")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install missing packages.")
        return False
    
    # Create test directories
    create_test_directories()
    
    # Define all test files to run
    test_files = [
        {
            "name": "Enhanced Chat Models",
            "file": "backend/services/01-core/nava-logic-controller/tests/test_enhanced_chat.py"
        },
        {
            "name": "Feedback System", 
            "file": "backend/services/01-core/nava-logic-controller/tests/test_feedback.py"
        },
        {
            "name": "Multimodal Handler",
            "file": "backend/services/03-external-ai/gemini-client/tests/test_multimodal_handler.py"
        },
        {
            "name": "Memory Manager",
            "file": "backend/services/shared/common/tests/test_memory_manager.py"
        },
        {
            "name": "Trust Calculator",
            "file": "backend/services/shared/common/tests/test_trust_calculator.py"
        }
    ]
    
    # Check if test files exist
    print("📂 Checking test files...")
    missing_files = []
    for test in test_files:
        if os.path.exists(test["file"]):
            print(f"✅ {test['file']}")
        else:
            print(f"❌ {test['file']} - NOT FOUND")
            missing_files.append(test["file"])
    
    if missing_files:
        print(f"\n⚠️  Missing test files: {len(missing_files)}")
        print("Please create the missing test files before running the test suite.")
        return False
    
    # Run all tests
    print(f"\n🎯 Starting test execution for {len(test_files)} test suites...")
    test_results = []
    
    start_time = time.time()
    
    for test_info in test_files:
        result = run_test_file(test_info["file"], test_info["name"])
        test_results.append(result)
    
    total_execution_time = time.time() - start_time
    
    # Generate and display report
    deployment_ready = generate_test_report(test_results)
    
    print(f"\n{'='*80}")
    print(f"🏁 Test suite completed in {total_execution_time:.2f} seconds")
    
    if deployment_ready:
        print(f"🎯 DEPLOYMENT READY - All critical tests passed!")
        print(f"✅ You can proceed with deployment")
        return True
    else:
        print(f"🔧 NEEDS FIXES - Please address failing tests before deployment")
        print(f"❌ Do not deploy until issues are resolved")
        return False

if __name__ == "__main__":
    # Set up Python path to find modules
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    # Change to the project root directory if needed
    if os.path.basename(current_dir) != "nava-projects":
        # Try to find the project root
        project_root = current_dir
        while project_root != "/" and not os.path.exists(os.path.join(project_root, "backend")):
            project_root = os.path.dirname(project_root)
        
        if os.path.exists(os.path.join(project_root, "backend")):
            os.chdir(project_root)
            print(f"📁 Changed to project root: {project_root}")
    
    # Run the test suite
    try:
        success = main()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {str(e)}")
        sys.exit(1)
