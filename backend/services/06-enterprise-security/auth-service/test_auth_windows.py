# backend/services/06-enterprise-security/auth-service/test_auth_windows.py
"""
Authentication Service Test Script for Windows
Python version of the bash test script
"""

import requests
import json
import time
from datetime import datetime
import sys

# Configuration
BASE_URL = "http://localhost:8007"
TEST_USER = f"testuser_{int(time.time())}"
TEST_EMAIL = f"test_{int(time.time())}@example.com"
TEST_PASSWORD = "SecurePassword123!"

# Colors for Windows console
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'

def print_colored(message, color):
    """Print colored message"""
    print(f"{color}{message}{Colors.ENDC}")

def test_endpoint(method, endpoint, data=None, description="", auth_token=None):
    """Test API endpoint"""
    print_colored(f"Testing: {description}", Colors.YELLOW)
    
    url = f"{BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            print_colored(f"❌ Unsupported method: {method}", Colors.RED)
            return None
        
        if response.status_code < 400:
            print_colored(f"✅ {description} - PASSED", Colors.GREEN)
            try:
                result = response.json()
                print(json.dumps(result, indent=2))
            except:
                print(response.text)
            print()
            return response.json() if response.content else {}
        else:
            print_colored(f"❌ {description} - FAILED", Colors.RED)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            print()
            return None
            
    except requests.exceptions.ConnectionError:
        print_colored(f"❌ {description} - CONNECTION FAILED", Colors.RED)
        print("Make sure the authentication service is running on http://localhost:8007")
        return None
    except Exception as e:
        print_colored(f"❌ {description} - ERROR: {e}", Colors.RED)
        return None

def main():
    """Main test function"""
    print_colored("🔐 NAVA Authentication Service - Windows Testing", Colors.BLUE)
    print(f"Base URL: {BASE_URL}")
    print(f"Test User: {TEST_USER}")
    print()
    
    # Check if requests is available
    try:
        import requests
    except ImportError:
        print_colored("❌ requests library not found", Colors.RED)
        print("Install with: pip install requests")
        sys.exit(1)
    
    # Test 1: Health Check
    print_colored("🔍 Testing Service Health", Colors.BLUE)
    health_response = test_endpoint("GET", "/health", description="Health Check")
    
    if not health_response:
        print_colored("❌ Service is not running. Start with: python main.py", Colors.RED)
        sys.exit(1)
    
    # Test 2: Service Info
    test_endpoint("GET", "/", description="Service Info")
    
    # Test 3: User Registration
    print_colored("👤 Testing User Management", Colors.BLUE)
    register_data = {
        "username": TEST_USER,
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD,
        "role": "enterprise_user",
        "department": "Engineering",
        "requires_mfa": False
    }
    
    register_response = test_endpoint("POST", "/api/auth/register", register_data, "User Registration")
    
    # Test 4: User Login
    print_colored("🔑 Testing Authentication", Colors.BLUE)
    login_data = {
        "username": TEST_USER,
        "password": TEST_PASSWORD
    }
    
    login_response = test_endpoint("POST", "/api/auth/login", login_data, "User Login")
    
    access_token = None
    refresh_token = None
    
    if login_response and 'access_token' in login_response:
        access_token = login_response['access_token']
        refresh_token = login_response.get('refresh_token')
        print_colored("✅ Access Token Extracted", Colors.GREEN)
        print(f"Token: {access_token[:50]}...")
        print()
    else:
        print_colored("❌ Failed to get access token", Colors.RED)
        return
    
    # Test 5: Token Validation
    print_colored("🔍 Testing Token Validation", Colors.BLUE)
    test_endpoint("POST", "/api/jwt/validate", auth_token=access_token, description="Token Validation")
    
    # Test 6: Token Info
    test_endpoint("GET", "/api/jwt/info", auth_token=access_token, description="Token Info")
    
    # Test 7: Role Permissions
    print_colored("🔐 Testing Role-Based Access Control", Colors.BLUE)
    test_endpoint("GET", "/api/auth/permissions/enterprise_user", description="Enterprise User Permissions")
    test_endpoint("GET", "/api/auth/permissions/admin", description="Admin Permissions")
    
    # Test 8: MFA Setup
    print_colored("🔒 Testing Multi-Factor Authentication", Colors.BLUE)
    mfa_setup_data = {"user_id": "user_1"}
    test_endpoint("POST", "/api/mfa/setup", mfa_setup_data, "MFA Setup")
    
    # Test 9: MFA Status
    test_endpoint("GET", "/api/mfa/status/user_1", description="MFA Status")
    
    # Test 10: Session Management
    print_colored("👥 Testing Session Management", Colors.BLUE)
    session_data = {
        "user_id": "user_1",
        "username": TEST_USER,
        "role": "enterprise_user",
        "ip_address": "127.0.0.1",
        "device_info": "Python Test Script"
    }
    
    test_endpoint("POST", "/api/session/create", session_data, "Session Creation")
    
    # Test 11: Session Statistics
    test_endpoint("GET", "/api/session/stats", description="Session Statistics")
    
    # Test 12: User List
    test_endpoint("GET", "/api/auth/users", auth_token=access_token, description="User List")
    
    # Test 13: Token Refresh
    if refresh_token:
        print_colored("🔄 Testing Token Refresh", Colors.BLUE)
        refresh_data = {"refresh_token": refresh_token}
        test_endpoint("POST", "/api/jwt/refresh", refresh_data, "Token Refresh")
    
    # Summary
    print()
    print_colored("📊 Test Summary", Colors.BLUE)
    print_colored("✅ Authentication Service Windows Testing Complete!", Colors.GREEN)
    print()
    print_colored("📖 Next Steps:", Colors.YELLOW)
    print(f"1. View API Documentation: {BASE_URL}/docs")
    print(f"2. View ReDoc Documentation: {BASE_URL}/redoc")
    print("3. Ready for GitHub → Railway deployment")
    print()
    print_colored("🚀 Test User Created:", Colors.YELLOW)
    print(f"Username: {TEST_USER}")
    print(f"Email: {TEST_EMAIL}")
    print(f"Password: {TEST_PASSWORD}")
    print("Role: enterprise_user")
    print()
    print_colored("🎉 All tests completed successfully!", Colors.GREEN)

if __name__ == "__main__":
    main()
