# E:\nava-projects\backend\services\01-core\nava-logic-controller\tests\test_service_discovery.py
import sys
import os

# เพิ่ม path ไปยัง parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# แก้ path ให้ถูกต้อง - ไฟล์อยู่ใน app/service/ ไม่ใช่ app/config/
from app.service.Phase2_service_discovery import service_discovery
import requests

def test_service_discovery():
    """ทดสอบ Service Discovery Configuration"""
    
    print("🔍 Testing Service Discovery Configuration...")
    print("=" * 50)
    
    # ทดสอบการเรียกใช้ service URLs
    services = ["decision_engine", "quality_service", "slf_framework", "cache_engine"]
    
    for service_name in services:
        print(f"\n📡 Testing {service_name}:")
        
        # ทดสอบ get_service_url
        service_url = service_discovery.get_service_url(service_name)
        print(f"   URL: {service_url}")
        
        # ทดสอบ get_health_url
        health_url = service_discovery.get_health_url(service_name)
        print(f"   Health URL: {health_url}")
        
        # ทดสอบ get_service_endpoint
        test_endpoint = service_discovery.get_service_endpoint(service_name, "/test")
        print(f"   Test Endpoint: {test_endpoint}")
        
        # ทดสอบการเชื่อมต่อจริง (เฉพาะ decision_engine ที่รันอยู่)
        if service_name == "decision_engine":
            try:
                response = requests.get(health_url, timeout=5)
                print(f"   🟢 Health Check: {response.status_code}")
                if response.status_code == 200:
                    json_data = response.json()
                    print(f"   Service Status: {json_data.get('status', 'unknown')}")
            except requests.exceptions.RequestException as e:
                print(f"   🔴 Health Check Failed: {e}")
        else:
            print(f"   ⚠️  Service not running (expected)")

def test_decision_engine_apis():
    """ทดสอบ Decision Engine APIs"""
    
    print("\n🎯 Testing Decision Engine APIs...")
    print("=" * 50)
    
    base_url = service_discovery.get_service_url("decision_engine")
    
    # ทดสอบ API endpoints
    endpoints = [
        "/health",
        "/api/decision/models",
        "/api/decision/patterns"
    ]
    
    for endpoint in endpoints:
        full_url = f"{base_url}{endpoint}"
        print(f"\n📡 Testing: {full_url}")
        
        try:
            response = requests.get(full_url, timeout=5)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                json_response = response.json()
                print(f"   Response Keys: {list(json_response.keys())}")
            else:
                print(f"   Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"   🔴 Connection Error: {e}")

if __name__ == "__main__":
    print("🚀 NAVA Service Discovery Test Suite")
    print("=" * 60)
    
    test_service_discovery()
    test_decision_engine_apis()
    
    print("\n✅ Test completed!")