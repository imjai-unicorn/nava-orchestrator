#!/usr/bin/env python3
"""
NAVA Phase 2 Services Startup Script
====================================
Starts all Phase 2 Intelligence Enhancement Services
"""

import os
import time
import subprocess
import requests
import sys
from pathlib import Path
import threading
import signal

class ServiceManager:
    def __init__(self):
        self.services = [
            {
                "name": "Decision Engine",
                "path": "backend/services/05-enhanced-intelligence/decision-engine",
                "port": 8008,
                "process": None
            },
            {
                "name": "Quality Service", 
                "path": "backend/services/05-enhanced-intelligence/quality-service",
                "port": 8009,
                "process": None
            },
            {
                "name": "SLF Framework",
                "path": "backend/services/05-enhanced-intelligence/slf-framework", 
                "port": 8010,
                "process": None
            },
            {
                "name": "Cache Engine",
                "path": "backend/services/05-enhanced-intelligence/cache-engine",
                "port": 8013,
                "process": None
            }
        ]
        self.base_path = Path("E:/nava-projects")
        self.running_services = []
        
    def start_service(self, service):
        """Start a single service"""
        service_path = self.base_path / service["path"]
        main_py = service_path / "main.py"
        
        print(f"🔄 Starting {service['name']} (Port {service['port']})...")
        
        # Check if service directory and main.py exist
        if not service_path.exists():
            print(f"❌ Service directory not found: {service_path}")
            return False
            
        if not main_py.exists():
            print(f"❌ main.py not found in: {service_path}")
            return False
        
        try:
            # Change to service directory and start
            process = subprocess.Popen(
                [sys.executable, "main.py"],
                cwd=service_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            service["process"] = process
            self.running_services.append(service)
            
            print(f"✅ {service['name']} started with PID: {process.pid}")
            
            # Wait a moment for service to start
            time.sleep(2)
            
            # Test if service is responding
            if self.test_service_health(service["port"]):
                print(f"✅ {service['name']} is responding on port {service['port']}")
                return True
            else:
                print(f"⚠️  {service['name']} may need more time to start")
                return True
                
        except Exception as e:
            print(f"❌ Failed to start {service['name']}: {e}")
            return False
    
    def test_service_health(self, port, timeout=3):
        """Test if service is healthy"""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=timeout)
            return response.status_code == 200
        except:
            return False
    
    def test_all_services(self):
        """Test all services health"""
        print("\n🧪 Testing All Services:")
        print("========================")
        
        all_healthy = True
        for service in self.services:
            print(f"Testing {service['name']} (Port {service['port']}): ", end="")
            if self.test_service_health(service["port"]):
                print("✅ HEALTHY")
            else:
                print("❌ NOT RESPONDING")
                all_healthy = False
        
        return all_healthy
    
    def start_all_services(self):
        """Start all Phase 2 services"""
        print("🚀 Starting NAVA Phase 2 Services")
        print("==================================")
        print("Starting Phase 2 Intelligence Enhancement Services...")
        print()
        
        success_count = 0
        for service in self.services:
            if self.start_service(service):
                success_count += 1
            print()
        
        print("==================================")
        print(f"🎯 {success_count}/{len(self.services)} services started successfully")
        print()
        
        # Wait for services to fully initialize
        print("⏳ Waiting 10 seconds for services to fully start...")
        time.sleep(10)
        
        # Test all services
        all_healthy = self.test_all_services()
        
        print()
        if all_healthy:
            print("🎉 All Phase 2 Services are healthy and ready!")
        else:
            print("⚠️  Some services may need more time or have issues")
        
        self.show_service_info()
        
        return all_healthy
    
    def show_service_info(self):
        """Show service information"""
        print()
        print("📋 Service Information:")
        print("======================")
        print("Health Check URLs:")
        for service in self.services:
            print(f"   - {service['name']}: http://localhost:{service['port']}/health")
        
        print()
        print("📖 API Documentation:")
        for service in self.services:
            print(f"   - {service['name']}: http://localhost:{service['port']}/docs")
        
        print()
        print("🔧 Management Commands:")
        print("   - To stop all services: Ctrl+C or close this window")
        print("   - To check individual service: curl http://localhost:PORT/health")
    
    def stop_all_services(self):
        """Stop all running services"""
        print("\n🛑 Stopping all services...")
        for service in self.running_services:
            if service["process"] and service["process"].poll() is None:
                try:
                    print(f"Stopping {service['name']}...")
                    if os.name == 'nt':  # Windows
                        service["process"].terminate()
                    else:  # Unix/Linux
                        service["process"].terminate()
                    
                    # Wait for process to terminate
                    service["process"].wait(timeout=5)
                    print(f"✅ {service['name']} stopped")
                except subprocess.TimeoutExpired:
                    print(f"⚠️  Force killing {service['name']}...")
                    service["process"].kill()
                except Exception as e:
                    print(f"❌ Error stopping {service['name']}: {e}")
        
        print("🛑 All services stopped")
    
    def monitor_services(self):
        """Monitor services and keep them running"""
        print("\n👀 Monitoring services (Ctrl+C to stop)...")
        print("==========================================")
        
        try:
            while True:
                time.sleep(30)  # Check every 30 seconds
                
                print("\n📊 Service Health Check...")
                for service in self.running_services:
                    if service["process"].poll() is not None:
                        print(f"❌ {service['name']} has stopped! Restarting...")
                        self.start_service(service)
                    elif self.test_service_health(service["port"]):
                        print(f"✅ {service['name']}: Healthy")
                    else:
                        print(f"⚠️  {service['name']}: Not responding")
                        
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
            self.stop_all_services()

def signal_handler(signum, frame):
    """Handle termination signals"""
    print("\n🛑 Received termination signal, stopping services...")
    manager.stop_all_services()
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    manager = ServiceManager()
    
    try:
        # Start all services
        success = manager.start_all_services()
        
        if success:
            print("\n🚀 Ready to run Phase 2 comprehensive tests!")
            print("   Run your test script now...")
            
            # Keep services running
            manager.monitor_services()
        else:
            print("\n❌ Some services failed to start properly")
            print("   Check the error messages above and fix issues")
            manager.stop_all_services()
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        manager.stop_all_services()
        sys.exit(1)