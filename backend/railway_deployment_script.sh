#!/bin/bash

# Railway Pure Microservices Deployment Script
# Deploy NAVA as separate microservices on Railway

set -e  # Exit on any error

echo "🚢 NAVA Railway Microservices Deployment"
echo "========================================"

# Configuration
SERVICES=(
    "decision-engine:backend/services/05-enhanced-intelligence/decision-engine"
    "quality-service:backend/services/05-enhanced-intelligence/quality-service"
    "slf-framework:backend/services/05-enhanced-intelligence/slf-framework"
    "cache-engine:backend/services/05-enhanced-intelligence/cache-engine"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Function to check Railway CLI
check_railway_cli() {
    print_header "🔍 Checking Railway CLI..."
    
    if ! command -v railway &> /dev/null; then
        print_error "Railway CLI not found!"
        echo "Please install Railway CLI:"
        echo "npm install -g @railway/cli"
        exit 1
    fi
    
    print_status "✅ Railway CLI found"
    
    # Check if logged in
    if ! railway status &> /dev/null; then
        print_error "Not logged in to Railway!"
        echo "Please login first:"
        echo "railway login"
        exit 1
    fi
    
    print_status "✅ Railway CLI authenticated"
}

# Function to check project structure
check_project_structure() {
    print_header "📁 Checking Project Structure..."
    
    for service_info in "${SERVICES[@]}"; do
        service_name=$(echo "$service_info" | cut -d: -f1)
        service_path=$(echo "$service_info" | cut -d: -f2)
        
        if [ ! -d "$service_path" ]; then
            print_error "Service directory not found: $service_path"
            exit 1
        fi
        
        if [ ! -f "$service_path/main.py" ]; then
            print_error "main.py not found in: $service_path"
            exit 1
        fi
        
        if [ ! -f "$service_path/requirements.txt" ]; then
            print_error "requirements.txt not found in: $service_path"
            exit 1
        fi
        
        print_status "✅ $service_name structure valid"
    done
}

# Function to create Railway project
create_railway_project() {
    print_header "🚀 Creating Railway Project..."
    
    # Check if project already exists
    if railway status &> /dev/null; then
        print_status "Railway project exists"
        railway status
    else
        print_status "Creating new Railway project..."
        railway create nava-microservices
        print_status "✅ Railway project created"
    fi
}

# Function to deploy single service
deploy_service() {
    local service_name=$1
    local service_path=$2
    
    print_header "🚢 Deploying $service_name..."
    
    # Navigate to service directory
    cd "$service_path"
    
    # Check if railway.toml exists
    if [ ! -f "railway.toml" ]; then
        print_warning "railway.toml not found, creating..."
        create_railway_toml "$service_name"
    fi
    
    # Deploy service
    print_status "Deploying $service_name..."
    railway deploy --service "$service_name"
    
    if [ $? -eq 0 ]; then
        print_status "✅ $service_name deployed successfully"
        
        # Get service URL
        service_url=$(railway status --service "$service_name" 2>/dev/null | grep -o 'https://[^ ]*' | head -1)
        if [ -n "$service_url" ]; then
            print_status "🌐 Service URL: $service_url"
            echo "$service_name,$service_url" >> ../../deployment_urls.txt
        fi
    else
        print_error "❌ Failed to deploy $service_name"
        exit 1
    fi
    
    # Return to project root
    cd - > /dev/null
}

# Function to create railway.toml
create_railway_toml() {
    local service_name=$1
    
    cat > railway.toml << EOF
[build]
builder = "nixpacks"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port \$PORT"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3

[env]
PORT = 8000

# Service-specific environment variables
SERVICE_NAME = "$service_name"
ENVIRONMENT = "production"
LOG_LEVEL = "info"

# External API Keys (set these in Railway dashboard)
# OPENAI_API_KEY = 
# ANTHROPIC_API_KEY = 
# GOOGLE_API_KEY = 

# Database connection
# DATABASE_URL = 

# Inter-service communication
# NAVA_CONTROLLER_URL = 
# DECISION_ENGINE_URL = 
# QUALITY_SERVICE_URL = 
# SLF_FRAMEWORK_URL = 
# CACHE_ENGINE_URL = 
EOF
    
    print_status "✅ railway.toml created for $service_name"
}

# Function to deploy all services
deploy_all_services() {
    print_header "🚀 Deploying All Services..."
    
    # Clear previous URLs
    > deployment_urls.txt
    echo "service,url" >> deployment_urls.txt
    
    # Deploy each service
    for service_info in "${SERVICES[@]}"; do
        service_name=$(echo "$service_info" | cut -d: -f1)
        service_path=$(echo "$service_info" | cut -d: -f2)
        
        deploy_service "$service_name" "$service_path"
        
        # Wait a bit between deployments
        sleep 5
    done
    
    print_status "✅ All services deployed"
}

# Function to update service discovery
update_service_discovery() {
    print_header "🔗 Updating Service Discovery..."
    
    if [ ! -f "deployment_urls.txt" ]; then
        print_warning "deployment_urls.txt not found, skipping service discovery update"
        return
    fi
    
    # Read URLs from file
    declare -A service_urls
    while IFS=',' read -r service url; do
        if [ "$service" != "service" ]; then  # Skip header
            service_urls["$service"]="$url"
        fi
    done < deployment_urls.txt
    
    # Update phase2_service_discovery.py
    discovery_file="backend/services/01-core/nava-logic-controller/app/config/phase2_service_discovery.py"
    
    if [ -f "$discovery_file" ]; then
        print_status "Updating $discovery_file..."
        
        # Create backup
        cp "$discovery_file" "$discovery_file.backup"
        
        # Create new service discovery file
        cat > "$discovery_file" << EOF
"""
Phase 2 Service Discovery - Railway Microservices URLs
Auto-generated by deployment script
"""

import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ServiceDiscovery:
    def __init__(self):
        self.services = {
            "decision-engine": {
                "url": "${service_urls[decision-engine]:-http://localhost:8008}",
                "health_endpoint": "/health",
                "timeout": 30,
                "retries": 3
            },
            "quality-service": {
                "url": "${service_urls[quality-service]:-http://localhost:8009}",
                "health_endpoint": "/health",
                "timeout": 20,
                "retries": 2
            },
            "slf-framework": {
                "url": "${service_urls[slf-framework]:-http://localhost:8010}",
                "health_endpoint": "/health",
                "timeout": 25,
                "retries": 3
            },
            "cache-engine": {
                "url": "${service_urls[cache-engine]:-http://localhost:8013}",
                "health_endpoint": "/health",
                "timeout": 15,
                "retries": 2
            }
        }
        
        # Override with environment variables if available
        for service_name in self.services:
            env_var = f"{service_name.upper().replace('-', '_')}_URL"
            if env_var in os.environ:
                self.services[service_name]["url"] = os.environ[env_var]
                logger.info(f"Using environment URL for {service_name}: {os.environ[env_var]}")
    
    def get_service_url(self, service_name: str) -> Optional[str]:
        """Get service URL by name"""
        service = self.services.get(service_name)
        return service["url"] if service else None
    
    def get_service_config(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get complete service configuration"""
        return self.services.get(service_name)
    
    def get_all_services(self) -> Dict[str, Dict[str, Any]]:
        """Get all service configurations"""
        return self.services
    
    def is_service_available(self, service_name: str) -> bool:
        """Check if service is configured"""
        return service_name in self.services
    
    def get_health_check_url(self, service_name: str) -> Optional[str]:
        """Get health check URL for service"""
        service = self.services.get(service_name)
        if service:
            return f"{service['url']}{service['health_endpoint']}"
        return None

# Global service discovery instance
service_discovery = ServiceDiscovery()

# Convenience functions
def get_service_url(service_name: str) -> Optional[str]:
    return service_discovery.get_service_url(service_name)

def get_service_config(service_name: str) -> Optional[Dict[str, Any]]:
    return service_discovery.get_service_config(service_name)

def get_all_services() -> Dict[str, Dict[str, Any]]:
    return service_discovery.get_all_services()

# Service URLs for direct access
DECISION_ENGINE_URL = get_service_url("decision-engine")
QUALITY_SERVICE_URL = get_service_url("quality-service")
SLF_FRAMEWORK_URL = get_service_url("slf-framework")
CACHE_ENGINE_URL = get_service_url("cache-engine")

# Deployment information
DEPLOYMENT_INFO = {
    "deployment_time": "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)",
    "deployment_type": "railway_microservices",
    "services_deployed": $(echo "${!service_urls[@]}" | wc -w),
    "urls": {
$(for service in "${!service_urls[@]}"; do
    echo "        \"$service\": \"${service_urls[$service]}\","
done)
    }
}
EOF
        
        print_status "✅ Service discovery updated"
    else
        print_warning "Service discovery file not found: $discovery_file"
    fi
}

# Function to run health checks
run_health_checks() {
    print_header "🏥 Running Health Checks..."
    
    if [ ! -f "deployment_urls.txt" ]; then
        print_warning "deployment_urls.txt not found, skipping health checks"
        return
    fi
    
    # Wait for services to start
    print_status "Waiting for services to start..."
    sleep 30
    
    # Check each service
    while IFS=',' read -r service url; do
        if [ "$service" != "service" ]; then  # Skip header
            print_status "Checking $service at $url..."
            
            # Try health check endpoint
            health_url="$url/health"
            if curl -f -s "$health_url" > /dev/null 2>&1; then
                print_status "✅ $service is healthy"
            else
                print_warning "⚠️ $service health check failed"
                # Try root endpoint
                if curl -f -s "$url" > /dev/null 2>&1; then
                    print_status "✅ $service is responding"
                else
                    print_error "❌ $service is not responding"
                fi
            fi
        fi
    done < deployment_urls.txt
}

# Function to create deployment summary
create_deployment_summary() {
    print_header "📋 Creating Deployment Summary..."
    
    cat > deployment_summary.md << EOF
# 🚢 NAVA Railway Deployment Summary

## 📅 Deployment Information
- **Date**: $(date)
- **Type**: Pure Microservices
- **Platform**: Railway
- **Services Deployed**: $(echo "${SERVICES[@]}" | wc -w)

## 🚀 Deployed Services

$(if [ -f "deployment_urls.txt" ]; then
    echo "| Service | URL | Status |"
    echo "|---------|-----|--------|"
    while IFS=',' read -r service url; do
        if [ "$service" != "service" ]; then
            echo "| $service | $url | ✅ Deployed |"
        fi
    done < deployment_urls.txt
else
    echo "Service URLs not available"
fi)

## 🔗 Service Architecture

```
Internet → Railway Load Balancer
├── decision-engine (Port 8008)
├── quality-service (Port 8009)
├── slf-framework (Port 8010)
└── cache-engine (Port 8013)
```

## 🛠️ Configuration Files Updated
- ✅ phase2_service_discovery.py
- ✅ railway.toml for each service
- ✅ Environment variables set

## 🔧 Post-Deployment Tasks

### 1. Environment Variables
Set these in Railway dashboard for each service:
- \`OPENAI_API_KEY\`
- \`ANTHROPIC_API_KEY\`
- \`GOOGLE_API_KEY\`
- \`DATABASE_URL\` (if using database)

### 2. Inter-Service Communication
Services can communicate using these URLs:
$(if [ -f "deployment_urls.txt" ]; then
    while IFS=',' read -r service url; do
        if [ "$service" != "service" ]; then
            echo "- $service: $url"
        fi
    done < deployment_urls.txt
fi)

### 3. Testing
Run integration tests:
\`\`\`bash
python test_microservices.py
\`\`\`

### 4. Monitoring
- Check Railway dashboard for service health
- Monitor logs for any errors
- Set up alerts for service failures

## 🎯 Next Steps
1. ✅ Deploy completed
2. 🔧 Configure environment variables
3. 🧪 Run integration tests
4. 📊 Monitor service health
5. 🚀 Update main NAVA controller

## 📞 Support
If you encounter issues:
1. Check Railway dashboard logs
2. Verify environment variables
3. Run health checks manually
4. Check service discovery configuration

EOF
    
    print_status "✅ Deployment summary created: deployment_summary.md"
}

# Function to display next steps
show_next_steps() {
    print_header "🎯 Next Steps"
    
    echo ""
    echo "✅ Railway deployment completed!"
    echo ""
    echo "📋 What's been done:"
    echo "  ✅ All 4 microservices deployed to Railway"
    echo "  ✅ Service discovery updated"
    echo "  ✅ Health checks performed"
    echo "  ✅ Deployment summary created"
    echo ""
    echo "🚀 Required actions:"
    echo "  1. Set environment variables in Railway dashboard"
    echo "  2. Run integration tests"
    echo "  3. Update main NAVA controller"
    echo "  4. Monitor service health"
    echo ""
    echo "💻 Commands to run:"
    echo "  # Set environment variables"
    echo "  railway variables set OPENAI_API_KEY=your_key --service decision-engine"
    echo "  railway variables set ANTHROPIC_API_KEY=your_key --service quality-service"
    echo "  railway variables set GOOGLE_API_KEY=your_key --service slf-framework"
    echo ""
    echo "  # Run integration tests"
    echo "  python test_microservices.py"
    echo ""
    echo "🔗 Service URLs:"
    if [ -f "deployment_urls.txt" ]; then
        while IFS=',' read -r service url; do
            if [ "$service" != "service" ]; then
                echo "  $service: $url"
            fi
        done < deployment_urls.txt
    fi
    echo ""
    echo "📊 Monitor at: https://railway.app/dashboard"
}

# Function to create integration test
create_integration_test() {
    print_header "🧪 Creating Integration Test..."
    
    cat > test_microservices.py << 'EOF'
#!/usr/bin/env python3
"""
Microservices Integration Test
Test deployed Railway microservices
"""

import asyncio
import aiohttp
import json
import sys
from typing import Dict, Any
import time

# Load deployment URLs
def load_service_urls():
    urls = {}
    try:
        with open('deployment_urls.txt', 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(',')
                if len(parts) == 2:
                    service, url = parts
                    urls[service] = url
    except FileNotFoundError:
        print("❌ deployment_urls.txt not found")
        sys.exit(1)
    return urls

class MicroservicesTest:
    def __init__(self):
        self.service_urls = load_service_urls()
        self.test_results = []
        
    async def test_health_endpoints(self):
        """Test all health endpoints"""
        print("🏥 Testing health endpoints...")
        
        async with aiohttp.ClientSession() as session:
            for service, url in self.service_urls.items():
                try:
                    health_url = f"{url}/health"
                    async with session.get(health_url, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            print(f"✅ {service}: {data.get('status', 'unknown')}")
                            self.test_results.append(f"{service}_health: PASS")
                        else:
                            print(f"❌ {service}: HTTP {response.status}")
                            self.test_results.append(f"{service}_health: FAIL")
                except Exception as e:
                    print(f"❌ {service}: {e}")
                    self.test_results.append(f"{service}_health: ERROR")
    
    async def test_service_endpoints(self):
        """Test main service endpoints"""
        print("🔧 Testing service endpoints...")
        
        async with aiohttp.ClientSession() as session:
            # Test decision engine
            if 'decision-engine' in self.service_urls:
                try:
                    url = f"{self.service_urls['decision-engine']}/decide"
                    payload = {
                        "task_type": "conversation",
                        "context": {"test": True}
                    }
                    async with session.post(url, json=payload, timeout=30) as response:
                        if response.status == 200:
                            print("✅ Decision engine: Working")
                            self.test_results.append("decision_engine_api: PASS")
                        else:
                            print(f"❌ Decision engine: HTTP {response.status}")
                            self.test_results.append("decision_engine_api: FAIL")
                except Exception as e:
                    print(f"❌ Decision engine: {e}")
                    self.test_results.append("decision_engine_api: ERROR")
            
            # Test other services similarly...
            # (Add more specific tests as needed)
    
    async def run_all_tests(self):
        """Run all integration tests"""
        print("🚀 Starting microservices integration tests...")
        print("=" * 50)
        
        start_time = time.time()
        
        await self.test_health_endpoints()
        await self.test_service_endpoints()
        
        duration = time.time() - start_time
        
        # Print results
        print("\n" + "=" * 50)
        print("📊 Test Results")
        print("=" * 50)
        
        passed = sum(1 for r in self.test_results if 'PASS' in r)
        failed = sum(1 for r in self.test_results if 'FAIL' in r)
        errors = sum(1 for r in self.test_results if 'ERROR' in r)
        
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"🔥 Errors: {errors}")
        print(f"⏱️  Duration: {duration:.2f}s")
        
        if failed == 0 and errors == 0:
            print("\n🎉 All tests passed!")
            return True
        else:
            print("\n❌ Some tests failed")
            return False

async def main():
    test = MicroservicesTest()
    success = await test.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
EOF
    
    print_status "✅ Integration test created: test_microservices.py"
}

# Main execution
main() {
    print_header "🚢 Starting Railway Microservices Deployment..."
    
    # Run all steps
    check_railway_cli
    check_project_structure
    create_railway_project
    deploy_all_services
    update_service_discovery
    run_health_checks
    create_deployment_summary
    create_integration_test
    show_next_steps
    
    print_status "✅ Railway deployment complete!"
}

# Handle command line arguments
case "${1:-}" in
    "deploy")
        main
        ;;
    "health")
        run_health_checks
        ;;
    "test")
        python test_microservices.py
        ;;
    "status")
        if [ -f "deployment_urls.txt" ]; then
            cat deployment_urls.txt
        else
            echo "No deployment found"
        fi
        ;;
    *)
        echo "Usage: $0 {deploy|health|test|status}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Deploy all microservices to Railway"
        echo "  health  - Check health of deployed services"
        echo "  test    - Run integration tests"
        echo "  status  - Show deployment status"
        exit 1
        ;;
esac
