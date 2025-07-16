#!/bin/bash
# scripts/start_internal_dashboard.sh

echo "🚀 Starting NAVA Internal Developer Dashboard..."

# Navigate to dashboard directory
cd frontend/internal-dev-dashboard

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "❌ package.json not found. Creating..."
    
    # Create package.json if it doesn't exist
    cat > package.json << 'EOF'
{
  "name": "nava-internal-dashboard",
  "version": "1.0.0",
  "description": "NAVA Enterprise Internal Developer Dashboard",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "web-vitals": "^3.3.2"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  },
  "proxy": "http://localhost:8005"
}
EOF
fi

# Check if src directory exists
if [ ! -d "src" ]; then
    echo "📁 Creating src directory..."
    mkdir -p src
fi

# Check if src/components directory exists
if [ ! -d "src/components" ]; then
    echo "📁 Creating src/components directory..."
    mkdir -p src/components
fi

# Check if public directory has index.html
if [ ! -f "public/index.html" ]; then
    echo "📄 Creating public/index.html..."
    mkdir -p public
    # The index.html content would be created here
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Check if NAVA backend is running
echo "🔍 Checking NAVA backend connection..."
if curl -s http://localhost:8005/api/health > /dev/null; then
    echo "✅ NAVA backend is running on port 8005"
else
    echo "⚠️  NAVA backend not detected on port 8005"
    echo "   Please start NAVA backend first:"
    echo "   cd backend/services/01-core/nava-logic-controller"
    echo "   python main.py"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start dashboard on port 3001
echo "🌐 Starting dashboard on http://localhost:3001..."
PORT=3001 npm start

echo "✅ Dashboard should open automatically in your browser"
echo "   If not, visit: http://localhost:3001"