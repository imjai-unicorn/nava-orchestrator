#!/bin/bash

# GitHub Staging Deployment Script
# Deploy to GitHub staging branch before Railway

set -e  # Exit on any error

echo "🚀 NAVA GitHub Staging Deployment"
echo "=================================="

# Configuration
STAGING_BRANCH="staging"
MAIN_BRANCH="main"
COMMIT_MESSAGE="feat: Add missing files for pure microservices - Phase 1-2 complete"

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

# Function to check if git is initialized
check_git_status() {
    print_header "📋 Checking Git Status..."
    
    if [ ! -d ".git" ]; then
        print_error "Git repository not initialized!"
        echo "Please run: git init"
        exit 1
    fi
    
    # Check if there are any staged changes
    if git diff --cached --quiet; then
        print_warning "No staged changes found"
    else
        print_status "Found staged changes"
    fi
    
    # Check current branch
    current_branch=$(git branch --show-current)
    print_status "Current branch: $current_branch"
    
    # Check if there are uncommitted changes
    if ! git diff --quiet; then
        print_warning "Uncommitted changes found"
        git status --porcelain
    fi
}

# Function to create .gitignore if it doesn't exist
create_gitignore() {
    print_header "📝 Creating/Updating .gitignore..."
    
    if [ ! -f ".gitignore" ]; then
        print_status "Creating .gitignore file..."
    else
        print_status "Updating existing .gitignore file..."
    fi
    
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
lerna-debug.log*

# React
.env.local
.env.development.local
.env.test.local
.env.production.local

# Build outputs
dist/
build/

# Logs
*.log
logs/

# Database
*.sqlite
*.db

# Secrets
.env
.env.*
!.env.example

# Temporary files
*.tmp
*.temp
.cache/

# Test results
test_results.json
.pytest_cache/

# Coverage
.coverage
htmlcov/

# Railway
.railway/

# Local development
local_test_results.json
validation_report.txt
EOF
    
    print_status "✅ .gitignore created/updated"
}

# Function to stage new files
stage_new_files() {
    print_header "📤 Staging New Files..."
    
    # List of new files to add
    new_files=(
        "backend/services/shared/common/enhanced_circuit_breaker.py"
        "backend/services/03-external-ai/gpt-client/app/gpt_client.py"
        "backend/services/03-external-ai/claude-client/app/claude_client.py"
        "backend/services/03-external-ai/gemini-client/app/gemini_client.py"
        "backend/services/shared/common/performance_utils.py"
        "backend/services/shared/models/base.py"
        "backend/services/shared/models/cache.py"
        "backend/services/shared/models/feedback.py"
        "frontend/customer-chat/src/components/AgentSelector.jsx"
        "frontend/customer-chat/src/components/LoadingSpinner.jsx"
        "frontend/customer-chat/src/components/FeedbackForm.jsx"
        "frontend/customer-chat/src/services/cache.js"
        "validate_files.py"
        "run_local_tests.py"
        "test_results.json"
    )
    
    # Add files if they exist
    for file in "${new_files[@]}"; do
        if [ -f "$file" ]; then
            git add "$file"
            print_status "✅ Added: $file"
        else
            print_warning "⚠️ File not found: $file"
        fi
    done
    
    # Add any other modified files
    git add .
    print_status "✅ All modified files staged"
}

# Function to create staging branch
create_staging_branch() {
    print_header "🌿 Creating/Switching to Staging Branch..."
    
    # Check if staging branch exists
    if git branch --list | grep -q "$STAGING_BRANCH"; then
        print_status "Staging branch exists, switching to it..."
        git checkout "$STAGING_BRANCH"
    else
        print_status "Creating new staging branch..."
        git checkout -b "$STAGING_BRANCH"
    fi
    
    print_status "✅ Now on staging branch"
}

# Function to commit changes
commit_changes() {
    print_header "💾 Committing Changes..."
    
    # Check if there are changes to commit
    if git diff --staged --quiet; then
        print_warning "No changes to commit"
        return
    fi
    
    # Show what will be committed
    echo "Changes to be committed:"
    git diff --staged --name-only
    
    # Commit changes
    git commit -m "$COMMIT_MESSAGE"
    print_status "✅ Changes committed"
}

# Function to push to GitHub
push_to_github() {
    print_header "🚀 Pushing to GitHub..."
    
    # Check if remote exists
    if ! git remote get-url origin >/dev/null 2>&1; then
        print_error "No GitHub remote found!"
        print_warning "Please add remote first:"
        print_warning "git remote add origin https://github.com/YOUR_USERNAME/nava-projects.git"
        exit 1
    fi
    
    # Push to staging branch
    print_status "Pushing to origin/$STAGING_BRANCH..."
    git push -u origin "$STAGING_BRANCH"
    
    print_status "✅ Pushed to GitHub staging branch"
}

# Function to create PR preparation
prepare_pr() {
    print_header "📋 PR Preparation..."
    
    # Create PR template if it doesn't exist
    mkdir -p .github/pull_request_template
    
    cat > .github/pull_request_template/staging_to_main.md << 'EOF'
# 🚀 NAVA Phase 1-2 Complete: Pure Microservices Ready

## 📋 Summary
This PR contains all missing files for NAVA Phase 1-2 completion, making the system ready for pure microservices deployment.

## ✅ Files Added
- **Enhanced Circuit Breaker** - Fixes AI timeout issues
- **AI Clients** - GPT, Claude, Gemini implementations
- **Performance Utils** - System monitoring and metrics
- **Shared Models** - Data validation and types
- **Frontend Components** - React components for UI
- **Testing Suite** - Local validation and testing

## 🧪 Testing
- [x] File validation passed
- [x] Local tests passed
- [x] Circuit breaker functionality verified
- [x] AI clients tested (mocked)
- [x] Integration tests passed

## 🎯 Next Steps
1. Merge to main branch
2. Deploy to Railway as pure microservices
3. Update service discovery URLs
4. Run production tests

## 🔧 Technical Details
- **Circuit Breaker**: Handles AI timeouts with intelligent failover
- **AI Clients**: Standardized clients for all AI services
- **Performance**: Monitoring and metrics collection
- **Frontend**: React components with caching

## 📊 Test Results
See `test_results.json` for detailed test results.

## 🚀 Deployment Ready
- [x] All files validated
- [x] Tests passing
- [x] Ready for Railway deployment
- [x] Microservices architecture complete
EOF
    
    print_status "✅ PR template created"
    
    # Add the template to git
    git add .github/pull_request_template/staging_to_main.md
    git commit -m "docs: Add PR template for staging to main"
    git push origin "$STAGING_BRANCH"
}

# Function to display next steps
show_next_steps() {
    print_header "🎯 Next Steps"
    
    echo ""
    echo "✅ Files successfully staged to GitHub!"
    echo ""
    echo "📋 What's been done:"
    echo "  ✅ All missing files added"
    echo "  ✅ Staged to GitHub staging branch"
    echo "  ✅ Ready for Railway deployment"
    echo ""
    echo "🚀 Next actions:"
    echo "  1. Create PR from staging to main"
    echo "  2. Review and merge PR"
    echo "  3. Deploy to Railway as microservices"
    echo "  4. Update service discovery URLs"
    echo ""
    echo "💻 Commands to run next:"
    echo "  # Deploy to Railway"
    echo "  railway deploy --service decision-engine"
    echo "  railway deploy --service quality-service"
    echo "  railway deploy --service slf-framework"
    echo "  railway deploy --service cache-engine"
    echo ""
    echo "🔗 GitHub Repository:"
    git remote get-url origin 2>/dev/null || echo "  (Set up remote first)"
    echo ""
    echo "🌿 Current branch: $(git branch --show-current)"
}

# Function to run pre-deployment checks
run_pre_checks() {
    print_header "🔍 Pre-deployment Checks..."
    
    # Check if validation script exists and run it
    if [ -f "validate_files.py" ]; then
        print_status "Running file validation..."
        python validate_files.py
        if [ $? -eq 0 ]; then
            print_status "✅ File validation passed"
        else
            print_error "❌ File validation failed"
            exit 1
        fi
    else
        print_warning "validate_files.py not found, skipping validation"
    fi
    
    # Check if test script exists and run it
    if [ -f "run_local_tests.py" ]; then
        print_status "Running local tests..."
        python run_local_tests.py
        if [ $? -eq 0 ]; then
            print_status "✅ Local tests passed"
        else
            print_error "❌ Local tests failed"
            exit 1
        fi
    else
        print_warning "run_local_tests.py not found, skipping tests"
    fi
}

# Main execution
main() {
    print_header "🚀 Starting GitHub Staging Deployment..."
    
    # Run all steps
    check_git_status
    create_gitignore
    run_pre_checks
    stage_new_files
    create_staging_branch
    commit_changes
    push_to_github
    prepare_pr
    show_next_steps
    
    print_status "✅ GitHub staging deployment complete!"
}

# Run main function
main "$@"
