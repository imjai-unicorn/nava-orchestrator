# 🚀 NAVA - Neural Allocation & Workflow Assistant

Advanced AI orchestration platform with intelligent model selection and microservice architecture.

## 🎯 Production Ready - Phase 2 Complete

- ✅ **NAVA Main Orchestrator** (Port 8005) - Railway Deployed
- ✅ **Real AI Integration** (GPT, Claude, Gemini)
- ✅ **Enhanced Decision Engine** - Intelligent model selection
- ✅ **Quality Validation Service** - Multi-dimensional scoring
- ✅ **Cache Engine** - Semantic similarity caching
- ✅ **SLF Framework** - Systematic reasoning enhancement
- ✅ **Microservice Architecture** - Pure microservices on Railway

## 🌐 Live Services

**Production URLs:**
- **Main Orchestrator:** https://nava-orchestrator-production.up.railway.app
- **API Documentation:** https://nava-orchestrator-production.up.railway.app/docs
- **Health Check:** https://nava-orchestrator-production.up.railway.app/health

**AI Services:**
- **GPT Service:** https://nava-orchestrator-gpt-production.up.railway.app
- **Claude Service:** https://nava-orchestrator-claude-production.up.railway.app  
- **Gemini Service:** https://nava-orchestrator-gemini-production.up.railway.app

**Enhanced Intelligence Services:**
- **Decision Engine:** https://nava-decision-engine-production.up.railway.app
- **Quality Service:** https://nava-quality-service-production.up.railway.app
- **Cache Engine:** https://nava-cache-engine-production.up.railway.app
- **SLF Framework:** https://nava-slf-framework-production.up.railway.app

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   NAVA Main     │    │  Decision       │    │  Quality        │
│  Orchestrator   │◄──►│   Engine        │◄──►│   Service       │
│   (Port 8005)   │    │  (Port 8008)    │    │  (Port 8009)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Services   │    │   Cache Engine  │    │  SLF Framework  │
│ GPT│Claude│Gemini│   │   (Port 8013)   │    │  (Port 8010)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Usage Examples

### **Enhanced Chat API:**
```bash
curl -X POST https://nava-orchestrator-production.up.railway.app/api/chat/enhanced \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello NAVA!", "user_id": "demo"}'
```

### **Health Check:**
```bash
curl https://nava-orchestrator-production.up.railway.app/health
```

### **System Status:**
```bash
curl https://nava-orchestrator-production.up.railway.app/api/admin/system-status
```

## 🔧 Local Development

```bash
# 1. Clone repository
git clone https://github.com/injai-unicorn/nava-orchestrator.git
cd nava-orchestrator

# 2. Setup Main Orchestrator
cd backend/services/01-core/nava-logic-controller
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Fill in your API keys and service URLs

# 4. Run locally
python run.py
```

## 📋 Environment Variables

```env
# AI Services
GPT_SERVICE_URL=https://nava-orchestrator-gpt-production.up.railway.app
CLAUDE_SERVICE_URL=https://nava-orchestrator-claude-production.up.railway.app
GEMINI_SERVICE_URL=https://nava-orchestrator-gemini-production.up.railway.app

# Enhanced Services
DECISION_ENGINE_URL=https://nava-decision-engine-production.up.railway.app
QUALITY_SERVICE_URL=https://nava-quality-service-production.up.railway.app
SLF_FRAMEWORK_URL=https://nava-slf-framework-production.up.railway.app
CACHE_ENGINE_URL=https://nava-cache-engine-production.up.railway.app

# Database
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_key
```

## 🎯 Features

- **🧠 Intelligent Model Selection** - Automatic AI model routing based on request complexity
- **⚡ Response Caching** - Semantic similarity-based response optimization  
- **📊 Quality Scoring** - Multi-dimensional response quality validation
- **🔄 Workflow Orchestration** - Sequential and parallel AI task execution
- **📈 Learning System** - Continuous improvement through user feedback
- **🛡️ Circuit Breaker** - Resilient service communication with automatic failover
- **📝 Comprehensive Logging** - Complete audit trail and performance metrics
- **🌐 RESTful API** - OpenAPI 3.0 compliant with interactive documentation

## 🧪 Testing

**Test Suite Status:** ✅ 206 tests passing

```bash
# Run tests
cd backend/services/01-core/nava-logic-controller
python -m pytest tests/ -v
```

## 📚 API Documentation

- **Interactive Docs:** https://nava-orchestrator-production.up.railway.app/docs
- **OpenAPI Schema:** https://nava-orchestrator-production.up.railway.app/openapi.json

## 🔄 Development Status

- **Phase 1:** ✅ Core Logic Controller (Complete)
- **Phase 2:** ✅ Intelligence Services (Complete) 
- **Phase 3:** 🚧 Enterprise Features (In Progress)
- **Phase 4:** 📋 Local AI Integration (Planned)
- **Phase 5:** 📋 Advanced Analytics (Planned)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**🚀 NAVA - Intelligence at Scale**"# Force NAVA Main redeploy $(date)" 
