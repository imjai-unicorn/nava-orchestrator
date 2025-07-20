# 🚀 NAVA - Logic Orchestrator

Intelligent Logic model selection and routing system.

## 🎯 Phase 1 Complete

- ✅ Core Logic Controller
- ✅ Decision Engine (Rule-based)
- ✅ Database Integration (Supabase)
- ✅ API Endpoints (`/chat`, `/health`)
- ✅ Mock AI Responses

## 📊 System Status

**Backend Services:**
- `nava-logic-controller` → Port 8005 ✅

**Database:**
- Supabase → Connected ✅

**API Documentation:**
- http://localhost:8005/docs

## 🚀 Quick Start

```bash
# 1. Setup environment
cd backend/services/01-core/nava-logic-controller
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
# Copy .env.example to .env and fill in Supabase credentials

# 3. Run service
python run.py
