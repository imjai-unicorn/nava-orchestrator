# 🤖 NAVA Agent Registry

**AI Service Discovery, Health Monitoring, and Load Balancing for NAVA Enterprise**

## 📋 Overview

The Agent Registry is the central service discovery and load balancing system for NAVA Enterprise. It manages all AI services, monitors their health, and intelligently routes requests to the best available service.

### 🎯 Key Features

- **Service Discovery**: Automatic registration and discovery of AI services
- **Health Monitoring**: Real-time health checks and status tracking
- **Load Balancing**: Intelligent routing based on performance, cost, and availability
- **Failover Management**: Automatic failover chains (GPT → Claude → Gemini → Local AI)
- **Performance Tracking**: Response time and success rate monitoring
- **Enterprise Features**: Audit logging, compliance tracking, and analytics
- **Local AI Ready**: Built-in support for Phase 4 local AI integration

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  NAVA Logic     │───▶│  Agent Registry  │───▶│   AI Services   │
│  Controller     │    │    (Port 8006)   │    │  (8002,8003,    │
│  (Port 8005)    │    │                  │    │   8004,etc.)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Health        │
                       │   Monitoring    │
                       │   & Analytics   │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Supabase account and database
- Redis (optional, for caching)

### 1. Installation

```bash
# Clone or create the agent-registry directory
cd backend/services/01-core/agent-registry

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 2. Environment Configuration

Edit `.env` file with your actual values:

```bash
# Database (get from Supabase dashboard)
DATABASE_URL=postgresql://postgres:your_password@db.your_project.supabase.co:5432/postgres
SUPABASE_URL=https://your_project.supabase.co
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_key

# AI Services URLs (development)
GPT_CLIENT_URL=http://localhost:8002
CLAUDE_CLIENT_URL=http://localhost:8003
GEMINI_CLIENT_URL=http://localhost:8004
```

### 3. Database Setup

Run the database migration script:

```bash
# Create Agent Registry tables in Supabase
python scripts/setup_database.py
```

### 4. Start the Service

```bash
# Development mode
python main.py

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8006 --reload
```

## 🐳 Docker Development

### Local Development

```bash
# Start Agent Registry + Redis
docker-compose up agent-registry redis

# With mock AI services for testing
docker-compose --profile mock-services up

# Full development environment
docker-compose --profile with-controller --profile mock-services up
```

### Docker Commands

```bash
# Build image
docker build -t nava/agent-registry .

# Run container
docker run -p 8006:8006 --env-file .env nava/agent-registry
```

## 🚄 Railway Deployment

### Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Deploy
railway up
```

### Environment Variables

Set these in Railway dashboard:
- `DATABASE_URL`
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`

Production URLs are automatically configured in `railway.toml`.

## 📊 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/services` | GET | List all services |
| `/api/services/select` | POST | Select best service |
| `/api/services/register` | POST | Register new service |
| `/api/services/register-local` | POST | Register local AI |
| `/api/services/{id}` | GET | Get service details |
| `/api/services/{id}` | DELETE | Unregister service |
| `/api/statistics` | GET | Registry statistics |
| `/api/healthy-services` | GET | List healthy services |

### Interactive Documentation

- **Swagger UI**: http://localhost:8006/docs
- **ReDoc**: http://localhost:8006/redoc

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=app

# Specific test file
pytest tests/test_agent_registry.py -v
```

### Test Categories

- **Unit Tests**: Core logic testing
- **Integration Tests**: Service interaction testing
- **Health Check Tests**: Monitoring system testing
- **Load Balancing Tests**: Selection algorithm testing

## 🔧 Configuration

### Service Selection Strategies

```python
# Cost-based selection
POST /api/services/select
{
    "cost_priority": true
}

# Performance-based selection
POST /api/services/select
{
    "performance_priority": true
}

# Capability-based selection
POST /api/services/select
{
    "capability": "reasoning"
}

# Model preference
POST /api/services/select
{
    "model_preference": "gpt-4"
}
```

### Health Monitoring

- **Check Interval**: 30 seconds (configurable)
- **Timeout**: 10 seconds
- **Failure Threshold**: 5 consecutive failures
- **Recovery Check**: 60 seconds

### Load Balancing

- **Strategy**: Priority-based (configurable)
- **Max Concurrent**: 10 per service
- **Global Limit**: 100 concurrent requests
- **Queue Timeout**: 30 seconds

## 🏢 Enterprise Features

### Audit Logging

All service selections and health changes are logged to:
- Database audit tables
- External audit service (if configured)
- Structured logs

### Compliance Tracking

- **Data Residency**: Track where data is processed
- **Privacy Requirements**: Enforce privacy routing
- **Regulatory Compliance**: GDPR, SOX, HIPAA ready

### Analytics

- **Performance Metrics**: Response times, success rates
- **Cost Analytics**: Cost per request, optimization suggestions
- **Usage Patterns**: Service utilization trends

## 🤖 Local AI Integration (Phase 4)

### Register Local AI Services

```python
# Register Phi3-Mini
POST /api/services/register-local
{
    "service_id": "phi3-mini",
    "name": "Phi3 Mini Local AI",
    "port": 8019,
    "models": ["phi3-mini-4k", "phi3-mini-128k"],
    "capabilities": ["chat", "privacy", "fast_response"]
}

# Register DeepSeek Coder
POST /api/services/register-local
{
    "service_id": "deepseek-coder",
    "name": "DeepSeek Coder",
    "port": 8020,
    "models": ["deepseek-coder-6.7b"],
    "capabilities": ["coding", "analysis", "debugging"]
}
```

### Local AI Benefits

- **Zero Cost**: No API costs for local processing
- **Privacy**: Complete data residency
- **Speed**: Sub-500ms response times
- **Offline**: Works without internet

## 🔍 Monitoring & Observability

### Health Dashboard

```bash
# Check service health
curl http://localhost:8006/api/services

# Get statistics
curl http://localhost:8006/api/statistics

# View healthy services only
curl http://localhost:8006/api/healthy-services
```

### Prometheus Metrics

Enable Prometheus metrics:

```bash
ENABLE_PROMETHEUS=true
METRICS_PORT=9006
```

Access metrics at: http://localhost:9006/metrics

### Logging

Structured JSON logging with:
- Request/response tracking
- Health check events
- Load balancing decisions
- Performance metrics

## 🚨 Troubleshooting

### Common Issues

1. **Service Not Found**
   ```bash
   # Check if service is registered
   curl http://localhost:8006/api/services
   ```

2. **Health Check Failing**
   ```bash
   # Check service endpoints
   curl http://localhost:8002/health  # GPT
   curl http://localhost:8003/health  # Claude
   curl http://localhost:8004/health  # Gemini
   ```

3. **Database Connection**
   ```bash
   # Test database connection
   python scripts/test_database.py
   ```

4. **High Response Times**
   - Check AI service health
   - Verify network connectivity
   - Review timeout settings

### Debug Mode

Enable debug logging:

```bash
DEBUG=true
LOG_LEVEL=DEBUG
ENABLE_DEBUG_LOGGING=true
```

## 📚 Development

### Project Structure

```
agent-registry/
├── app/
│   └── agent_registry.py      # Core logic
├── tests/
│   └── test_agent_registry.py # Test suite
├── scripts/
│   ├── setup_database.py      # Database setup
│   └── test_database.py       # Database testing
├── main.py                    # FastAPI app
├── requirements.txt           # Dependencies
├── Dockerfile                 # Container
├── docker-compose.yml         # Local dev
├── railway.toml              # Railway config
└── README.md                 # This file
```

### Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit pull request

### Code Style

- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **pytest**: Testing framework

## 🔗 Integration

### NAVA Logic Controller Integration

```python
# In NAVA Logic Controller
import httpx

async def select_ai_service(capability: str = None):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8006/api/services/select",
            json={"capability": capability}
        )
        return response.json()
```

### Frontend Integration

```javascript
// In React dashboard
const getServices = async () => {
    const response = await fetch('/api/services');
    return response.json();
};

const selectService = async (criteria) => {
    const response = await fetch('/api/services/select', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(criteria)
    });
    return response.json();
};
```

## 📈 Performance

### Benchmarks

- **Service Selection**: <50ms
- **Health Check**: <100ms per service
- **Registration**: <10ms
- **Statistics**: <20ms

### Optimization Tips

1. **Enable Caching**: Set `ENABLE_REGISTRY_CACHE=true`
2. **Adjust Check Intervals**: Increase for stable services
3. **Use Local AI**: For cost and speed optimization
4. **Monitor Metrics**: Track performance trends

## 🛡️ Security

### Security Features

- **JWT Authentication**: Token-based auth (optional)
- **Rate Limiting**: Prevent abuse
- **Input Validation**: Pydantic models
- **CORS Protection**: Configurable origins
- **Audit Logging**: Complete audit trail

### Security Checklist

- [ ] Change default JWT secret
- [ ] Enable API key authentication in production
- [ ] Configure CORS for production domains
- [ ] Enable HTTPS in production
- [ ] Set up monitoring alerts
- [ ] Regular security updates

## 📞 Support

### Documentation

- **API Docs**: http://localhost:8006/docs
- **System Architecture**: See NAVA Enterprise documentation
- **Deployment Guide**: See Railway/Docker sections

### Contact

- **Issues**: Create GitHub issue
- **Questions**: See NAVA Enterprise documentation
- **Contributions**: Submit pull request

---

**🚀 NAVA Agent Registry - Making AI service management intelligent and reliable!**