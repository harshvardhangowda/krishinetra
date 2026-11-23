# KrishiNetra Enterprise Backend

Production-grade FastAPI platform for KrishiNetra autonomous agriculture robotics, non-chemical pest management, and AI evidence verification.

## 🚀 Quick Start Guide

### 1. Install Dependencies
```powershell
# From the project repository root:
pip install -r backend/requirements_backend.txt
```

### 2. Apply Database Migrations
```powershell
# From project repository root:
cd backend
alembic upgrade head
cd ..
```

### 3. Bootstrap First-Run Administrator (KN-01)
For safety and security, production default credentials are not auto-seeded. Use the bootstrap CLI to provision the initial organization administrator:
```powershell
python backend/scripts/bootstrap_admin.py --username admin --email admin@krishinetra.ai --password "YourStrongPassword123!" --org-name "KrishiNetra Agro Labs"
```

### 4. Start the FastAPI Application
```powershell
# Run from repository root:
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive documentation:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Live**: [http://localhost:8000/health/live](http://localhost:8000/health/live)
- **Health Ready**: [http://localhost:8000/health/ready](http://localhost:8000/health/ready)

### 5. Running Test Suite
```powershell
# Run full backend acceptance and integration test suite:
pytest backend/tests/test_backend_api.py -v
```

---

## 🔒 Security Architecture
- **Multi-Tenant Isolation**: Every rover, telemetry stream, inspection frame, and human review is strictly scoped to the authenticated organization.
- **Fail-Closed Actuation**: Unsupported commands and missing microcontrollers return explicit failure; physical motors and emitters remain permanently disabled by hardware interlocks.
- **Tamper-Evident Evidence**: Every inspection frame is indexed with SHA-256 integrity hashes and preserved non-destructively alongside human entomological reviews.
