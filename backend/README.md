# KrishiNetra Enterprise Backend Platform

A secure, multi-tenant, asynchronous FastAPI platform for the KrishiNetra Precision Agriculture Robotics system.

---

## 🌟 Key Features

1. **Multi-Tenant Foundation & Security:**
   - Organization tenant isolation.
   - Cryptographic password hashing (`bcrypt`).
   - Session tokens with CSRF defense headers and login throttling.
   - Strict Role-Based Access Control (**RBAC**): `Owner`, `Operator`, `Reviewer`, and `Viewer`.

2. **Rover Control Authority & Safety Leases:**
   - Exclusive, time-bounded control leases (`RoverControlLease`).
   - Command lifecycle state machine: `PENDING -> SENT -> ACK_RECEIVED -> COMPLETED / FAILED / UNKNOWN`.
   - Idempotency protection to prevent duplicate network retries from executing twice.
   - **Non-bypassable Universal Emergency STOP**: Can be dispatched by any authorized role regardless of lease expiration.

3. **AI Evidence Platform & Inspections:**
   - SHA-256 tamper-evident media storage (`InspectionFrame`).
   - Integrated inference adapter with trained YOLOv8 PyTorch and ONNX models (`pest_yolov8n_best.onnx`).
   - Human-in-the-loop review workflow (`HumanReview`) that preserves raw AI predictions without destructive overwrites.

4. **Real-time Telemetry & WebSockets:**
   - Authenticated WebSocket streaming for live rover telemetry, sensor feeds, and detection alerts (`/ws/rovers/{rover_id}/telemetry`).

5. **Alembic Database Migrations:**
   - Full async SQLAlchemy 2.0 declarative schema supporting SQLite (local zero-setup) and PostgreSQL (production).

---

## 🚀 Getting Started

### 1. Installation
```powershell
pip install -r backend/requirements_backend.txt
```

### 2. Apply Database Migrations
```powershell
cd backend
alembic upgrade head
```

### 3. Start the FastAPI Development Server
```powershell
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Live**: [http://localhost:8000/health/live](http://localhost:8000/health/live)
- **Health Ready**: [http://localhost:8000/health/ready](http://localhost:8000/health/ready)

### 4. Running Backend Test Suite
```powershell
pytest backend/tests/test_backend_api.py -v
```

---

## 🔒 Default Bootstrap Credentials (Development Mode)
- **Organization**: `KrishiNetra Agro Labs` (Slug: `krishinetra-labs`)
- **Username**: `admin`
- **Password**: `KrishiNetra@2026`
- **Role**: `owner`
- **Simulated Rover**: `KrishiNetra Autonomous Rover 01` (`KNR-ALPHA-001`)
