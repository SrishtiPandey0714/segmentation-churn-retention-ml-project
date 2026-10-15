# Customer Segmentation, Churn Prediction & Retention System (CRIP)

[![CI/CD Pipeline](https://github.com/SrishtiPandey0714/segmentation-churn-retention-ml-project/actions/workflows/ci.yml/badge.svg)](https://github.com/SrishtiPandey0714/segmentation-churn-retention-ml-project/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-enabled-2496ED.svg)](https://www.docker.com/)

An enterprise-grade, production-ready machine learning platform covering customer behavioral segmentation, churn risk prediction, SHAP explainability, and proactive retention strategy recommendations.

---

## 📌 Business Problem & Objectives
Customer churn directly erodes recurring revenue and escalates customer acquisition costs.
This project delivers an end-to-end ML solution that:
- **Segments customers** based on tenure, service count, and spending behavior.
- **Predicts churn probability** using a calibrated LightGBM + Logistic Regression ensemble.
- **Explains predictions** in real-time using TreeSHAP values.
- **Generates automated retention actions** tailored to the customer's risk profile.
- **Provides automated CI/CD & cloud-native containerized deployment**.

---

## 🛠️ Tech Stack
- **Language & ML:** Python 3.11, Pandas, NumPy, Scikit-learn, LightGBM, SHAP, Joblib
- **Experiment Tracking:** MLflow
- **Backend API:** FastAPI, Uvicorn, Pydantic v2
- **Frontend Dashboard:** React 19, Vite, Tailwind CSS, Lucide Icons, Recharts
- **Containerization & Orchestration:** Docker, Docker Compose
- **CI/CD Pipeline:** GitHub Actions (Automated linting, testing, Docker builds, GHCR & AWS ECR deploy)
- **Cloud Infrastructure:** AWS (ECR, ECS Fargate, S3, RDS PostgreSQL)

---

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)
Launch the complete stack (Backend API + Frontend Dashboard) with a single command:

```bash
# Clone the repository
git clone https://github.com/SrishtiPandey0714/segmentation-churn-retention-ml-project.git
cd segmentation-churn-retention-ml-project

# Run services
docker compose up --build -d
```

- **Frontend Dashboard:** [http://localhost:3000](http://localhost:3000)
- **FastAPI Interactive Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

### Option 2: Local Development

#### 1. Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## 🧪 Testing & Code Quality

Run the automated test suite with pytest:
```bash
pytest tests/ -v
```

Run code quality / syntax checks:
```bash
flake8 api/ src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
```

---

## 🔄 CI/CD Pipeline Architecture

The automated GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every `push` and `pull_request` to `main`:

```mermaid
flowchart LR
    A[Push / PR to main] --> B[Backend CI: Tests & Lint]
    A --> C[Frontend CI: Vite Build]
    B & C --> D[Docker Verify: Build & Healthcheck]
    D --> E{Merge to main?}
    E -- Yes --> F[Deploy to GitHub Container Registry]
    E -- Yes --> G[Deploy to AWS ECR (Optional)]
```

1. **`backend-ci`**: Sets up Python 3.11, validates code syntax with `flake8`, runs 8 unit and integration tests with `pytest`.
2. **`frontend-ci`**: Installs dependencies and verifies React/Vite builds without compilation errors.
3. **`docker-verify`**: Builds the production Docker container and executes dynamic healthcheck verification against `/health`.
4. **`deploy`**: Automatically pushes versioned and latest images to **GitHub Container Registry (`ghcr.io`)** and **Amazon ECR** (if AWS secrets are configured).

See [aws_deployment_guide.md](file:///c:/ds%20projects/churn-segmentation-ml/aws_deployment_guide.md) for full cloud configuration steps.

---

## 📂 Project Structure

```text
churn-segmentation-ml/
├── .github/workflows/
│   └── ci.yml               # Production CI/CD workflow
├── api/
│   ├── main.py              # FastAPI application & endpoints
│   ├── schemas.py           # Pydantic request/response schemas
│   └── database.py          # PostgreSQL / SQLAlchemy models
├── frontend/                # React + Vite interactive intelligence dashboard
│   ├── src/                 # UI components and API integration
│   ├── Dockerfile           # Multi-stage production Nginx container
│   └── nginx.conf           # Reverse proxy configuration
├── src/                     # Core ML pipeline modules
│   ├── data/                # Ingestion & data cleaning
│   ├── features/            # Feature engineering (interaction terms, scaling)
│   ├── models/              # Ensemble model, K-Means segmentation & evaluation
│   ├── retention/           # Proactive retention recommendation engine
│   └── utils/               # Centralized logging utilities
├── tests/
│   └── test_api.py          # Unit & integration test suite
├── Dockerfile               # Backend production Dockerfile
├── docker-compose.yml       # Multi-container orchestration
├── requirements.txt         # Production dependencies
└── README.md
```