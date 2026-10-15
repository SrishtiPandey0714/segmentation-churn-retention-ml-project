# Customer Retention Intelligence Platform (CRIP) - Deployment Guide

This guide details deployment options for the CRIP machine learning system across local environments, Docker, GitHub Actions CI/CD, and AWS cloud infrastructure.

---

## 1. Automated CI/CD Deployment (GitHub Actions)

The repository includes a 4-stage GitHub Actions pipeline (`.github/workflows/ci.yml`):
1. **Backend CI**: Python 3.11 linting (`flake8`) & unit tests (`pytest`).
2. **Frontend CI**: React / Vite build verification.
3. **Container Verification**: Builds Docker images and verifies `/health` endpoints.
4. **Continuous Deployment (CD)**:
   - Publishes Docker images to **GitHub Container Registry (GHCR)**:
     - `ghcr.io/<username>/segmentation-churn-retention-ml-project/crip-api:latest`
     - `ghcr.io/<username>/segmentation-churn-retention-ml-project/crip-frontend:latest`
   - Automatically pushes to **Amazon ECR** if AWS repository secrets are configured.

### Setting up AWS Secrets in GitHub (Optional):
Under GitHub Repository -> **Settings** -> **Secrets and variables** -> **Actions**, add:
- `AWS_ACCESS_KEY_ID`: IAM access key with ECR permissions.
- `AWS_SECRET_ACCESS_KEY`: IAM secret access key.
- `AWS_REGION`: Target AWS region (e.g., `us-east-1`).

---

## 2. One-Command Deployment via Docker Compose

Run the entire system (FastAPI backend + React frontend) locally or on any cloud VPS (EC2, Lightsail, DigitalOcean):

```bash
# Clone the repository
git clone https://github.com/SrishtiPandey0714/segmentation-churn-retention-ml-project.git
cd segmentation-churn-retention-ml-project

# Copy and configure environment variables
cp .env.example .env

# Build and start services
docker compose up --build -d
```

- **Frontend Application**: `http://localhost:3000`
- **FastAPI API & Docs**: `http://localhost:8000/docs`
- **Health Check Endpoint**: `http://localhost:8000/health`

---

## 3. AWS Production Architecture

### A. Backend API (FastAPI) on AWS ECS / Fargate
1. **ECR Repository**:
   ```bash
   aws ecr create-repository --repository-name crip-api --region us-east-1
   ```
2. **Build and Push**:
   ```bash
   docker build -t crip-api:latest .
   docker tag crip-api:latest <account_id>.dkr.ecr.us-east-1.amazonaws.com/crip-api:latest
   docker push <account_id>.dkr.ecr.us-east-1.amazonaws.com/crip-api:latest
   ```
3. **ECS Fargate Task**:
   - Container port: `8000`
   - CPU: `0.5 vCPU`, Memory: `1 GB`
   - Health check path: `/health`

### B. Database (PostgreSQL) on AWS RDS
1. Provision an RDS PostgreSQL instance (db.t4g.micro for free-tier/testing).
2. Set `DATABASE_URL` in ECS Task Definition / environment:
   ```text
   postgresql://crip_user:<password>@<rds-endpoint>:5432/crip_db
   ```

### C. Data & Model Storage (AWS S3)
1. Store raw customer datasets in an S3 bucket (`s3://crip-ml-artifacts/data/`).
2. Store and version trained models (`s3://crip-ml-artifacts/models/churn_model.pkl`).
3. Set `MODEL_PATH=src/models/churn_model.pkl` or pull model artifact during startup.

### D. Frontend on AWS S3 + CloudFront or Vercel
1. Build static production bundle:
   ```bash
   cd frontend
   npm ci
   npm run build
   ```
2. Deploy the `frontend/dist/` folder to S3 configured for Static Website Hosting and attach an AWS CloudFront distribution for SSL and CDN caching.
