# AWS Deployment Guide for CRIP

This project is designed for deployment on AWS to handle full production ML traffic.

## 1. Backend API (FastAPI) -> AWS EC2 / ECS
- Push the Docker image to ECR:
  ```bash
  aws ecr create-repository --repository-name crip-api
  docker tag crip-api:latest <account_id>.dkr.ecr.<region>.amazonaws.com/crip-api
  docker push <account_id>.dkr.ecr.<region>.amazonaws.com/crip-api
  ```
- Deploy to an EC2 instance or setup an ECS cluster (Fargate recommended) to run the container.
- Map port 80 to container port 8000.

## 2. Database -> AWS RDS (PostgreSQL)
- Create an RDS instance for PostgreSQL.
- Update the `DATABASE_URL` environment variable for the FastAPI backend to point to the RDS endpoint.
  ```text
  postgresql://user:password@<rds-endpoint>:5432/crip_db
  ```

## 3. Data Storage & Model Registry -> AWS S3
- Store training raw files (`telco_churn.csv`) in an S3 bucket.
- Store trained models (`churn_model.pkl`) to S3 in the pipeline.
- Set `MODEL_PATH=s3://crip-bucket/models/churn_model.pkl` or download locally during container startup.

## 4. Frontend -> AWS S3 + CloudFront / Vercel
- For Next.js/React, export the static build (`npm run export`) and host it highly available on S3 + CloudFront, or map the domain on Vercel/Netlify pointing to the backend EC2 server.
