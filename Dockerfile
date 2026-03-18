FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
# Add necessary libraries since we are building from scratch
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install fastapi uvicorn pandas numpy scikit-learn lightgbm shap sqlalchemy pydantic joblib python-multipart

# Copy application source code
COPY api/ api/
COPY src/ src/
# model.pkl might be in src/models/churn_model.pkl

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
