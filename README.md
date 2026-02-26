# Customer Segmentation, Churn Prediction & Retention System

## Business Problem
Customer churn leads to direct revenue loss and increased acquisition costs.  
This project aims to **segment customers based on behavioral and monetary value**, **predict churn risk within each segment**, and **recommend retention actions** for high-risk customers.  

The system is designed as a **production-ready machine learning pipeline**, covering the entire lifecycle from data ingestion to cloud deployment and monitoring.

---

## Business Objectives
- Identify distinct customer segments based on usage and spending behavior
- Predict customer churn probability within each segment
- Prioritize high-value churn risks
- Support data-driven retention strategies
- Deploy a scalable, automated ML system

---

## Machine Learning Tasks
- **Customer Segmentation** (Unsupervised Learning – K-Means)
- **Churn Prediction** (Supervised Learning – Binary Classification)
- **Retention Recommendation** (Rule-based decision logic)

---

## Tech Stack
- **Language:** Python  
- **Data Analysis & ML:** Pandas, NumPy, Scikit-learn  
- **Experiment Tracking:** MLflow  
- **Backend API:** FastAPI  
- **Database:** PostgreSQL  
- **Containerization:** Docker  
- **CI/CD:** GitHub Actions / GitLab CI  
- **Cloud:** AWS (EC2, RDS, S3)  

---

## Project Structure

churn-segmentation-ml/
│
├── notebooks/                # Research & EDA only (no production logic)
│   └── 01_data_understanding_eda.ipynb
│
├── src/                      # Production-ready modular code
│   ├── data/
│   │   ├── ingestion.py      # Load raw data
│   │   └── preprocessing.py  # Cleaning & validation
│   │
│   ├── features/
│   │   └── build_features.py # Feature engineering
│   │
│   ├── models/
│   │   ├── segmentation.py   # Customer clustering
│   │   ├── churn_model.py    # Churn prediction
│   │   └── evaluate.py       # Model evaluation
│   │
│   ├── retention/
│   │   └── strategy.py       # Retention logic
│   │
│   ├── api/
│   │   └── app.py            # FastAPI inference service
│   │
│   ├── database/
│   │   └── db.py             # PostgreSQL connection & queries
│   │
│   └── utils/
│       └── logger.py         # Logging utilities
│
├── tests/                    # Unit & integration tests
│
├── data/                     # Temporary local data storage
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore