import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/crip_db")

# Uncomment to actually connect to DB (useful later for Phase 4)
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Customer(Base):
    __tablename__ = "customers"
    
    id = Column(String, primary_key=True, index=True)
    tenure = Column(Integer)
    monthly_charges = Column(Float)
    total_charges = Column(Float)
    contract = Column(String)

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    customer_id = Column(String, index=True)
    churn_prob = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class FeatureLog(Base):
    __tablename__ = "features_log"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    prediction_id = Column(Integer)
    feature_name = Column(String)
    importance = Column(Float)

class ApiLog(Base):
    __tablename__ = "api_logs"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    endpoint = Column(String)
    latency_ms = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
