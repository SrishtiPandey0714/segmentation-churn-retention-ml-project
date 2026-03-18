import os
import sys
import argparse
from sklearn.model_selection import train_test_split

# Add src to path if needed (when running from root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.ingestion import load_data
from src.data.preprocessing import clean_data
from src.features.build_features import build_features
from src.models.churn_model import ChurnModelEnsemble, evaluate_model, save_model

def run_training_pipeline(data_path: str, model_save_path: str):
    print("1. Ingesting Data...")
    df = load_data(data_path)
    
    print("2. Cleaning Data...")
    df_clean = clean_data(df)
    
    print("3. Building Features...")
    X, y = build_features(df_clean, is_train=True)
    
    print("4. Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("5. Training Model...")
    model = ChurnModelEnsemble(blend_weight=0.8) # 80% LGBM, 20% LR
    model.fit(X_train, y_train)
    
    print("6. Evaluating Model...")
    evaluate_model(model, X_test, y_test)
    
    print("7. Saving Model...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    save_model(model, model_save_path)
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/raw/telco_churn.csv")
    parser.add_argument("--model_path", type=str, default="src/models/churn_model.pkl")
    args = parser.parse_args()
    
    run_training_pipeline(args.data_path, args.model_path)
