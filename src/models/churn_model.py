import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import joblib

class ChurnModelEnsemble:
    """
    Final Blend Model: LGBM + Logistic Regression built from notebook phase 10/11
    """
    def __init__(self, lgbm_params=None, lr_params=None, blend_weight=0.7):
        if lgbm_params is None:
            # Optimal params from notebook's optuna study
            lgbm_params = {
                'n_estimators': 300,
                'learning_rate': 0.05,
                'max_depth': -1,
                'random_state': 42
            }
        if lr_params is None:
            lr_params = {
                'max_iter': 5000,
                'random_state': 42
            }
            
        self.lgbm = LGBMClassifier(**lgbm_params)
        self.lr = LogisticRegression(**lr_params)
        self.blend_weight = blend_weight # Weight for LGBM
        
    def fit(self, X_train, y_train):
        print("Training LightGBM model...")
        self.lgbm.fit(X_train, y_train)
        
        print("Training Logistic Regression model...")
        self.lr.fit(X_train, y_train)
        
    def predict_proba(self, X):
        prob_lgbm = self.lgbm.predict_proba(X)[:, 1]
        prob_lr = self.lr.predict_proba(X)[:, 1]
        
        # Blend probabilities
        blended_prob = self.blend_weight * prob_lgbm + (1 - self.blend_weight) * prob_lr
        
        # Return in (N, 2) format for compatibility
        return np.vstack((1 - blended_prob, blended_prob)).T
        
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

def evaluate_model(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC Score:", auc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return {"roc_auc": auc}

def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    return joblib.load(filepath)
