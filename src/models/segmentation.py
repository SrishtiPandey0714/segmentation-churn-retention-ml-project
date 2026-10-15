import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

class CustomerSegmentation:
    """
    K-Means clustering for customer segmentation based on tenure, charges, and service count.
    """
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.features = ["Tenure Months", "Monthly Charges", "Total Charges", "ServiceCount"]
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate ServiceCount if not present
        if "ServiceCount" not in df.columns:
            service_cols = [
                'Phone Service', 'Multiple Lines', 'Internet Service',
                'Online Security', 'Online Backup', 'Device Protection',
                'Tech Support', 'Streaming TV', 'Streaming Movies'
            ]
            existing = [c for c in service_cols if c in df.columns]
            if existing:
                df["ServiceCount"] = (df[existing] == "Yes").sum(axis=1) + (df[existing] == True).sum(axis=1)
            else:
                df["ServiceCount"] = 0
                
        # Ensure required columns exist
        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                
        return df[self.features]

    def fit(self, df: pd.DataFrame):
        X = self._prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Segmentation model is not fitted yet.")
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CustomerSegmentation":
        return joblib.load(filepath)
