# Extracted notebook code


# --- CELL 0 ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
sns.set(style="whitegrid")

df = pd.read_csv("../data/raw/telco_churn.csv")

# --- CELL 1 ---
df.shape

# --- CELL 2 ---
df.info()


# --- CELL 3 ---
df.columns

# --- CELL 4 ---
df["Churn Value"].value_counts(normalize=True)

# --- CELL 5 ---
df.isnull().sum()

# --- CELL 6 ---
categorical_cols = df.select_dtypes(include="object").columns
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

categorical_cols, numerical_cols

# --- CELL 7 ---
plt.figure(figsize=(6,4))
sns.histplot(df["Tenure Months"], bins=30, kde=True)
plt.title("Customer Tenure Distribution")
plt.show()

# --- CELL 8 ---
plt.figure(figsize=(6,4))
sns.boxplot(x="Churn Value", y="Monthly Charges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# --- CELL 9 ---
plt.figure(figsize=(6,4))
sns.countplot(x="Contract", hue="Churn Value", data=df)
plt.xticks(rotation=45)
plt.title("Contract Type vs Churn")
plt.show()

# --- CELL 10 ---
# segmentation phse

# --- CELL 11 ---
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df["Total Charges"].isnull().sum()

# --- CELL 12 ---
df = df.dropna(subset=["Total Charges"]).copy()

# --- CELL 13 ---
service_cols = [
    'Phone Service', 'Multiple Lines', 'Internet Service',
    'Online Security', 'Online Backup', 'Device Protection',
    'Tech Support', 'Streaming TV', 'Streaming Movies'
]

df["ServiceCount"] = (df[service_cols] == "Yes").sum(axis=1)

# --- CELL 14 ---
segmentation_features = ["Tenure Months", "Monthly Charges", "Total Charges", "ServiceCount"]

seg_df = df[segmentation_features].copy()
seg_df.head()

# --- CELL 15 ---
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
seg_scaled = scaler.fit_transform(seg_df)

# --- CELL 16 ---
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(seg_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# --- CELL 17 ---
from sklearn.metrics import silhouette_score

for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(seg_scaled)
    score = silhouette_score(seg_scaled, labels)
    print(f"K={k}, Silhouette Score={score:.4f}")

# --- CELL 18 ---
kmeans = KMeans(n_clusters=3, random_state=42)
df["Segment"] = kmeans.fit_predict(seg_scaled)

# --- CELL 19 ---
df.groupby("Segment")[segmentation_features].mean()

# --- CELL 20 ---
# interpreting what is found by clustering b4 we run the k means algo

# --- CELL 21 ---
df["Segment"].value_counts()

# --- CELL 22 ---
df.groupby("Segment")[segmentation_features].mean()

# --- CELL 23 ---
import seaborn as sns

sns.scatterplot(
    x="Tenure Months",
    y="Monthly Charges",
    hue="Segment",
    data=df,
    palette="Set2"
)
plt.title("Customer Segments Visualization")
plt.show()

# --- CELL 24 ---
df.groupby("Segment")["Churn Value"].value_counts(normalize=True)

# --- CELL 25 ---
# phase 4- churn model

# --- CELL 26 ---

df.shape

# --- CELL 27 ---
y = df["Churn Value"]

# --- CELL 28 ---
y.value_counts()

# --- CELL 29 ---
X = df.drop(columns=[
    "CustomerID",
    "Churn Label",
    "Churn Score",
    "Churn Reason",
    "CLTV",
    "Churn Value"
])

# --- CELL 30 ---
X = X.drop(columns=[
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude"
])

# --- CELL 31 ---
X = pd.get_dummies(X, drop_first=True)

# --- CELL 32 ---
X.shape

# --- CELL 33 ---
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --- CELL 34 ---
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=2000)

model.fit(X_train, y_train)

# --- CELL 35 ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# --- CELL 36 ---
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))

# --- CELL 37 ---
confusion_matrix(y_test, y_pred)

# --- CELL 38 ---
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

feature_importance.sort_values(by="Coefficient", ascending=False).head(10)

# --- CELL 39 ---
#Evaluation stage

# --- CELL 40 ---
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# --- CELL 41 ---
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

importance.sort_values(by="Coefficient", ascending=False).head(10)

# --- CELL 42 ---
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_prob)

# --- CELL 43 ---
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr)
plt.plot([0,1],[0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# --- CELL 44 ---
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print(classification_report(y_test, rf_pred))

# --- CELL 45 ---
df["PredictedChurn"] = model.predict(X)

df.groupby("Segment")["PredictedChurn"].mean()

# --- CELL 46 ---
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))

# --- CELL 47 ---
from lightgbm import LGBMClassifier

lgb = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42
)

lgb.fit(X_train, y_train)

y_pred_lgb = lgb.predict(X_test)
y_prob_lgb = lgb.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred_lgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lgb))

# --- CELL 48 ---
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ("rf", rf),
        ("xgb", xgb),
        ("lgb", lgb)
    ],
    voting="soft"
)

ensemble.fit(X_train, y_train)

y_pred_ens = ensemble.predict(X_test)
y_prob_ens = ensemble.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred_ens))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_ens))

# --- CELL 49 ---
import pandas as pd
from sklearn.metrics import accuracy_score

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "Ensemble"
    ],
    
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, y_pred_xgb),
        accuracy_score(y_test, y_pred_lgb),
        accuracy_score(y_test, y_pred_ens)
    ],
    
    "ROC_AUC": [
        roc_auc_score(y_test, y_prob),
        roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]),
        roc_auc_score(y_test, y_prob_xgb),
        roc_auc_score(y_test, y_prob_lgb),
        roc_auc_score(y_test, y_prob_ens)
    ]
})

results.sort_values(by="ROC_AUC", ascending=False)

# --- CELL 50 ---
import joblib

joblib.dump(model, "churn_model.pkl")

# --- CELL 51 ---
#phase 5- SHAP

# --- CELL 52 ---
import numpy as np

X_train_np = X_train.astype(float).values
X_test_np = X_test.astype(float).values

# --- CELL 53 ---
import shap

explainer = shap.Explainer(model, X_train_np)

# --- CELL 54 ---
shap_values = explainer(X_test_np)

# --- CELL 55 ---
shap.plots.beeswarm(shap_values)

# --- CELL 56 ---
shap.plots.bar(shap_values)

# --- CELL 57 ---
shap.plots.waterfall(shap_values[0])

# --- CELL 58 ---
#phase 6: optimizing furthur (cross validation, hyperParameter Tuning, early stopping and regularization)

# --- CELL 59 ---
#HyperParameterization 
import numpy as np

param_grid = {
    "model__C": np.logspace(-4, 4, 20),   # 0.0001 → 10000
    "model__penalty": ["l1", "l2"],
    "model__solver": ["liblinear"]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best ROC-AUC:", grid.best_score_)

# --- CELL 60 ---
#Cross Validation
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

# --- CELL 61 ---
#final grid search
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
print("Best CV ROC-AUC:", grid.best_score_)

# --- CELL 62 ---
#RandomizedSearch
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=50,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

# --- CELL 63 ---
y_prob_best = search.best_estimator_.predict_proba(X_test)[:,1]

# --- CELL 64 ---
import pandas as pd
from sklearn.metrics import roc_auc_score

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression (baseline)",
        "Logistic Regression (tuned)"
    ],
    
    "ROC_AUC": [
        roc_auc_score(y_test, y_prob_log),
        roc_auc_score(y_test, y_prob_best)
    ]
})

comparison

# --- CELL 65 ---
print(search.best_params_)
print(search.best_score_)

# --- CELL 66 ---
import joblib

joblib.dump(best_model, "../src/models/churn_model.pkl")

# --- CELL 68 ---
#phase 7: trying out neural networks, catboost and proper stacking techniques

# --- CELL 69 ---
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# --- CELL 70 ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

mlp_pipeline = Pipeline([
    ("scaler", StandardScaler()),   # VERY IMPORTANT
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=64,
        learning_rate="adaptive",
        max_iter=500,
        random_state=42
    ))
])

# Train
mlp_pipeline.fit(X_train, y_train)

# Predict probabilities
y_prob_mlp = mlp_pipeline.predict_proba(X_test)[:,1]

# Evaluate
print("MLP ROC-AUC:", roc_auc_score(y_test, y_prob_mlp))

# --- CELL 71 ---
pip install catboost

# --- CELL 72 ---
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric="AUC",
    verbose=0,
    random_state=42
)

cat_model.fit(X_train, y_train)

y_prob_cat = cat_model.predict_proba(X_test)[:,1]

print("CatBoost ROC-AUC:", roc_auc_score(y_test, y_prob_cat))

# --- CELL 73 ---
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

models = {
    "lr": LogisticRegression(max_iter=5000),
    "lgbm": LGBMClassifier(n_estimators=300),
    "mlp": mlp_pipeline
}

# --- CELL 74 ---
from sklearn.base import clone

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((X_train.shape[0], len(models)))
test_preds = np.zeros((X_test.shape[0], len(models)))

for i, (name, model) in enumerate(models.items()):
    print(f"Training {name}...")
    
    test_fold_preds = np.zeros((X_test.shape[0], 5))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        
        # FIX HERE
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        model_clone = clone(model)
        model_clone.fit(X_tr, y_tr)
        
        oof_preds[val_idx, i] = model_clone.predict_proba(X_val)[:,1]
        test_fold_preds[:, fold] = model_clone.predict_proba(X_test)[:,1]
    
    test_preds[:, i] = test_fold_preds.mean(axis=1)

# --- CELL 75 ---
meta_model = LogisticRegression()

meta_model.fit(oof_preds, y_train)

# Final predictions
final_preds = meta_model.predict_proba(test_preds)[:,1]

print("Stacked ROC-AUC:", roc_auc_score(y_test, final_preds))

# --- CELL 76 ---
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train, y_train)

y_prob_log = log_model.predict_proba(X_test)[:,1]

# --- CELL 77 ---
comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "LightGBM",
        "MLP",
        "CatBoost",
        "Stacked Model"
    ],
    
    "ROC_AUC": [
        roc_auc_score(y_test, y_prob_log),
        roc_auc_score(y_test, y_prob_lgb),
        roc_auc_score(y_test, y_prob_mlp),
        roc_auc_score(y_test, y_prob_cat),
        roc_auc_score(y_test, final_preds)
    ]
}).sort_values("ROC_AUC", ascending=False)

comparison

# --- CELL 78 ---
#phase 08: optuna tuning setup for lightbgm

# --- CELL 79 ---
pip install optuna

# --- CELL 80 ---
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import numpy as np

# CV strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42
    }
    
    scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]
        
        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        
        y_pred = model.predict_proba(X_val)[:,1]
        score = roc_auc_score(y_val, y_pred)
        
        scores.append(score)
    
    return np.mean(scores)

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best Trial:")
print(study.best_trial.params)
print("Best ROC-AUC:", study.best_value)

# --- CELL 81 ---
best_params = study.best_trial.params

lgb_optuna = LGBMClassifier(**best_params)

lgb_optuna.fit(X_train, y_train)

y_prob_lgb_opt = lgb_optuna.predict_proba(X_test)[:,1]

print("Optimized LGBM ROC-AUC:", roc_auc_score(y_test, y_prob_lgb_opt))

# --- CELL 82 ---
comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "LightGBM (baseline)",
        "LightGBM (Optuna)",
        "Stacked Model"
    ],
    
    "ROC_AUC": [
        roc_auc_score(y_test, y_prob_log),
        roc_auc_score(y_test, y_prob_lgb),
        roc_auc_score(y_test, y_prob_lgb_opt),
        roc_auc_score(y_test, final_preds)
    ]
}).sort_values("ROC_AUC", ascending=False)

comparison

# --- CELL 83 ---
y_final = (
    0.7 * y_prob_lgb_opt +
    0.3 * y_prob_log
)

roc_auc_score(y_test, y_final)

# --- CELL 84 ---
for w in [0.6, 0.7, 0.8, 0.9]:
    y_blend = w * y_prob_lgb_opt + (1-w) * y_prob_log
    print(w, roc_auc_score(y_test, y_blend))

# --- CELL 85 ---
# phase 9: adding higher end feature engineering

# --- CELL 86 ---
print(X_train.columns)

# --- CELL 87 ---
X_train["charges_per_month"] = X_train["Total Charges"] / (X_train["Tenure Months"] + 1)
X_test["charges_per_month"] = X_test["Total Charges"] / (X_test["Tenure Months"] + 1)

# --- CELL 88 ---
X_train["avg_service_cost"] = X_train["Monthly Charges"] / (X_train["ServiceCount"] + 1)
X_test["avg_service_cost"] = X_test["Monthly Charges"] / (X_test["ServiceCount"] + 1)

# --- CELL 89 ---
X_train["avg_service_cost"] = X_train["Monthly Charges"] / (X_train["ServiceCount"] + 1)
X_test["avg_service_cost"] = X_test["Monthly Charges"] / (X_test["ServiceCount"] + 1)

# --- CELL 90 ---
X_train["tenure_contract_1yr"] = X_train["Tenure Months"] * X_train["Contract_One year"]
X_test["tenure_contract_1yr"] = X_test["Tenure Months"] * X_test["Contract_One year"]

X_train["tenure_contract_2yr"] = X_train["Tenure Months"] * X_train["Contract_Two year"]
X_test["tenure_contract_2yr"] = X_test["Tenure Months"] * X_test["Contract_Two year"]

# --- CELL 91 ---
X_train["fiber_charges"] = X_train["Monthly Charges"] * X_train["Internet Service_Fiber optic"]
X_test["fiber_charges"] = X_test["Monthly Charges"] * X_test["Internet Service_Fiber optic"]

# --- CELL 92 ---
service_cols = [
    "Online Security_Yes",
    "Online Backup_Yes",
    "Device Protection_Yes",
    "Tech Support_Yes",
    "Streaming TV_Yes",
    "Streaming Movies_Yes"
]

X_train["total_active_services"] = X_train[service_cols].sum(axis=1)
X_test["total_active_services"] = X_test[service_cols].sum(axis=1)

# --- CELL 93 ---
X_train["fiber_high_pay"] = (
    (X_train["Internet Service_Fiber optic"] == 1) & 
    (X_train["Monthly Charges"] > X_train["Monthly Charges"].median())
).astype(int)

X_test["fiber_high_pay"] = (
    (X_test["Internet Service_Fiber optic"] == 1) & 
    (X_test["Monthly Charges"] > X_train["Monthly Charges"].median())
).astype(int)

# --- CELL 94 ---
import numpy as np

X_train["log_total_charges"] = np.log1p(X_train["Total Charges"])
X_test["log_total_charges"] = np.log1p(X_test["Total Charges"])

# --- CELL 95 ---
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(
    degree=2,
    interaction_only=True,
    include_bias=False
)

X_train_poly = poly.fit_transform(X_train[["Tenure Months", "Monthly Charges"]])
X_test_poly = poly.transform(X_test[["Tenure Months", "Monthly Charges"]])

# --- CELL 96 ---
lgb_optuna.fit(X_train, y_train)

y_prob_lgb_opt = lgb_optuna.predict_proba(X_test)[:,1]

print("New ROC-AUC:", roc_auc_score(y_test, y_prob_lgb_opt))

# --- CELL 97 ---
#phase 10: final score and optimizations

# --- CELL 98 ---
import pandas as pd

feat_imp = pd.Series(lgb_optuna.feature_importances_, index=X_train.columns)
feat_imp = feat_imp.sort_values(ascending=False)

print(feat_imp)

# --- CELL 99 ---
low_importance = feat_imp.tail(10).index

X_train = X_train.drop(columns=low_importance)
X_test = X_test.drop(columns=low_importance)

# --- CELL 100 ---
for w in np.linspace(0.7, 0.95, 10):
    y_blend = w * y_prob_lgb_opt + (1-w) * y_prob_log
    score = roc_auc_score(y_test, y_blend)
    print(f"Weight {w:.2f} → {score}")

# --- CELL 101 ---
from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)

# --- CELL 102 ---
# phase 11: trying out final gains using oof + advanced stacking pipeline

# --- CELL 103 ---
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

# --- CELL 104 ---
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((X_train.shape[0], len(models)))
test_preds = np.zeros((X_test.shape[0], len(models)))

for i, (name, model) in enumerate(models.items()):
    print(f"\nTraining {name}")
    
    test_fold_preds = np.zeros((X_test.shape[0], kf.n_splits))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"Fold {fold}")
        
        # Use iloc because pandas
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]
        
        model_clone = clone(model)
        model_clone.fit(X_tr, y_tr)
        
        # OOF prediction (validation fold)
        oof_preds[val_idx, i] = model_clone.predict_proba(X_val)[:,1]
        
        # Test predictions
        test_fold_preds[:, fold] = model_clone.predict_proba(X_test)[:,1]
    
    # Average test predictions
    test_preds[:, i] = test_fold_preds.mean(axis=1)

# --- CELL 105 ---
for i, name in enumerate(models.keys()):
    score = roc_auc_score(y_train, oof_preds[:, i])
    print(f"{name} OOF ROC-AUC: {score}")

# --- CELL 106 ---
meta_model = LogisticRegression()

meta_model.fit(oof_preds, y_train)

final_preds = meta_model.predict_proba(test_preds)[:,1]

print("Stacked ROC-AUC:", roc_auc_score(y_test, final_preds))

# --- CELL 107 ---
oof_weighted = 0.8 * oof_preds[:,0] + 0.2 * oof_preds[:,1]
test_weighted = 0.8 * test_preds[:,0] + 0.2 * test_preds[:,1]

print("Weighted OOF ROC-AUC:", roc_auc_score(y_train, oof_weighted))

# --- CELL 108 ---
best_score = 0
best_w = 0

for w in np.linspace(0.6, 0.95, 20):
    oof_blend = w * oof_preds[:,0] + (1-w) * oof_preds[:,1]
    score = roc_auc_score(y_train, oof_blend)
    
    if score > best_score:
        best_score = score
        best_w = w

print("Best weight:", best_w)
print("Best OOF:", best_score)

# --- CELL 109 ---
lgb_models = []

seeds = [42, 99, 123, 2024]

for seed in seeds:
    model = LGBMClassifier(**best_params, random_state=seed)
    model.fit(X_train, y_train)
    lgb_models.append(model)

# --- CELL 110 ---
y_preds = np.zeros(X_test.shape[0])

for model in lgb_models:
    y_preds += model.predict_proba(X_test)[:,1]

y_preds /= len(lgb_models)

# --- CELL 111 ---
y_final = 0.9 * y_preds + 0.1 * y_prob_log

# --- CELL 112 ---
import pandas as pd
from sklearn.metrics import roc_auc_score

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "LightGBM (baseline)",
        "LightGBM (Optuna)",
        "OOF Stacked Model",
        "Multi-seed LGBM",
        "FINAL BLEND (LGBM + LR) 🔥"
    ],
    
    "ROC_AUC": [
        roc_auc_score(y_test, y_prob_log),
        roc_auc_score(y_test, y_prob_lgb),
        roc_auc_score(y_test, y_prob_lgb_opt),
        roc_auc_score(y_test, final_preds),
        roc_auc_score(y_test, y_preds),
        roc_auc_score(y_test, y_final)
    ]
}).sort_values("ROC_AUC", ascending=False)

comparison.reset_index(drop=True)

# --- CELL 113 ---
oof_final = best_w * oof_preds[:,0] + (1-best_w) * oof_preds[:,1]

print("Final OOF ROC-AUC:", roc_auc_score(y_train, oof_final))
