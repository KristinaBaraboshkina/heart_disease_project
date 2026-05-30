"""
train_docker.py — entrypoint для Docker-контейнера.
Запускает полный ML-пайплайн: загрузка → обучение → метрики → сохранение.
"""

import os, sys, json, joblib, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                              recall_score, precision_recall_curve, auc)
import mlflow, mlflow.sklearn
warnings.filterwarnings('ignore')

# ── Пути (внутри контейнера) ──────────────────────────────────────────────────
DATA_PATH   = os.getenv('DATA_PATH',  '/app/data/heart.csv')
MODEL_DIR   = os.getenv('MODEL_DIR',  '/app/models')
MLFLOW_URI  = os.getenv('MLFLOW_URI', '/app/mlruns')
EXPERIMENT  = os.getenv('EXPERIMENT', 'heart-disease-docker')

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 55)
print("  Heart Disease ML Pipeline — Docker run")
print("=" * 55)
print(f"  Data     : {DATA_PATH}")
print(f"  Models   : {MODEL_DIR}")
print(f"  MLflow   : {MLFLOW_URI}")
print(f"  Experiment: {EXPERIMENT}")
print()

# ── 1. Загрузка данных ────────────────────────────────────────────────────────
print("[1/5] Загрузка данных...")
df = pd.read_csv(DATA_PATH)
X, y = df.drop('target', axis=1), df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"      Train={X_train.shape}, Test={X_test.shape}")

# ── 2. Построение пайплайна ───────────────────────────────────────────────────
print("[2/5] Построение Pipeline...")
numeric_features     = ['age','trestbps','chol','thalach','oldpeak','ca','thal']
categorical_features = ['sex','cp','fbs','restecg','exang','slope']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc',  StandardScaler())]), numeric_features),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('enc', OneHotEncoder(handle_unknown='ignore',
                                            sparse_output=False))]),
     categorical_features),
])

pipeline = Pipeline([
    ('preprocessor',      preprocessor),
    ('feature_selection', SelectKBest(score_func=f_classif, k=12)),
    ('classifier',        LogisticRegression(C=0.1, penalty='l2',
                                             solver='liblinear',
                                             random_state=42, max_iter=1000)),
])

# ── 3. Обучение + метрики ─────────────────────────────────────────────────────
print("[3/5] Обучение модели...")
pipeline.fit(X_train, y_train)
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

f1      = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
prec    = precision_score(y_test, y_pred)
rec     = recall_score(y_test, y_pred)
p_c, r_c, _ = precision_recall_curve(y_test, y_proba)
pr_auc  = auc(r_c, p_c)
cv_f1   = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1').mean()

metrics = dict(f1=round(f1,4), roc_auc=round(roc_auc,4),
               pr_auc=round(pr_auc,4), precision=round(prec,4),
               recall=round(rec,4), cv_f1=round(cv_f1,4))

print(f"      F1={f1:.3f}  ROC-AUC={roc_auc:.3f}  "
      f"PR-AUC={pr_auc:.3f}  Recall={rec:.3f}")

# ── 4. MLflow логирование ─────────────────────────────────────────────────────
print("[4/5] Логирование в MLflow...")
mlflow.set_tracking_uri(f"file:{MLFLOW_URI}")
mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="docker-run"):
    mlflow.set_tags({'environment': 'docker', 'dataset': 'heart_disease'})
    mlflow.log_params({
        'C': 0.1, 'penalty': 'l2', 'k_features': 12,
        'scaler': 'StandardScaler', 'encoder': 'OneHotEncoder',
    })
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(pipeline, name='model')
    print(f"      Run залогирован: experiment='{EXPERIMENT}'")

# ── 5. Сохранение модели ──────────────────────────────────────────────────────
print("[5/5] Сохранение модели...")
model_path = os.path.join(MODEL_DIR, 'pipeline.pkl')
joblib.dump(pipeline, model_path)

# Сохраняем метрики в JSON для CI/CD
metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"      Модель   → {model_path}")
print(f"      Метрики  → {metrics_path}")
print()
print("=" * 55)
print("  ГОТОВО!")
print(f"  F1={f1:.3f}  ROC-AUC={roc_auc:.3f}  Recall={rec:.3f}")
print("=" * 55)
