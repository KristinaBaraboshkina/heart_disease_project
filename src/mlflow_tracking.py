"""
Step 5: Experiment Tracking with MLflow
=========================================
Логируем все три модели как отдельные runs в одном эксперименте.
Для каждого run фиксируем:
  - параметры       (гиперпараметры + параметры пайплайна)
  - метрики         (F1, ROC-AUC, PR-AUC, Recall, Precision)
  - артефакты       (confusion matrix PNG, модель)
  - теги            (тип модели, датасет, автор)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                              recall_score, precision_recall_curve, auc,
                              confusion_matrix, ConfusionMatrixDisplay)
import mlflow
import mlflow.sklearn
import warnings, os
warnings.filterwarnings('ignore')

# ── Данные ───────────────────────────────────────────────────────────────────
df = pd.read_csv('data/heart.csv')
X, y = df.drop('target', axis=1), df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

numeric_features     = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc',  StandardScaler())]), numeric_features),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('enc', OneHotEncoder(handle_unknown='ignore',
                                            sparse_output=False))]),
     categorical_features),
])

def make_pipeline(clf):
    return Pipeline([
        ('preprocessor',      preprocessor),
        ('feature_selection', SelectKBest(score_func=f_classif, k=10)),
        ('classifier',        clf),
    ])

# ── MLflow: настройка эксперимента ───────────────────────────────────────────
EXPERIMENT_NAME = "heart-disease-classification"
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)

os.makedirs('artifacts', exist_ok=True)

def save_cm_plot(y_true, y_pred, model_name):
    """Сохраняем confusion matrix как PNG-артефакт."""
    fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0f1117')
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No disease', 'Disease'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_facecolor('#1a1d2e')
    ax.set_title(f'Confusion Matrix — {model_name}', color='white', fontsize=11)
    plt.tight_layout()
    path = f'artifacts/cm_{model_name.replace(" ","_")}.png'
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    return path

# ── Конфигурации экспериментов ────────────────────────────────────────────────
experiments = [
    {
        'name':   'LogisticRegression',
        'clf':    LogisticRegression(C=0.1, penalty='l2',
                                     solver='liblinear', random_state=42,
                                     max_iter=1000),
        'params': {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear',
                   'k_features': 10},
    },
    {
        'name':   'RandomForest',
        'clf':    RandomForestClassifier(n_estimators=200, max_depth=5,
                                          random_state=42),
        'params': {'n_estimators': 200, 'max_depth': 5,
                   'k_features': 10},
    },
    {
        'name':   'GradientBoosting',
        'clf':    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                              max_depth=3, random_state=42),
        'params': {'n_estimators': 100, 'learning_rate': 0.1,
                   'max_depth': 3, 'k_features': 10},
    },
]

print(f"Эксперимент: '{EXPERIMENT_NAME}'")
print(f"Tracking URI: {mlflow.get_tracking_uri()}\n")
print("=" * 60)

run_ids = {}

for exp in experiments:
    with mlflow.start_run(run_name=exp['name']) as run:

        # ── Теги ─────────────────────────────────────────────────────────────
        mlflow.set_tags({
            'model_type': exp['name'],
            'dataset':    'heart_disease_cleveland',
            'task':       'binary_classification',
            'author':     'student',
            'step':       'training',
        })

        # ── Параметры ─────────────────────────────────────────────────────────
        mlflow.log_params(exp['params'])
        mlflow.log_params({
            'imputer_strategy_num': 'median',
            'imputer_strategy_cat': 'most_frequent',
            'scaler':               'StandardScaler',
            'encoder':              'OneHotEncoder',
            'feature_selector':     'SelectKBest_f_classif',
            'test_size':            0.2,
            'random_state':         42,
        })

        # ── Обучение ──────────────────────────────────────────────────────────
        pipe = make_pipeline(exp['clf'])
        pipe.fit(X_train, y_train)

        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        # ── Метрики ───────────────────────────────────────────────────────────
        f1        = f1_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        cv_f1     = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1').mean()

        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(rec_curve, prec_curve)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        mlflow.log_metrics({
            'f1':        round(f1,        4),
            'roc_auc':   round(roc_auc,   4),
            'pr_auc':    round(pr_auc,    4),
            'precision': round(precision, 4),
            'recall':    round(recall,    4),
            'cv_f1_mean':round(cv_f1,     4),
            'true_positives':  int(tp),
            'false_positives': int(fp),
            'true_negatives':  int(tn),
            'false_negatives': int(fn),
        })

        # ── Артефакты ─────────────────────────────────────────────────────────
        cm_path = save_cm_plot(y_test, y_pred, exp['name'])
        mlflow.log_artifact(cm_path, artifact_path='plots')

        # ── Модель ────────────────────────────────────────────────────────────
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path='model',
            registered_model_name=f"heart_disease_{exp['name'].lower()}",
            input_example=X_test.iloc[:3],
        )

        run_ids[exp['name']] = run.info.run_id

        print(f"[{exp['name']}]  run_id={run.info.run_id[:8]}...")
        print(f"  F1={f1:.3f}  ROC-AUC={roc_auc:.3f}  "
              f"PR-AUC={pr_auc:.3f}  CV-F1={cv_f1:.3f}")
        print(f"  Recall={recall:.3f}  Precision={precision:.3f}")
        print()

print("=" * 60)
print("✓ Все runs залогированы в MLflow!")
print("\nЧтобы открыть UI — выполни в терминале:")
print("  cd heart_disease_project && mlflow ui")
print("  Затем открой http://localhost:5000")

# ── Сравнительная таблица из MLflow ──────────────────────────────────────────
print("\n" + "=" * 60)
print("СРАВНЕНИЕ RUNS (из MLflow)")
print("=" * 60)

client = mlflow.tracking.MlflowClient()
exp_obj = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(exp_obj.experiment_id,
                           order_by=["metrics.f1 DESC"])

print(f"{'Модель':<22} {'F1':>6} {'ROC-AUC':>8} {'PR-AUC':>7} {'Recall':>7} {'CV-F1':>7}")
print("-" * 60)
for r in runs:
    m = r.data.metrics
    print(f"{r.data.tags.get('model_type','?'):<22} "
          f"{m.get('f1',0):>6.3f} "
          f"{m.get('roc_auc',0):>8.3f} "
          f"{m.get('pr_auc',0):>7.3f} "
          f"{m.get('recall',0):>7.3f} "
          f"{m.get('cv_f1_mean',0):>7.3f}")

# ── Загружаем лучшую модель из MLflow ─────────────────────────────────────────
best_run = runs[0]
best_run_id = best_run.info.run_id
best_model_uri = f"runs:/{best_run_id}/model"
loaded_model = mlflow.sklearn.load_model(best_model_uri)
y_pred_loaded = loaded_model.predict(X_test)
print(f"\n✓ Лучшая модель загружена из MLflow: "
      f"{best_run.data.tags.get('model_type')}")
print(f"  F1 после загрузки = {f1_score(y_test, y_pred_loaded):.3f}  ✓")

