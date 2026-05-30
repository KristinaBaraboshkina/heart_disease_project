"""
Step 4: Model Training + GridSearchCV
=======================================
Сравниваем три модели внутри Pipeline с SelectKBest:
  1. LogisticRegression
  2. RandomForestClassifier
  3. GradientBoostingClassifier

Для лучшей модели делаем GridSearchCV по гиперпараметрам.
Метрики: F1, ROC-AUC, PR-AUC, Confusion Matrix.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (f1_score, roc_auc_score, classification_report,
                              confusion_matrix, precision_recall_curve, auc)
import warnings, joblib, os
warnings.filterwarnings('ignore')

# ── Данные ───────────────────────────────────────────────────────────────────
df = pd.read_csv('data/heart.csv')
X, y = df.drop('target', axis=1), df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── Preprocessor ─────────────────────────────────────────────────────────────
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

# ── Шаг 4.1: Быстрое сравнение трёх моделей ─────────────────────────────────
print("=" * 60)
print("ШАГ 4.1 — Сравнение моделей (дефолтные параметры)")
print("=" * 60)
print(f"{'Модель':<30} {'F1':>6} {'ROC-AUC':>8} {'CV F1':>8}")
print("-" * 60)

candidates = {
    'LogisticRegression':    LogisticRegression(random_state=42, max_iter=1000),
    'RandomForest':          RandomForestClassifier(random_state=42, n_estimators=100),
    'GradientBoosting':      GradientBoostingClassifier(random_state=42),
}

best_cv, best_name, best_pipe = 0, None, None
for name, clf in candidates.items():
    pipe = make_pipeline(clf)
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    f1      = f1_score(y_test, y_pred)
    roc     = roc_auc_score(y_test, y_proba)
    cv_f1   = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1').mean()
    print(f"{name:<30} {f1:>6.3f} {roc:>8.3f} {cv_f1:>8.3f}")
    if cv_f1 > best_cv:
        best_cv, best_name, best_pipe = cv_f1, name, pipe

print(f"\n✓ Лучшая базовая модель: {best_name}  (CV F1={best_cv:.3f})")

# ── Шаг 4.2: GridSearchCV для лучшей модели ──────────────────────────────────
print("\n" + "=" * 60)
print(f"ШАГ 4.2 — GridSearchCV ({best_name})")
print("=" * 60)

if best_name == 'RandomForest':
    param_grid = {
        'feature_selection__k': [8, 10, 12],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5],
    }
elif best_name == 'GradientBoosting':
    param_grid = {
        'feature_selection__k': [8, 10, 12],
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 5],
    }
else:
    param_grid = {
        'feature_selection__k': [8, 10, 12],
        'classifier__C': [0.1, 0.5, 1.0, 5.0],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear'],
    }

grid = GridSearchCV(
    make_pipeline(candidates[best_name]),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=0,
)
grid.fit(X_train, y_train)

print(f"Лучшие параметры:")
for k, v in grid.best_params_.items():
    print(f"  {k}: {v}")
print(f"Лучший CV F1: {grid.best_score_:.3f}")

# ── Шаг 4.3: Финальная оценка лучшей модели ──────────────────────────────────
print("\n" + "=" * 60)
print("ШАГ 4.3 — Финальная оценка на тестовой выборке")
print("=" * 60)

best_model = grid.best_estimator_
y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

f1      = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
prec, rec, _ = precision_recall_curve(y_test, y_proba)
pr_auc  = auc(rec, prec)
cm      = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred, target_names=['No disease', 'Disease']))
print(f"ROC-AUC  : {roc_auc:.3f}")
print(f"PR-AUC   : {pr_auc:.3f}")
print(f"F1       : {f1:.3f}")
print(f"\nConfusion Matrix:\n{cm}")
tn, fp, fn, tp = cm.ravel()
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")
print(f"\n  Recall (болезнь): {tp/(tp+fn):.3f}  ← важно в медицине!")

# ── Сохраняем модель ──────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/best_pipeline.pkl')
print(f"\n✓ Модель сохранена → models/best_pipeline.pkl")

# ── Важность признаков (если RandomForest/GradientBoosting) ──────────────────
clf_step = best_model.named_steps['classifier']
if hasattr(clf_step, 'feature_importances_'):
    sel = best_model.named_steps['feature_selection']
    prep = best_model.named_steps['preprocessor']
    ohe_names = (prep.named_transformers_['cat']
                 .named_steps['enc']
                 .get_feature_names_out(categorical_features).tolist())
    all_names = numeric_features + ohe_names
    selected_names = [n for n, m in zip(all_names, sel.get_support()) if m]
    importances = clf_step.feature_importances_
    pairs = sorted(zip(selected_names, importances), key=lambda x: -x[1])
    print("\nВажность признаков (top-10):")
    for name, imp in pairs[:10]:
        bar = '█' * int(imp * 100)
        print(f"  {name:<20} {imp:.3f}  {bar}")

