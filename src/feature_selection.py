"""
Step 3: Feature Selection
==========================
Сравниваем три метода отбора признаков внутри Pipeline:
  1. Filter   — SelectKBest (f_classif)
  2. Wrapper  — RFE (с LogisticRegression)
  3. Embedded — L1-регуляризация (Lasso / LogisticRegression penalty='l1')

Для каждого метода смотрим: какие признаки выбраны + F1-score на тесте.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ── Данные ──────────────────────────────────────────────────────────────────
df = pd.read_csv('data/heart.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Preprocessor (из шага 2) ─────────────────────────────────────────────────
numeric_features     = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer,  numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

# ── Базовая линия (без отбора) ───────────────────────────────────────────────
base_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   LogisticRegression(random_state=42, max_iter=1000)),
])
base_pipeline.fit(X_train, y_train)
base_f1 = f1_score(y_test, base_pipeline.predict(X_test))
base_cv = cross_val_score(base_pipeline, X_train, y_train, cv=5, scoring='f1').mean()

print("=" * 55)
print(f"{'Метод':<25} {'F1 test':>8} {'F1 CV-5':>8} {'Признаков':>10}")
print("=" * 55)
print(f"{'Baseline (все 23)':<25} {base_f1:>8.3f} {base_cv:>8.3f} {'23':>10}")

results = {}

# ══════════════════════════════════════════════════════════════
# МЕТОД 1: Filter — SelectKBest (f_classif)
# ══════════════════════════════════════════════════════════════
K = 10
filter_pipeline = Pipeline([
    ('preprocessor',      preprocessor),
    ('feature_selection', SelectKBest(score_func=f_classif, k=K)),
    ('classifier',        LogisticRegression(random_state=42, max_iter=1000)),
])
filter_pipeline.fit(X_train, y_train)
f1_filter  = f1_score(y_test, filter_pipeline.predict(X_test))
cv_filter  = cross_val_score(filter_pipeline, X_train, y_train, cv=5, scoring='f1').mean()
print(f"{'Filter: SelectKBest(k=10)':<25} {f1_filter:>8.3f} {cv_filter:>8.3f} {K:>10}")
results['Filter'] = dict(f1=f1_filter, cv=cv_filter, k=K,
                          pipeline=filter_pipeline)

# ══════════════════════════════════════════════════════════════
# МЕТОД 2: Wrapper — RFE
# ══════════════════════════════════════════════════════════════
K_rfe = 10
rfe_estimator = LogisticRegression(random_state=42, max_iter=1000)
rfe_pipeline = Pipeline([
    ('preprocessor',      preprocessor),
    ('feature_selection', RFE(estimator=rfe_estimator, n_features_to_select=K_rfe)),
    ('classifier',        LogisticRegression(random_state=42, max_iter=1000)),
])
rfe_pipeline.fit(X_train, y_train)
f1_rfe = f1_score(y_test, rfe_pipeline.predict(X_test))
cv_rfe = cross_val_score(rfe_pipeline, X_train, y_train, cv=5, scoring='f1').mean()
print(f"{'Wrapper: RFE(k=10)':<25} {f1_rfe:>8.3f} {cv_rfe:>8.3f} {K_rfe:>10}")
results['Wrapper'] = dict(f1=f1_rfe, cv=cv_rfe, k=K_rfe,
                           pipeline=rfe_pipeline)

# ══════════════════════════════════════════════════════════════
# МЕТОД 3: Embedded — L1 (Lasso logistic regression)
# ══════════════════════════════════════════════════════════════
l1_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   LogisticRegression(penalty='l1', solver='liblinear',
                                        C=0.5, random_state=42, max_iter=1000)),
])
l1_pipeline.fit(X_train, y_train)
f1_l1 = f1_score(y_test, l1_pipeline.predict(X_test))
cv_l1 = cross_val_score(l1_pipeline, X_train, y_train, cv=5, scoring='f1').mean()

# Считаем ненулевые коэффициенты
coef = l1_pipeline.named_steps['classifier'].coef_[0]
k_l1 = int((coef != 0).sum())
print(f"{'Embedded: L1 (C=0.5)':<25} {f1_l1:>8.3f} {cv_l1:>8.3f} {k_l1:>10}")
results['Embedded'] = dict(f1=f1_l1, cv=cv_l1, k=k_l1,
                            pipeline=l1_pipeline)

print("=" * 55)

# ── Выбираем лучший метод по CV F1 ──────────────────────────────────────────
best_name = max(results, key=lambda m: results[m]['cv'])
best = results[best_name]
print(f"\n✓ Лучший метод: {best_name}  (CV F1 = {best['cv']:.3f})")

# ── Детальный отчёт для лучшего метода ──────────────────────────────────────
print(f"\nClassification report ({best_name}):")
y_pred_best = best['pipeline'].predict(X_test)
print(classification_report(y_test, y_pred_best,
      target_names=['No disease', 'Disease']))

# ── Какие признаки выбраны в Filter-методе ──────────────────────────────────
print("\nПризнаки, выбранные SelectKBest (Filter):")
selector  = filter_pipeline.named_steps['feature_selection']
prep_pipe = filter_pipeline.named_steps['preprocessor']

ohe_names = (prep_pipe
    .named_transformers_['cat']
    .named_steps['encoder']
    .get_feature_names_out(categorical_features).tolist())
all_names   = numeric_features + ohe_names
mask        = selector.get_support()
selected    = [n for n, m in zip(all_names, mask) if m]
scores_all  = selector.scores_
selected_scores = [(n, s) for n, s, m in zip(all_names, scores_all, mask) if m]
selected_scores.sort(key=lambda x: -x[1])

for name, score in selected_scores:
    print(f"  {name:<20} score={score:.2f}")

