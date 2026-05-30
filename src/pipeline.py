"""
Step 2: Preprocessing Pipeline
================================
Собираем полный ColumnTransformer + Pipeline из scikit-learn.

Числовые признаки:  SimpleImputer(median) → StandardScaler
Категориальные:     SimpleImputer(most_frequent) → OneHotEncoder
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# ── 1. Загрузка данных ──────────────────────────────────────────────────────
df = pd.read_csv('data/heart.csv')
X = df.drop('target', axis=1)
y = df['target']

# ── 2. Разделение на train/test ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── 3. Определяем типы признаков ────────────────────────────────────────────
# Числовые: заполняем медианой + масштабируем
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']

# Категориальные: заполняем модой + OHE (handle_unknown='ignore' — защита в продакшне)
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']

print(f"\nЧисловые признаки  ({len(numeric_features)}): {numeric_features}")
print(f"Категориальные     ({len(categorical_features)}): {categorical_features}")

# ── 4. Подпайплайны для каждого типа ────────────────────────────────────────
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),   # обработка пропусков
    ('scaler',  StandardScaler()),                   # масштабирование
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # обработка пропусков
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

# ── 5. ColumnTransformer — объединяем оба трансформера ──────────────────────
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer,  numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

# ── 6. Финальный Pipeline: preprocessor + модель ────────────────────────────
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',   LogisticRegression(random_state=42, max_iter=1000)),
])

# ── 7. Обучение и базовые результаты ────────────────────────────────────────
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n" + "=" * 50)
print("РЕЗУЛЬТАТЫ (базовая LogisticRegression)")
print("=" * 50)
print(classification_report(y_test, y_pred,
      target_names=['No disease', 'Disease']))

# ── 8. Проверяем структуру трансформированных данных ────────────────────────
X_transformed = preprocessor.fit_transform(X_train)
num_ohe_features = (preprocessor
    .named_transformers_['cat']
    .named_steps['encoder']
    .get_feature_names_out(categorical_features)
    .tolist())
all_features = numeric_features + num_ohe_features

print(f"Признаков после преобразования: {X_transformed.shape[1]}")
print(f"  Числовых (scaled): {len(numeric_features)}")
print(f"  После OHE:         {len(num_ohe_features)}")
print(f"\nВсе признаки после преобразования:")
for i, f in enumerate(all_features):
    print(f"  [{i:2d}] {f}")

print("\n✓ Pipeline построен и проверен!")
print("  Следующий шаг: добавить отбор признаков (SelectKBest / RFE)")
