"""
Step 1: EDA — Heart Disease Dataset
====================================
Датасет: UCI Cleveland Heart Disease (303 строки, 13 признаков + target)

Признаки:
  age       — возраст
  sex       — пол (1=мужской, 0=женский)
  cp        — тип боли в груди (0-3)
  trestbps  — давление в покое (мм рт. ст.)
  chol      — холестерин (мг/дл)
  fbs       — сахар натощак > 120 мг/дл (1=да)
  restecg   — результат ЭКГ в покое (0-2)
  thalach   — макс. ЧСС
  exang     — стенокардия при нагрузке (1=да)
  oldpeak   — депрессия ST при нагрузке
  slope     — наклон ST (0-2)
  ca        — кол-во крупных сосудов (0-3), есть пропуски
  thal      — тип таласемии (3,6,7), есть пропуски
  target    — 0=нет болезни, 1=болезнь
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Загрузка ---
df = pd.read_csv('data/heart.csv')

print("=" * 50)
print("БАЗОВАЯ ИНФОРМАЦИЯ")
print("=" * 50)
print(f"Размер датасета: {df.shape}")
print(f"\nТипы данных:\n{df.dtypes}")

print("\n" + "=" * 50)
print("ПРОПУСКИ")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА")
print("=" * 50)
print(df.describe().round(2))

print("\n" + "=" * 50)
print("БАЛАНС КЛАССОВ")
print("=" * 50)
vc = df['target'].value_counts()
print(f"  Нет болезни (0): {vc[0]}  ({vc[0]/len(df)*100:.1f}%)")
print(f"  Болезнь     (1): {vc[1]}  ({vc[1]/len(df)*100:.1f}%)")

print("\n" + "=" * 50)
print("КОРРЕЛЯЦИЯ С ТАРГЕТОМ (топ-5)")
print("=" * 50)
num_df = df.select_dtypes(include=np.number).dropna()
corr = num_df.corr()['target'].drop('target').abs().sort_values(ascending=False)
print(corr.head(5).round(3))

print("\nEDA завершён. Выводы:")
print("  ✓ Пропуски только в 'ca' и 'thal' — обработаем в Pipeline")
print("  ✓ Баланс классов ~46/54% — приемлем, но учтём при выборе метрик")
print("  ✓ Числовые и категориальные признаки — нужен ColumnTransformer")
