# check_pm_data.py
import pandas as pd
import numpy as np

df = pd.read_csv('data/predictive_maintenance/raw/predictive_maintenance.csv')

print("=" * 60)
print("📊 ПРОВЕРКА ДАННЫХ")
print("=" * 60)

print(f"\nРазмер: {df.shape}")
print(f"\nКолонки: {df.columns.tolist()}")
print(f"\nПервые 5 строк:")
print(df.head())

print(f"\nТипы данных:")
print(df.dtypes)

print(f"\nСтатистика:")
print(df.describe())

# ===== ИСПРАВЛЕНО: распределение отказов =====
print(f"\n📊 Распределение типов отказов:")
failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
for col in failure_cols:
    count = df[col].sum()
    print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")

# ===== ИСПРАВЛЕНО: общее количество отказов =====
df['FAILURE'] = df[failure_cols].sum(axis=1) > 0
total_failures = df['FAILURE'].sum()
print(f"\n  Всего отказов: {total_failures} ({total_failures/len(df)*100:.2f}%)")

print(f"\nПроверка на NaN:")
print(df.isnull().sum())