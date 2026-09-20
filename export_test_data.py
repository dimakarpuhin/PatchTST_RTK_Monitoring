# export_test_data.py
import numpy as np
import pandas as pd
from config import Config

# Загружаем тестовую выборку
X_test = np.load(f'{Config.PM_PROCESSED_PATH}/X_test.npy')
y_test = np.load(f'{Config.PM_PROCESSED_PATH}/y_test.npy')

# Берём первый образец (одно окно)
sample = X_test[0]  # shape: (500, 5)

# Создаём DataFrame
df = pd.DataFrame(sample, columns=[
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
])

# Сохраняем
df.to_csv('data/sample_pm_data.csv', index=False)
print(f"✅ Сохранён образец из тестовой выборки в data/sample_pm_data.csv")
print(f"   Форма: {sample.shape}")
print(f"   Класс: {y_test[0]} (0 — Норма, 1 — Отказ)")