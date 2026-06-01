import requests
import numpy as np
import pandas as pd

# Загрузка данных
df = pd.read_csv('data/synthetic_data.csv')
data_row = df.iloc[0].drop('label').values
window = data_row.reshape(512, 4).tolist()

print(f'Отправляем окно размером {len(window)}x{len(window[0])}')

# Отправка запроса
response = requests.post('http://localhost:8000/classify', json={'window': window})
print(f'Status: {response.status_code}')

if response.status_code == 200:
    result = response.json()
    print(f'Класс: {result["class_name"]}')
    print(f'Вероятность: {max(result["probabilities"]):.2%}')
    print(f'Время: {result["latency_ms"]} мс')
else:
    print(f'Ошибка: {response.text}')