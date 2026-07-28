# plot_few_shot_pm.py
# Построение графика Few-Shot Learning для диссертации

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Настройки русского шрифта
plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['axes.unicode_minus'] = False

# Данные из эксперимента (F1-score)
data = {
    'Размер выборки': ['5% (24)', '10% (48)', '25% (120)', '50% (240)', '100% (480)'],
    'PatchTST (модиф.)': [0.5714, 0.5607, 0.5545, 0.5155, 0.6154],
    'PatchTST (базовый)': [0.5524, 0.4444, 0.4176, 0.5437, 0.5361],
    'LSTM': [0.0000, 0.6602, 0.4938, 0.5067, 0.4533],
    'GRU': [0.5957, 0.5591, 0.3117, 0.5250, 0.5610],
    'TCN': [0.4330, 0.4902, 0.5391, 0.4225, 0.5301],
    'Transformer': [0.4381, 0.4752, 0.4848, 0.5849, 0.5814]
}

# Создаём DataFrame
df = pd.DataFrame(data)

# Количество образцов (для оси X)
samples = [24, 48, 120, 240, 480]

# Цвета для моделей
colors = {
    'PatchTST (модиф.)': '#2ecc71',    # зелёный
    'PatchTST (базовый)': '#e74c3c',   # красный
    'LSTM': '#3498db',                 # синий
    'GRU': '#f39c12',                  # оранжевый
    'TCN': '#9b59b6',                  # фиолетовый
    'Transformer': '#1abc9c'           # бирюзовый
}

# Стили линий
linestyles = {
    'PatchTST (модиф.)': '-',
    'PatchTST (базовый)': '--',
    'LSTM': '-.',
    'GRU': ':',
    'TCN': '-',
    'Transformer': '--'
}

# Создание графика
fig, ax = plt.subplots(figsize=(12, 7))

# Построение линий для каждой модели
for model in df.columns[1:]:
    ax.plot(samples, df[model], 
            marker='o', 
            linewidth=2.5, 
            markersize=8,
            color=colors.get(model, '#000000'),
            linestyle=linestyles.get(model, '-'),
            label=model)

# Настройка графика
ax.set_xlabel('Размер обучающей выборки (количество образцов)', fontsize=13)
ax.set_ylabel('F1-score', fontsize=13)
ax.set_title('Few-Shot Learning: зависимость F1-score от размера выборки', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 520)
ax.set_ylim(-0.05, 0.75)

# Добавление аннотаций для лучших значений
# 1. PatchTST при 100%
ax.annotate(f'{df["PatchTST (модиф.)"].iloc[-1]:.4f}', 
            xy=(480, 0.6154), xytext=(440, 0.66),
            arrowprops=dict(arrowstyle='->', color='#2ecc71'),
            fontsize=10, color='#2ecc71')

# 2. LSTM при 10%
ax.annotate(f'{df["LSTM"].iloc[1]:.4f}', 
            xy=(48, 0.6602), xytext=(80, 0.70),
            arrowprops=dict(arrowstyle='->', color='#3498db'),
            fontsize=10, color='#3498db')

# 3. GRU при 5%
ax.annotate(f'{df["GRU"].iloc[0]:.4f}', 
            xy=(24, 0.5957), xytext=(50, 0.62),
            arrowprops=dict(arrowstyle='->', color='#f39c12'),
            fontsize=10, color='#f39c12')

# 4. Transformer при 50%
ax.annotate(f'{df["Transformer"].iloc[3]:.4f}', 
            xy=(240, 0.5849), xytext=(200, 0.62),
            arrowprops=dict(arrowstyle='->', color='#1abc9c'),
            fontsize=10, color='#1abc9c')

plt.tight_layout()

# Сохранение
os.makedirs('images', exist_ok=True)
plt.savefig('images/few_shot_f1_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('images/few_shot_f1_comparison.pdf', format='pdf', bbox_inches='tight')
print("✅ График сохранён в images/few_shot_f1_comparison.png")
print("✅ График сохранён в images/few_shot_f1_comparison.pdf")

plt.show()

# ============================================================
# ВТОРОЙ ГРАФИК: Test Accuracy
# ============================================================

# Данные для Accuracy
acc_data = {
    'Размер выборки': ['5% (24)', '10% (48)', '25% (120)', '50% (240)', '100% (480)'],
    'PatchTST (модиф.)': [53.40, 54.37, 56.31, 54.37, 61.17],
    'PatchTST (базовый)': [54.37, 51.46, 48.54, 54.37, 56.31],
    'LSTM': [50.49, 66.02, 60.19, 64.08, 60.19],
    'GRU': [63.11, 60.19, 48.54, 63.11, 65.05],
    'TCN': [46.60, 49.51, 48.54, 60.19, 62.14],
    'Transformer': [42.72, 48.54, 50.49, 57.28, 65.05]
}

df_acc = pd.DataFrame(acc_data)

fig2, ax2 = plt.subplots(figsize=(12, 7))

for model in df_acc.columns[1:]:
    ax2.plot(samples, df_acc[model], 
             marker='s', 
             linewidth=2.5, 
             markersize=8,
             color=colors.get(model, '#000000'),
             linestyle=linestyles.get(model, '-'),
             label=model)

ax2.set_xlabel('Размер обучающей выборки (количество образцов)', fontsize=13)
ax2.set_ylabel('Test Accuracy (%)', fontsize=13)
ax2.set_title('Few-Shot Learning: зависимость точности от размера выборки', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 520)
ax2.set_ylim(35, 75)

plt.tight_layout()
plt.savefig('images/few_shot_acc_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('images/few_shot_acc_comparison.pdf', format='pdf', bbox_inches='tight')
print("✅ График сохранён в images/few_shot_acc_comparison.png")
print("✅ График сохранён в images/few_shot_acc_comparison.pdf")

plt.show()

print("\n" + "=" * 60)
print("📊 СТАТИСТИКА ПО ГРАФИКАМ")
print("=" * 60)
print(f"\nЛучшая модель при 100% данных: PatchTST (модиф.) F1 = {df['PatchTST (модиф.)'].iloc[-1]:.4f}")
print(f"Лучшая модель при 25% данных:  PatchTST (модиф.) F1 = {df['PatchTST (модиф.)'].iloc[2]:.4f}")
print(f"Лучшая модель при 10% данных:  LSTM F1 = {df['LSTM'].iloc[1]:.4f}")
print(f"Лучшая модель при 5% данных:   GRU F1 = {df['GRU'].iloc[0]:.4f}")
print("=" * 60)