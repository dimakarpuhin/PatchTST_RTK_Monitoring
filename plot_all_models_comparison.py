import pandas as pd
import matplotlib.pyplot as plt
import os

# Настройки русского шрифта
plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['axes.unicode_minus'] = False

# Список файлов и названий моделей
model_files = {
    'PatchTST (базовый)': 'logs/training_history_patchtst.csv',
    'PatchTST (модифицированный)': 'logs/training_history_patchtst_complex.csv',
    'LSTM': 'logs/training_history_lstm.csv',
    'GRU': 'logs/training_history_gru.csv',
    'TCN': 'logs/training_history_tcn.csv',
    'Transformer': 'logs/training_history_transformer.csv'
}

# Цвета для моделей
colors = {
    'PatchTST (базовый)': '#e74c3c',   # красный
    'PatchTST (модифицированный)': '#c0392b',    # тёмно-красный
    'LSTM': '#3498db',                      # синий
    'GRU': '#2ecc71',                       # зелёный
    'TCN': '#f39c12',                       # оранжевый
    'Transformer': '#9b59b6'                # фиолетовый
}

# Загрузка данных
data = {}
for name, path in model_files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        data[name] = df
        print(f"✅ Загружены {name}: {len(df)} эпох")
    else:
        print(f"⚠️ Файл {path} не найден для {name}")

# Проверка, что есть данные
if not data:
    print("❌ Нет данных для построения графиков!")
    exit()

# Создание фигуры
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. График точности (Val Acc)
ax1 = axes[0]
for name, df in data.items():
    if 'val_acc' in df.columns:
        ax1.plot(df['val_acc'], label=name, color=colors.get(name, '#000000'), linewidth=2)
ax1.set_xlabel('Эпоха', fontsize=12)
ax1.set_ylabel('Точность на валидации (%)', fontsize=12)
ax1.set_title('Сравнение точности моделей', fontsize=14)
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 105)

# 2. График потерь (Val Loss)
ax2 = axes[1]
for name, df in data.items():
    if 'val_loss' in df.columns:
        ax2.plot(df['val_loss'], label=name, color=colors.get(name, '#000000'), linewidth=2)
ax2.set_xlabel('Эпоха', fontsize=12)
ax2.set_ylabel('Потеря на валидации (Loss)', fontsize=12)
ax2.set_title('Сравнение функции потерь', fontsize=14)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 4.5)

plt.tight_layout()
plt.savefig('images/comparison_all_models.png', dpi=150, bbox_inches='tight')
plt.savefig('comparison_all_models.png', dpi=150, bbox_inches='tight')
plt.show()

# Вывод итоговой таблицы
print("\n" + "="*70)
print("ИТОГОВОЕ СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")
print("="*70)
print(f"{'Модель':<25} {'Best Val Acc':<15} {'Final Val Acc':<15} {'Epochs':<10}")
print("-"*70)

for name, df in data.items():
    best_acc = df['val_acc'].max()
    final_acc = df['val_acc'].iloc[-1]
    epochs = len(df)
    print(f"{name:<25} {best_acc:<15.2f}% {final_acc:<15.2f}% {epochs:<10}")

print("="*70)