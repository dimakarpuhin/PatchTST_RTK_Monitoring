import pandas as pd
import matplotlib.pyplot as plt
import os

# Настройки русского шрифта
plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['axes.unicode_minus'] = False

# Пути к файлам логов (укажите правильные)
log_files = {
    'PatchTST': 'logs/training_history_patchtst.csv',
    'LSTM': 'logs/training_history_lstm.csv',
    'GRU': 'logs/training_history.csv'  # после обучения GRU
}

# Проверка существования файлов
for name, path in log_files.items():
    if not os.path.exists(path):
        print(f"⚠️ Файл {path} не найден для {name}")
        log_files[name] = None

# Загрузка данных
data = {}
for name, path in log_files.items():
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        data[name] = df
        print(f"✅ Загружены {name}: {len(df)} эпох")

# Цвета для моделей
colors = {
    'PatchTST': '#e74c3c',  # красный
    'LSTM': '#3498db',      # синий
    'GRU': '#2ecc71'        # зелёный
}

# Создание фигуры с двумя подграфиками
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 1. График точности (Accuracy)
ax1 = axes[0]
for name, df in data.items():
    if 'val_acc' in df.columns:
        ax1.plot(df['val_acc'], label=name, color=colors[name], linewidth=2)
ax1.set_xlabel('Эпоха', fontsize=12)
ax1.set_ylabel('Точность на валидации (%)', fontsize=12)
ax1.set_title('Сравнение точности: PatchTST vs LSTM vs GRU', fontsize=14)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(20, 102)

# Добавление аннотаций с лучшими значениями
for name, df in data.items():
    if 'val_acc' in df.columns:
        best_acc = df['val_acc'].max()
        best_epoch = df['val_acc'].idxmax()
        ax1.annotate(f'{name}: {best_acc:.2f}%', 
                     xy=(best_epoch, best_acc),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, color=colors[name])

# 2. График потерь (Loss)
ax2 = axes[1]
for name, df in data.items():
    if 'val_loss' in df.columns:
        ax2.plot(df['val_loss'], label=name, color=colors[name], linewidth=2)
ax2.set_xlabel('Эпоха', fontsize=12)
ax2.set_ylabel('Потеря на валидации (Loss)', fontsize=12)
ax2.set_title('Сравнение функции потерь: PatchTST vs LSTM vs GRU', fontsize=14)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 3)

# Добавление аннотаций с лучшими значениями
for name, df in data.items():
    if 'val_loss' in df.columns:
        best_loss = df['val_loss'].min()
        best_epoch = df['val_loss'].idxmin()
        ax2.annotate(f'{name}: {best_loss:.4f}', 
                     xy=(best_epoch, best_loss),
                     xytext=(5, -10), textcoords='offset points',
                     fontsize=8, color=colors[name])

plt.tight_layout()
#plt.savefig('comparison_3_models.png', dpi=150, bbox_inches='tight')
plt.savefig('images/comparison_3_models.png', dpi=150, bbox_inches='tight')
plt.show()

# Вывод итоговой таблицы
print("\n" + "="*60)
print("ИТОГОВОЕ СРАВНЕНИЕ ТРЁХ МОДЕЛЕЙ")
print("="*60)
print(f"{'Модель':<12} {'Best Acc':<12} {'Best Loss':<12} {'Эпохи':<8} {'Параметры':<12}")
print("-"*60)

for name, df in data.items():
    best_acc = df['val_acc'].max()
    best_loss = df['val_loss'].min()
    epochs = len(df)
    params = {'PatchTST': '159k', 'LSTM': '~210k', 'GRU': '~160k'}.get(name, '?')
    print(f"{name:<12} {best_acc:<12.2f}% {best_loss:<12.4f} {epochs:<8} {params:<12}")
print("="*60)