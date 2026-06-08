import pandas as pd
import matplotlib.pyplot as plt
import os

# Настройки русского шрифта (для Windows)
plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['axes.unicode_minus'] = False

# Загрузка данных
log_file = 'logs/training_history.csv'
if not os.path.exists(log_file):
    print(f"Файл {log_file} не найден!")
    exit()

df = pd.read_csv(log_file)
print("Доступные колонки:", df.columns.tolist())
print(f"Загружено {len(df)} эпох")

# Создание фигуры с двумя подграфиками
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 1. График функции потерь (Loss)
ax1.plot(df['train_loss'], label='Потери при обучении', color='#e74c3c', linewidth=2)
ax1.plot(df['val_loss'], label='Потери при валидации', color='#3498db', linewidth=2)
ax1.set_xlabel('Эпоха', fontsize=12)
ax1.set_ylabel('Потери', fontsize=12)
ax1.set_title('Функция потерь', fontsize=14) # Для формулы 2.8
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. График точности (Accuracy)
ax2.plot(df['train_acc'], label='Точность обучения', color='#e74c3c', linewidth=2)
ax2.plot(df['val_acc'], label='Точность валидации', color='#3498db', linewidth=2)
ax2.set_xlabel('Эпоха', fontsize=12)
ax2.set_ylabel('Точность (%)', fontsize=12)
ax2.set_title('Точность классификации', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Вывод лучших значений
best_val_acc = df['val_acc'].max()
best_val_loss = df['val_loss'].min()
print(f"\nЛучшая точность на валидации: {best_val_acc:.2f}%")
print(f"Лучшая потеря на валидации: {best_val_loss:.4f}")

# Добавление аннотации на график
ax2.annotate(f'Best: {best_val_acc:.1f}%', 
             xy=(df['val_acc'].idxmax(), best_val_acc),
             xytext=(5, 5), textcoords='offset points',
             fontsize=10, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('logs/training_plot.png', dpi=150, bbox_inches='tight')
plt.savefig('training_plot.png', dpi=150, bbox_inches='tight')
print("\nГрафики сохранены:")
print("  - logs/training_plot.png")
print("  - training_plot.png")

plt.show()