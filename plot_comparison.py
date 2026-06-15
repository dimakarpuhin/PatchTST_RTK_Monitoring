import pandas as pd
import matplotlib.pyplot as plt
import os

# Настройки русского шрифта
plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['axes.unicode_minus'] = False

# Загрузка данных
patchtst_log = pd.read_csv('logs/training_history.csv')
lstm_log = pd.read_csv('logs/training_history_lstm.csv')  # переименуйте файл с LSTM

# Создание графика
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. График точности (Accuracy)
ax1 = axes[0]
ax1.plot(patchtst_log['val_acc'], label='PatchTST', color='#e74c3c', linewidth=2)
ax1.plot(lstm_log['val_acc'], label='LSTM', color='#3498db', linewidth=2)
ax1.set_xlabel('Эпоха', fontsize=12)
ax1.set_ylabel('Точность на валидации (%)', fontsize=12)
ax1.set_title('Сравнение точности: PatchTST vs LSTM', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Добавление аннотаций с лучшими значениями
best_patch = patchtst_log['val_acc'].max()
best_lstm = lstm_log['val_acc'].max()
ax1.annotate(f'Max: {best_patch:.2f}%', 
             xy=(patchtst_log['val_acc'].idxmax(), best_patch),
             xytext=(10, 5), textcoords='offset points',
             fontsize=9, color='#e74c3c')
ax1.annotate(f'Max: {best_lstm:.2f}%', 
             xy=(lstm_log['val_acc'].idxmax(), best_lstm),
             xytext=(10, -15), textcoords='offset points',
             fontsize=9, color='#3498db')

# 2. График потерь (Loss)
ax2 = axes[1]
ax2.plot(patchtst_log['val_loss'], label='PatchTST', color='#e74c3c', linewidth=2)
ax2.plot(lstm_log['val_loss'], label='LSTM', color='#3498db', linewidth=2)
ax2.set_xlabel('Эпоха', fontsize=12)
ax2.set_ylabel('Потеря на валидации (Loss)', fontsize=12)
ax2.set_title('Сравнение функции потерь: PatchTST vs LSTM', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Добавление аннотаций с лучшими значениями
min_loss_patch = patchtst_log['val_loss'].min()
min_loss_lstm = lstm_log['val_loss'].min()
ax2.annotate(f'Min: {min_loss_patch:.4f}', 
             xy=(patchtst_log['val_loss'].idxmin(), min_loss_patch),
             xytext=(10, 5), textcoords='offset points',
             fontsize=9, color='#e74c3c')
ax2.annotate(f'Min: {min_loss_lstm:.4f}', 
             xy=(lstm_log['val_loss'].idxmin(), min_loss_lstm),
             xytext=(10, -15), textcoords='offset points',
             fontsize=9, color='#3498db')

plt.tight_layout()
#plt.savefig('comparison_patchtst_vs_lstm.png', dpi=150, bbox_inches='tight')
plt.savefig('logs/comparison_patchtst_vs_lstm.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 50)
print("СВОДНОЕ СРАВНЕНИЕ")
print("=" * 50)
print(f"PatchTST - Лучшая точность: {best_patch:.2f}%")
print(f"LSTM     - Лучшая точность: {best_lstm:.2f}%")
print(f"Разница: {best_patch - best_lstm:.2f}%")
print(f"\nPatchTST - Лучшая потеря: {min_loss_patch:.4f}")
print(f"LSTM     - Лучшая потеря: {min_loss_lstm:.4f}")
print(f"Разница: {min_loss_lstm - min_loss_patch:.4f}")
print("=" * 50)