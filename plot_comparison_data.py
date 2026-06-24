import pandas as pd
import matplotlib.pyplot as plt
import os

# Настройки шрифта
plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['axes.unicode_minus'] = False

# Проверка наличия файлов
if not os.path.exists('logs/training_history_patchtst.csv'):
    print("❌ Файл training_history_patchtst.csv не найден!")
    print("   Убедитесь, что логи простых данных сохранены.")
    exit()

if not os.path.exists('logs/training_history.csv'):
    print("❌ Файл training_history.csv не найден!")
    print("   Это логи текущего эксперимента.")
    exit()

# Загрузка данных
df_simple = pd.read_csv('logs/training_history_patchtst.csv')
df_complex = pd.read_csv('logs/training_history.csv')

# Проверка колонок
print("Колонки в simple:", df_simple.columns.tolist())
print("Колонки в complex:", df_complex.columns.tolist())

# Создание графика
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. График точности
ax1 = axes[0]
ax1.plot(df_simple['val_acc'], label='Простые данные (100%)', color='#2ecc71', linewidth=2)
ax1.plot(df_complex['val_acc'], label='Усложнённые данные (99.75%)', color='#e74c3c', linewidth=2)
ax1.set_xlabel('Эпоха', fontsize=12)
ax1.set_ylabel('Точность на валидации (%)', fontsize=12)
ax1.set_title('Сравнение точности: простые vs усложнённые данные', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(90, 101)

# Аннотации лучших значений
best_simple = df_simple['val_acc'].max()
best_complex = df_complex['val_acc'].max()
ax1.annotate(f'Макс: {best_simple:.2f}%', 
             xy=(df_simple['val_acc'].idxmax(), best_simple),
             xytext=(10, 5), textcoords='offset points',
             fontsize=9, color='#2ecc71')
ax1.annotate(f'Макс: {best_complex:.2f}%', 
             xy=(df_complex['val_acc'].idxmax(), best_complex),
             xytext=(10, -15), textcoords='offset points',
             fontsize=9, color='#e74c3c')

# 2. График потерь
ax2 = axes[1]
ax2.plot(df_simple['val_loss'], label='Простые данные', color='#2ecc71', linewidth=2)
ax2.plot(df_complex['val_loss'], label='Усложнённые данные', color='#e74c3c', linewidth=2)
ax2.set_xlabel('Эпоха', fontsize=12)
ax2.set_ylabel('Потеря на валидации (Loss)', fontsize=12)
ax2.set_title('Сравнение потерь: простые vs усложнённые данные', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 0.15)

# Аннотации лучших значений
min_loss_simple = df_simple['val_loss'].min()
min_loss_complex = df_complex['val_loss'].min()
ax2.annotate(f'Мин: {min_loss_simple:.4f}', 
             xy=(df_simple['val_loss'].idxmin(), min_loss_simple),
             xytext=(10, 5), textcoords='offset points',
             fontsize=9, color='#2ecc71')
ax2.annotate(f'Мин: {min_loss_complex:.4f}', 
             xy=(df_complex['val_loss'].idxmin(), min_loss_complex),
             xytext=(10, -15), textcoords='offset points',
             fontsize=9, color='#e74c3c')

plt.tight_layout()
#plt.savefig('comparison_simple_vs_complex.png', dpi=150, bbox_inches='tight')
#plt.savefig('images/comparison_simple_vs_complex.png', dpi=150, bbox_inches='tight')
plt.show()

# Вывод статистики
print("\n" + "="*50)
print("СРАВНИТЕЛЬНАЯ СТАТИСТИКА")
print("="*50)
print(f"Простые данные:")
print(f"  Лучшая точность: {best_simple:.2f}%")
print(f"  Лучшая потеря:   {min_loss_simple:.4f}")
print(f"  Эпох:            {len(df_simple)}")
print(f"\nУсложнённые данные:")
print(f"  Лучшая точность: {best_complex:.2f}%")
print(f"  Лучшая потеря:   {min_loss_complex:.4f}")
print(f"  Эпох:            {len(df_complex)}")
print(f"\nРазница:")
print(f"  Точность:  {best_simple - best_complex:.2f}%")
print(f"  Потеря:    {min_loss_complex - min_loss_simple:.4f}")
print("="*50)