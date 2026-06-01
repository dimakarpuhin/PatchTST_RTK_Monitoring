# admin.py
# GUI администратора для управления нейросетевым модулем
# Функции: управление БД, дообучение модели, настройка порогов

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import requests
import json
import os
import subprocess
import threading
import time
from datetime import datetime
import pandas as pd
import numpy as np

from config import Config

# Попытка импорта matplotlib (для графиков)
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Предупреждение: matplotlib не установлен. Графики недоступны.")

# Конфигурация подключения к серверу
SERVER_URL = "http://localhost:8000"
API_HEALTH = f"{SERVER_URL}/health"
API_UPDATE_MODEL = f"{SERVER_URL}/model/update"
API_CLASSES = f"{SERVER_URL}/classes"

# Пути к файлам
MODEL_PATH = Config.MODEL_SAVE_PATH
DATA_PATH = Config.DATA_PATH
TRAIN_SCRIPT = "train.py"

# Классы неопределённостей (будут загружены с сервера, но есть значения по умолчанию)
CLASS_NAMES = {
    0: "Норма",
    1: "Скачок напряжения",
    2: "Токовая перегрузка",
    3: "Перегрев",
    4: "Электропомеха"
}


class AdminInterface:
    """Графический интерфейс администратора"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Нейросетевой модуль классификации - Администратор")
        self.root.geometry("1200x700")
        self.root.configure(bg='#f0f0f0')
        
        # Переменные
        self.server_available = False
        self.db_data = None
        self.db_labels = None
        self.last_result = None
        
        self.setup_ui()
        self.check_server_connection()
        self.fetch_classes()
        self.load_database_info()
        
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        
        # Верхняя панель
        top_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        top_frame.pack(fill=tk.X)
        top_frame.pack_propagate(False)
        
        title_label = tk.Label(
            top_frame,
            text="Панель администратора - Управление нейросетевым модулем",
            font=('Arial', 16, 'bold'),
            fg='white', bg='#2c3e50'
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Статус сервера
        status_frame = tk.Frame(top_frame, bg='#2c3e50')
        status_frame.pack(side=tk.RIGHT, padx=20, pady=15)
        
        self.status_indicator = tk.Label(
            status_frame,
            text="Проверка...",
            font=('Arial', 10, 'bold'),
            fg='white', bg='#2c3e50'
        )
        self.status_indicator.pack()
        
        # Основной контент
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - управление
        left_frame = tk.LabelFrame(main_frame, text="Управление", 
                                    font=('Arial', 14, 'bold'), bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Кнопки управления
        buttons = [
            ("🔄 Обновить модель (из файла)", self.update_model),
            ("📊 Запустить дообучение", self.retrain_model),
            ("📁 Управление БД", self.manage_database),
            ("⚙️ Настройка порогов", self.configure_thresholds),
            ("📤 Экспорт данных", self.export_data),
            ("📥 Импорт данных", self.import_data),
            ("📈 Показать метрики", self.show_metrics),
            ("🔄 Перезапустить сервер", self.restart_server)
        ]
        
        for text, command in buttons:
            btn = tk.Button(
                left_frame,
                text=text,
                command=command,
                font=('Arial', 11),
                bg='#3498db',
                fg='white',
                width=25,
                height=2
            )
            btn.pack(padx=10, pady=5)
        
        # Центральная панель - информация о модели
        center_frame = tk.LabelFrame(main_frame, text="Информация о модели",
                                      font=('Arial', 14, 'bold'), bg='#f0f0f0')
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Текстовое поле для вывода информации
        self.info_text = tk.Text(center_frame, font=('Courier', 10), height=20, width=50)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = tk.Scrollbar(self.info_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.info_text.yview)
        
        # Правая панель - база данных
        right_frame = tk.LabelFrame(main_frame, text="База данных неопределённостей",
                                     font=('Arial', 14, 'bold'), bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Таблица с данными
        columns = ('ID', 'Класс', 'Размер')
        self.tree = ttk.Treeview(right_frame, columns=columns, show='headings')
        
        self.tree.heading('ID', text='ID')
        self.tree.heading('Класс', text='Класс')
        self.tree.heading('Размер', text='Размер (отсчёты)')
        
        self.tree.column('ID', width=50)
        self.tree.column('Класс', width=150)
        self.tree.column('Размер', width=100)
        
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Кнопки управления БД
        db_buttons_frame = tk.Frame(right_frame, bg='#f0f0f0')
        db_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        add_btn = tk.Button(db_buttons_frame, text="➕ Добавить", command=self.add_sample,
                            bg='#2ecc71', fg='white')
        add_btn.pack(side=tk.LEFT, padx=5)
        
        delete_btn = tk.Button(db_buttons_frame, text="❌ Удалить", command=self.delete_sample,
                               bg='#e74c3c', fg='white')
        delete_btn.pack(side=tk.LEFT, padx=5)
        
        edit_btn = tk.Button(db_buttons_frame, text="✏️ Редактировать", command=self.edit_sample,
                             bg='#f39c12', fg='white')
        edit_btn.pack(side=tk.LEFT, padx=5)
        
        # Статусная строка
        status_frame = tk.Frame(self.root, bg='#ecf0f1', height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar(value="Готов к работе")
        status_label = tk.Label(status_frame, textvariable=self.status_var,
                                 bg='#ecf0f1', font=('Arial', 9))
        status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
    def fetch_classes(self):
        """Получение списка классов с сервера"""
        try:
            response = requests.get(API_CLASSES, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'classes' in data:
                    global CLASS_NAMES
                    CLASS_NAMES = data['classes']
                    self.status_var.set("Список классов загружен с сервера")
        except Exception as e:
            pass  # Используем локальные имена классов
        
    def check_server_connection(self):
        """Проверка доступности сервера"""
        try:
            response = requests.get(API_HEALTH, timeout=5)
            if response.status_code == 200:
                self.server_available = True
                self.status_indicator.config(text="Сервер: Доступен ✓", fg='#2ecc71')
                self.update_model_info()
            else:
                self.server_available = False
                self.status_indicator.config(text="Сервер: Ошибка ✗", fg='#e74c3c')
        except:
            self.server_available = False
            self.status_indicator.config(text="Сервер: Недоступен ✗", fg='#e74c3c')
            
        # Повторная проверка через 10 секунд
        self.root.after(10000, self.check_server_connection)
        
    def update_model_info(self):
        """Обновление информации о модели в текстовом поле"""
        self.info_text.delete(1.0, tk.END)
        
        # Информация из конфигурации
        self.info_text.insert(tk.END, "=" * 50 + "\n")
        self.info_text.insert(tk.END, "КОНФИГУРАЦИЯ МОДЕЛИ\n")
        self.info_text.insert(tk.END, "=" * 50 + "\n")
        self.info_text.insert(tk.END, f"Длина окна T:           {Config.WINDOW_LENGTH}\n")
        self.info_text.insert(tk.END, f"Число параметров d:     {Config.NUM_CHANNELS}\n")  # ИСПРАВЛЕНО
        self.info_text.insert(tk.END, f"Число классов C:        {Config.NUM_CLASSES}\n")
        self.info_text.insert(tk.END, f"Длина патча L:          {Config.PATCH_LEN}\n")
        self.info_text.insert(tk.END, f"Шаг патча S:            {Config.PATCH_STRIDE}\n")
        self.info_text.insert(tk.END, f"Размер эмбеддинга d_model: {Config.D_MODEL}\n")
        self.info_text.insert(tk.END, f"Число слоёв N_layers:   {Config.NUM_LAYERS}\n")
        self.info_text.insert(tk.END, f"Число голов H:          {Config.NUM_HEADS}\n")
        self.info_text.insert(tk.END, f"λ₁ (L2):                {Config.LAMBDA_1}\n")
        self.info_text.insert(tk.END, f"λ₂ (контраст):          {Config.LAMBDA_2}\n")
        self.info_text.insert(tk.END, f"Температура τ:          {Config.TEMPERATURE}\n")
        self.info_text.insert(tk.END, f"Устройство:             {Config.DEVICE}\n")
        
        self.info_text.insert(tk.END, "\n" + "=" * 50 + "\n")
        self.info_text.insert(tk.END, "ФАЙЛЫ МОДЕЛИ\n")
        self.info_text.insert(tk.END, "=" * 50 + "\n")
        
        if os.path.exists(Config.MODEL_SAVE_PATH):
            size = os.path.getsize(Config.MODEL_SAVE_PATH) / (1024 * 1024)
            self.info_text.insert(tk.END, f"Файл модели: {Config.MODEL_SAVE_PATH}\n")
            self.info_text.insert(tk.END, f"Размер: {size:.2f} МБ\n")
            self.info_text.insert(tk.END, f"Модифицирован: {datetime.fromtimestamp(os.path.getmtime(Config.MODEL_SAVE_PATH))}\n")
        else:
            self.info_text.insert(tk.END, "Файл модели не найден\n")
            self.info_text.insert(tk.END, f"Обучите модель: python {TRAIN_SCRIPT}\n")
        
        if os.path.exists(Config.DATA_PATH):
            size = os.path.getsize(Config.DATA_PATH) / (1024 * 1024)
            self.info_text.insert(tk.END, f"\nФайл данных: {Config.DATA_PATH}\n")
            self.info_text.insert(tk.END, f"Размер: {size:.2f} МБ\n")
        
    def load_database_info(self):
        """Загрузка информации о базе данных"""
        if os.path.exists(Config.DATA_PATH):
            try:
                df = pd.read_csv(Config.DATA_PATH)
                if 'label' in df.columns:
                    labels = df['label'].values
                    
                    # Очистка таблицы
                    for item in self.tree.get_children():
                        self.tree.delete(item)
                    
                    # ИСПРАВЛЕНО: показываем все классы (включая с нулевым количеством)
                    for class_id, class_name in CLASS_NAMES.items():
                        count = np.sum(labels == class_id)
                        if count > 0:
                            self.tree.insert('', tk.END, values=(class_id, class_name, count))
                        else:
                            self.tree.insert('', tk.END, values=(class_id, class_name, 0), tags=('empty',))
                    
                    # Настройка цвета для пустых классов
                    self.tree.tag_configure('empty', foreground='gray')
                    
                    self.status_var.set(f"Загружено {len(df)} образцов")
            except Exception as e:
                self.status_var.set(f"Ошибка загрузки БД: {str(e)}")
        
    def update_model(self):
        """Обновление модели из файла"""
        if not self.server_available:
            messagebox.showerror("Ошибка", "Сервер недоступен")
            return
            
        def _update():
            self.status_var.set("Обновление модели...")
            try:
                response = requests.post(API_UPDATE_MODEL, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == 'success':
                        self.status_var.set("Модель успешно обновлена")
                        messagebox.showinfo("Успех", "Модель обновлена")
                        self.update_model_info()
                    else:
                        self.status_var.set(f"Ошибка: {data['message']}")
                else:
                    self.status_var.set(f"Ошибка сервера: {response.status_code}")
            except Exception as e:
                self.status_var.set(f"Ошибка: {str(e)}")
                
        threading.Thread(target=_update, daemon=True).start()
        
    def retrain_model(self):
        """Запуск процесса дообучения модели"""
        result = messagebox.askyesno(
            "Подтверждение",
            "Запуск дообучения модели может занять несколько минут.\n"
            "Сервер будет недоступен на время обучения.\n"
            "Продолжить?"
        )
        
        if not result:
            return
            
        self.status_var.set("Запуск дообучения...")
        
        def _train():
            try:
                # Запуск процесса обучения
                result = subprocess.run(
                    ["python", TRAIN_SCRIPT],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 минут
                )
                
                if result.returncode == 0:
                    self.status_var.set("Дообучение завершено успешно")
                    self.root.after(0, lambda: messagebox.showinfo("Успех", "Модель дообучена"))
                    self.update_model_info()
                    # Обновляем модель на сервере
                    self.update_model()
                else:
                    self.status_var.set(f"Ошибка: {result.stderr[:200]}")
                    self.root.after(0, lambda: messagebox.showerror("Ошибка", result.stderr[:500]))
            except subprocess.TimeoutExpired:
                self.status_var.set("Превышено время ожидания")
            except Exception as e:
                self.status_var.set(f"Ошибка: {str(e)}")
                
        threading.Thread(target=_train, daemon=True).start()
        
    def manage_database(self):
        """Управление базой данных (открытие диалога)"""
        messagebox.showinfo("Управление БД", 
            "Для управления БД используйте кнопки:\n"
            "• Добавить - добавить новый образец\n"
            "• Удалить - удалить выбранный образец\n"
            "• Редактировать - изменить класс образца")
        
    def add_sample(self):
        """Добавление нового образца в БД"""
        # Открытие диалога выбора файла
        file_path = filedialog.askopenfilename(
            title="Выберите файл с данными для добавления",
            filetypes=[("CSV файлы", "*.csv"), ("JSON файлы", "*.json"), ("Numpy", "*.npy")]
        )
        
        if not file_path:
            return
            
        # Выбор класса
        class_window = tk.Toplevel(self.root)
        class_window.title("Выбор класса")
        class_window.geometry("350x300")
        class_window.configure(bg='#f0f0f0')
        
        tk.Label(class_window, text="Выберите класс неопределённости:", 
                 bg='#f0f0f0', font=('Arial', 12)).pack(pady=10)
        
        class_var = tk.IntVar()
        for class_id, class_name in CLASS_NAMES.items():
            rb = tk.Radiobutton(class_window, text=class_name, variable=class_var,
                                value=class_id, bg='#f0f0f0', font=('Arial', 10))
            rb.pack(anchor=tk.W, padx=20, pady=2)
        
        # Флаг для аугментации
        augment_var = tk.BooleanVar(value=False)
        augment_cb = tk.Checkbutton(class_window, text="Применить аугментацию (увеличит выборку)",
                                    variable=augment_var, bg='#f0f0f0', font=('Arial', 9))
        augment_cb.pack(pady=10)
        
        def confirm():
            class_window.destroy()
            self._add_sample_file(file_path, class_var.get(), augment_var.get())
            
        tk.Button(class_window, text="Добавить", command=confirm,
                  bg='#2ecc71', fg='white', font=('Arial', 12)).pack(pady=20)
    
    def _add_sample_file(self, file_path, class_id, augment=False):
        """Добавление файла в БД"""
        try:
            # Загрузка существующей БД
            if os.path.exists(Config.DATA_PATH):
                df = pd.read_csv(Config.DATA_PATH)
            else:
                # ИСПРАВЛЕНО: создаём DataFrame с правильными колонками
                df = pd.DataFrame()
            
            # Загрузка нового образца
            if file_path.endswith('.csv'):
                new_df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                # ИСПРАВЛЕНО: обработка списка
                if isinstance(data, list):
                    new_df = pd.DataFrame(data)
                else:
                    new_df = pd.DataFrame([data])
            elif file_path.endswith('.npy'):
                data = np.load(file_path)
                if len(data.shape) == 2:
                    new_df = pd.DataFrame(data.reshape(1, -1))
                else:
                    new_df = pd.DataFrame(data.reshape(1, -1))
            else:
                messagebox.showerror("Ошибка", "Неподдерживаемый формат")
                return
            
            # Добавление метки
            new_df['label'] = class_id
            
            # Применение аугментации
            if augment and len(new_df) > 0:
                augmented_rows = []
                for _ in range(3):  # 3 аугментации на образец
                    aug_row = new_df.iloc[0].copy()
                    for col in aug_row.index:
                        if col != 'label' and isinstance(aug_row[col], (int, float)):
                            noise = np.random.normal(0, 0.05 * abs(aug_row[col]))
                            aug_row[col] = aug_row[col] + noise
                    augmented_rows.append(aug_row)
                aug_df = pd.DataFrame(augmented_rows)
                new_df = pd.concat([new_df, aug_df], ignore_index=True)
            
            # Объединение
            df = pd.concat([df, new_df], ignore_index=True)
            
            # Сохранение
            os.makedirs(os.path.dirname(Config.DATA_PATH), exist_ok=True)
            df.to_csv(Config.DATA_PATH, index=False)
            
            self.status_var.set(f"Образец добавлен (класс {CLASS_NAMES[class_id]})")
            self.load_database_info()
            messagebox.showinfo("Успех", f"Образец добавлен в БД\nДобавлено {len(new_df)} записей")
            
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            
    def delete_sample(self):
        """Удаление выбранного образца"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Предупреждение", "Выберите класс для удаления")
            return
            
        item = self.tree.item(selected[0])
        class_id = item['values'][0]
        class_name = item['values'][1]
        
        result = messagebox.askyesno(
            "Подтверждение",
            f"Удалить все образцы класса '{class_name}'?\nЭто действие нельзя отменить."
        )
        
        if not result:
            return
            
        try:
            df = pd.read_csv(Config.DATA_PATH)
            df = df[df['label'] != class_id]
            df.to_csv(Config.DATA_PATH, index=False)
            
            self.status_var.set(f"Удалены образцы класса {class_name}")
            self.load_database_info()
            messagebox.showinfo("Успех", f"Образцы класса '{class_name}' удалены")
            
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            
    def edit_sample(self):
        """Редактирование класса образца"""
        messagebox.showinfo("Редактирование", 
            "Для редактирования откройте CSV файл в любом редакторе\n"
            f"Путь: {Config.DATA_PATH}\n\n"
            "Столбец 'label' содержит метку класса (0-4)")
            
    def configure_thresholds(self):
        """Настройка пороговых значений"""
        # Создание диалогового окна
        threshold_window = tk.Toplevel(self.root)
        threshold_window.title("Настройка порогов")
        threshold_window.geometry("400x350")
        threshold_window.configure(bg='#f0f0f0')
        
        tk.Label(threshold_window, text="Настройка пороговых значений", 
                 font=('Arial', 14, 'bold'), bg='#f0f0f0').pack(pady=10)
        
        tk.Label(threshold_window, text="Пороги уверенности для каждого класса", 
                 font=('Arial', 10), bg='#f0f0f0').pack(pady=5)
        
        # Поля для ввода порогов
        thresholds = {}
        for class_id, class_name in CLASS_NAMES.items():
            frame = tk.Frame(threshold_window, bg='#f0f0f0')
            frame.pack(fill=tk.X, padx=20, pady=5)
            
            tk.Label(frame, text=f"{class_name}:", width=20, anchor='w',
                     bg='#f0f0f0', font=('Arial', 10)).pack(side=tk.LEFT)
            
            var = tk.DoubleVar(value=0.5)
            thresholds[class_id] = var
            scale = tk.Scale(frame, from_=0.0, to=1.0, resolution=0.01,
                             orient=tk.HORIZONTAL, variable=var,
                             length=150)
            scale.pack(side=tk.LEFT, padx=5)
            
            value_label = tk.Label(frame, textvariable=var, width=5,
                                   bg='#f0f0f0', font=('Arial', 10))
            value_label.pack(side=tk.LEFT, padx=5)
        
        # Загрузка сохранённых порогов
        if os.path.exists('thresholds.json'):
            try:
                with open('thresholds.json', 'r') as f:
                    saved = json.load(f)
                for k, v in saved.items():
                    if int(k) in thresholds:
                        thresholds[int(k)].set(v)
            except:
                pass
        
        def save_thresholds():
            # Сохранение порогов в файл
            thresholds_dict = {str(k): v.get() for k, v in thresholds.items()}
            with open('thresholds.json', 'w') as f:
                json.dump(thresholds_dict, f, indent=2)
            messagebox.showinfo("Успех", "Пороги сохранены")
            threshold_window.destroy()
        
        button_frame = tk.Frame(threshold_window, bg='#f0f0f0')
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Сохранить", command=save_thresholds,
                  bg='#2ecc71', fg='white', font=('Arial', 12), width=15).pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, text="Отмена", command=threshold_window.destroy,
                  bg='#e74c3c', fg='white', font=('Arial', 12), width=15).pack(side=tk.LEFT, padx=10)
        
    def export_data(self):
        """Экспорт базы данных"""
        if not os.path.exists(Config.DATA_PATH):
            messagebox.showerror("Ошибка", "База данных не найдена")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Сохранить БД как",
            defaultextension=".csv",
            filetypes=[("CSV файлы", "*.csv"), ("Excel", "*.xlsx")]
        )
        
        if not file_path:
            return
            
        try:
            import shutil
            shutil.copy(Config.DATA_PATH, file_path)
            self.status_var.set(f"Экспорт завершён: {file_path}")
            messagebox.showinfo("Успех", f"Данные экспортированы в {file_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            
    def import_data(self):
        """Импорт базы данных"""
        file_path = filedialog.askopenfilename(
            title="Выберите файл для импорта",
            filetypes=[("CSV файлы", "*.csv"), ("Excel", "*.xlsx")]
        )
        
        if not file_path:
            return
            
        result = messagebox.askyesno(
            "Подтверждение",
            "Импорт заменит существующую базу данных.\nПродолжить?"
        )
        
        if not result:
            return
            
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
                
            df.to_csv(Config.DATA_PATH, index=False)
            self.status_var.set(f"Импорт завершён: {file_path}")
            self.load_database_info()
            messagebox.showinfo("Успех", "База данных обновлена")
            
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            
    def show_metrics(self):
        """Отображение метрик модели"""
        log_file = f"{Config.LOG_PATH}/training_history.csv"
        
        if not os.path.exists(log_file) or not os.path.exists(Config.LOG_PATH):
            messagebox.showinfo("Информация", 
                "История обучения не найдена. Запустите обучение: python train.py")
            return
        
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showerror("Ошибка", 
                "Для отображения графиков установите matplotlib:\n"
                "pip install matplotlib")
            return
        
        # Создание окна с графиками
        metrics_window = tk.Toplevel(self.root)
        metrics_window.title("Метрики модели")
        metrics_window.geometry("800x600")
        
        try:
            df = pd.read_csv(log_file)
            
            fig = Figure(figsize=(10, 8))
            
            # График потерь
            ax1 = fig.add_subplot(211)
            if 'train_loss' in df.columns:
                ax1.plot(df['train_loss'], label='Train Loss', color='#e74c3c', linewidth=2)
            if 'val_loss' in df.columns:
                ax1.plot(df['val_loss'], label='Val Loss', color='#3498db', linewidth=2)
            ax1.set_xlabel('Эпоха', fontsize=10)
            ax1.set_ylabel('Loss', fontsize=10)
            ax1.set_title('Функция потерь (формула 2.8)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # График точности
            ax2 = fig.add_subplot(212)
            if 'train_acc' in df.columns:
                ax2.plot(df['train_acc'], label='Train Accuracy', color='#e74c3c', linewidth=2)
            if 'val_acc' in df.columns:
                ax2.plot(df['val_acc'], label='Val Accuracy', color='#3498db', linewidth=2)
            ax2.set_xlabel('Эпоха', fontsize=10)
            ax2.set_ylabel('Accuracy (%)', fontsize=10)
            ax2.set_title('Точность классификации', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            canvas = FigureCanvasTkAgg(fig, master=metrics_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Информация о лучших значениях
            best_val_acc = df['val_acc'].max() if 'val_acc' in df.columns else 0
            best_val_loss = df['val_loss'].min() if 'val_loss' in df.columns else 0
            
            info_frame = tk.Frame(metrics_window, bg='#f0f0f0')
            info_frame.pack(fill=tk.X, padx=10, pady=5)
            
            tk.Label(info_frame, text=f"Лучшая точность валидации: {best_val_acc:.2f}%",
                     font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='#2ecc71').pack()
            tk.Label(info_frame, text=f"Лучшая потеря валидации: {best_val_loss:.4f}",
                     font=('Arial', 10), bg='#f0f0f0').pack()
                     
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            
    def restart_server(self):
        """Перезапуск сервера (без зависимости от psutil)"""
        result = messagebox.askyesno(
            "Подтверждение",
            "Перезапуск сервера займёт несколько секунд.\nПродолжить?"
        )
        
        if not result:
            return
            
        self.status_var.set("Перезапуск сервера...")
        
        def _restart():
            try:
                import os
                import time
                
                # ИСПРАВЛЕНО: без psutil
                if os.name == 'nt':  # Windows
                    # Закрываем окно сервера по заголовку (более щадящий способ)
                    os.system('taskkill /f /im python.exe 2>nul')
                else:  # Linux/Mac
                    os.system('pkill -f "python.*server.py" 2>/dev/null')
                
                # Небольшая задержка перед запуском
                time.sleep(2)
                
                # Запуск нового сервера
                if os.name == 'nt':  # Windows
                    subprocess.Popen(['python', 'server.py'], 
                                     creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:  # Linux/Mac
                    subprocess.Popen(['python', 'server.py'])
                
                self.status_var.set("Сервер перезапущен")
                # Задержка перед проверкой соединения
                self.root.after(3000, self.check_server_connection)
                messagebox.showinfo("Успех", "Сервер перезапущен")
                
            except Exception as e:
                self.status_var.set(f"Ошибка: {str(e)}")
                messagebox.showerror("Ошибка", str(e))
                
        threading.Thread(target=_restart, daemon=True).start()


def main():
    root = tk.Tk()
    app = AdminInterface(root)
    root.mainloop()


if __name__ == "__main__":
    main()