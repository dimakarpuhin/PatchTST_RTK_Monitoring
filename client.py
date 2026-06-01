# client.py
# GUI клиент оператора для взаимодействия с нейросетевым сервером

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import requests
import json
import numpy as np
import pandas as pd
import threading
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from config import Config


# Конфигурация подключения к серверу
SERVER_URL = "http://localhost:8000"
API_CLASSIFY = f"{SERVER_URL}/classify"
API_HEALTH = f"{SERVER_URL}/health"
API_CLASSES = f"{SERVER_URL}/classes"

# Классы неопределённостей (будут загружены с сервера)
CLASS_NAMES = {
    0: "Норма",
    1: "Скачок напряжения",
    2: "Токовая перегрузка",
    3: "Перегрев",
    4: "Электропомеха"
}

# Цвета для классов
CLASS_COLORS = {
    0: "#2ecc71",  # зелёный - норма
    1: "#e74c3c",  # красный - скачок напряжения
    2: "#e67e22",  # оранжевый - перегрузка
    3: "#f39c12",  # жёлтый - перегрев
    4: "#9b59b6"   # фиолетовый - помехи
}


class OperatorClient:
    """Графический интерфейс оператора"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Нейросетевой модуль классификации - Оператор")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')
        
        # Переменные для хранения данных
        self.current_data = None
        self.current_window = None
        self.server_available = False
        self.running = False
        self.simulation_thread = None
        
        # Иконка статуса сервера
        self.server_status_var = tk.StringVar(value="Проверка...")
        
        self.setup_ui()
        self.check_server_connection()
        self.fetch_classes()
        
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        
        # Верхняя панель с кнопками
        top_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        top_frame.pack(fill=tk.X)
        top_frame.pack_propagate(False)
        
        # Заголовок
        title_label = tk.Label(
            top_frame, 
            text="Нейросетевой модуль классификации неопределённостей РТК",
            font=('Arial', 16, 'bold'),
            fg='white', bg='#2c3e50'
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Статус сервера
        status_frame = tk.Frame(top_frame, bg='#2c3e50')
        status_frame.pack(side=tk.RIGHT, padx=20, pady=15)
        
        status_label = tk.Label(
            status_frame,
            text="Сервер:",
            font=('Arial', 10),
            fg='white', bg='#2c3e50'
        )
        status_label.pack(side=tk.LEFT)
        
        self.status_indicator = tk.Label(
            status_frame,
            textvariable=self.server_status_var,
            font=('Arial', 10, 'bold'),
            fg='white', bg='#2c3e50'
        )
        self.status_indicator.pack(side=tk.LEFT, padx=5)
        
        # Основной контент
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - график
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # График временного ряда
        plot_frame = tk.LabelFrame(left_frame, text="Временной ряд параметров", 
                                    font=('Arial', 12, 'bold'), bg='#f0f0f0')
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Правая панель - результаты и управление
        right_frame = tk.Frame(main_frame, bg='#f0f0f0', width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        right_frame.pack_propagate(False)
        
        # Панель загрузки файла
        load_frame = tk.LabelFrame(right_frame, text="Загрузка данных", 
                                    font=('Arial', 12, 'bold'), bg='#f0f0f0')
        load_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.file_path_var = tk.StringVar()
        file_entry = tk.Entry(load_frame, textvariable=self.file_path_var, width=30)
        file_entry.pack(side=tk.LEFT, padx=5, pady=10)
        
        browse_btn = tk.Button(load_frame, text="Обзор", command=self.browse_file,
                                bg='#3498db', fg='white')
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        load_btn = tk.Button(load_frame, text="Загрузить", command=self.load_data,
                             bg='#2ecc71', fg='white')
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # Панель результатов классификации
        result_frame = tk.LabelFrame(right_frame, text="Результат классификации",
                                      font=('Arial', 12, 'bold'), bg='#f0f0f0')
        result_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Текущий класс
        class_label = tk.Label(result_frame, text="Текущий класс:", 
                                font=('Arial', 12), bg='#f0f0f0')
        class_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        
        self.class_value = tk.Label(result_frame, text="—", 
                                     font=('Arial', 20, 'bold'), bg='#f0f0f0')
        self.class_value.grid(row=0, column=1, padx=10, pady=10)
        
        # Вероятности
        prob_label = tk.Label(result_frame, text="Вероятности:", 
                               font=('Arial', 12), bg='#f0f0f0')
        prob_label.grid(row=1, column=0, padx=10, pady=5, sticky='w')
        
        self.prob_text = tk.Text(result_frame, height=8, width=30, font=('Arial', 10))
        self.prob_text.grid(row=2, column=0, columnspan=2, padx=10, pady=5)
        
        # Время обработки
        latency_label = tk.Label(result_frame, text="Время обработки:", 
                                  font=('Arial', 10), bg='#f0f0f0')
        latency_label.grid(row=3, column=0, padx=10, pady=5, sticky='w')
        
        self.latency_value = tk.Label(result_frame, text="—", 
                                       font=('Arial', 10), bg='#f0f0f0')
        self.latency_value.grid(row=3, column=1, padx=10, pady=5, sticky='w')
        
        # Кнопка классификации
        self.classify_btn = tk.Button(right_frame, text="Классифицировать", 
                                      command=self.classify_current,
                                      font=('Arial', 14, 'bold'),
                                      bg='#e74c3c', fg='white', height=2)
        self.classify_btn.pack(fill=tk.X, padx=5, pady=10)
        
        # Кнопка сохранения результата (ДОБАВЛЕНО)
        save_btn = tk.Button(right_frame, text="Сохранить результат", 
                             command=self.save_result,
                             font=('Arial', 12),
                             bg='#3498db', fg='white', height=1)
        save_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Панель логов
        log_frame = tk.LabelFrame(right_frame, text="Лог событий",
                                   font=('Arial', 12, 'bold'), bg='#f0f0f0')
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_frame, height=8, font=('Arial', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Скроллбар для лога
        scrollbar = tk.Scrollbar(self.log_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)
        
        # Переменная для хранения последнего результата
        self.last_result = None
        
    def browse_file(self):
        """Выбор файла для загрузки"""
        file_path = filedialog.askopenfilename(
            title="Выберите файл с данными",
            filetypes=[("CSV файлы", "*.csv"), ("JSON файлы", "*.json"), ("Все файлы", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
            
    def load_data(self):
        """Загрузка данных из файла"""
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showerror("Ошибка", "Выберите файл")
            return
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            
                # ИСПРАВЛЕНО: правильная обработка CSV формата
                if 'label' in df.columns:
                    labels = df['label'].values
                    df = df.drop('label', axis=1)
            
                # Получаем значения как массив
                data = df.values
            
                # Определяем размерность
                if len(data.shape) == 2:
                    # Проверяем, что это одно окно [window_len, num_channels]
                    if data.shape[0] == Config.WINDOW_LENGTH:
                        self.current_window = data
                    elif data.shape[1] > 100:
                        # Данные в формате [1, window_len * num_channels] - reshape
                        window_len = Config.WINDOW_LENGTH
                        num_channels = Config.NUM_CHANNELS
                        self.current_window = data[0].reshape(window_len, num_channels)
                    else:
                        self.current_window = data
                else:
                    self.current_window = data
                
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict) and 'data' in data:
                    self.current_window = np.array(data['data'])
                else:
                    self.current_window = np.array(data)
            else:
                messagebox.showerror("Ошибка", "Неподдерживаемый формат файла")
                return
            
            # Проверка размерности
            if len(self.current_window.shape) != 2:
                raise ValueError(f"Некорректная размерность данных: {self.current_window.shape}")
        
            if self.current_window.shape[0] != Config.WINDOW_LENGTH:
                self.add_log(f"Предупреждение: длина окна {self.current_window.shape[0]}, ожидается {Config.WINDOW_LENGTH}")
        
            self.plot_data(self.current_window)
            self.add_log(f"Загружен файл: {file_path}")
            self.add_log(f"Размер данных: {self.current_window.shape}")
        
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")
    

        
            
    def plot_data(self, data):
        """Отображение временного ряда на графике"""
        self.figure.clear()
        
        if len(data.shape) == 2:
            window_len, num_channels = data.shape
            channel_names = ['Напряжение (В)', 'Ток (А)', 'Температура (°C)', 'Уровень помех (дБ)']
            
            # ИСПРАВЛЕНО: динамическое создание подграфиков
            for i in range(min(num_channels, 4)):
                ax = self.figure.add_subplot(min(num_channels, 4), 1, i + 1)
                ax.plot(data[:, i], linewidth=1, color=CLASS_COLORS.get(i, '#3498db'))
                ax.set_ylabel(channel_names[i] if i < len(channel_names) else f'Канал {i}')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, window_len)
            
            if num_channels > 0:
                ax.set_xlabel('Время (отсчёты)')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def classify_current(self):
        """Отправка данных на классификацию"""
        if self.current_window is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return
            
        if not self.server_available:
            messagebox.showerror("Ошибка", "Сервер недоступен")
            return
            
        # Отправка запроса в отдельном потоке
        threading.Thread(target=self._classify_thread, daemon=True).start()
        
    def _classify_thread(self):
        """Поток для отправки запроса на сервер"""
        # ИСПРАВЛЕНО: блокировка кнопки
        self.root.after(0, lambda: self.classify_btn.config(state=tk.DISABLED, text="Классификация..."))
        
        try:
            # Подготовка данных
            window_list = self.current_window.tolist()
            
            # Отправка запроса
            response = requests.post(
                API_CLASSIFY,
                json={"window": window_list},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.last_result = result
                self.root.after(0, self.update_results, result)
            else:
                self.root.after(0, lambda: self.add_log(f"Ошибка сервера: {response.status_code}"))
                
        except requests.exceptions.ConnectionError:
            self.root.after(0, lambda: self.add_log("Ошибка: Не удалось подключиться к серверу"))
            self.root.after(0, self.check_server_connection)
        except Exception as e:
            self.root.after(0, lambda: self.add_log(f"Ошибка: {str(e)}"))
        finally:
            # ИСПРАВЛЕНО: разблокировка кнопки
            self.root.after(0, lambda: self.classify_btn.config(state=tk.NORMAL, text="Классифицировать"))
            
    def update_results(self, result):
        """Обновление интерфейса с результатами классификации"""
        class_id = result['class_id']
        class_name = result['class_name']
        probabilities = result['probabilities']
        latency = result['latency_ms']
        
        # Обновление метки класса
        self.class_value.config(text=class_name, fg=CLASS_COLORS.get(class_id, '#000000'))
        
        # Обновление вероятностей
        prob_text = ""
        for i, prob in enumerate(probabilities):
            name = CLASS_NAMES.get(i, f"Класс {i}")
            prob_text += f"{name}: {prob*100:.1f}%\n"
        self.prob_text.delete(1.0, tk.END)
        self.prob_text.insert(1.0, prob_text)
        
        # Обновление времени
        self.latency_value.config(text=f"{latency} мс")
        
        # Добавление в лог
        self.add_log(f"Классификация: {class_name} (вероятность {max(probabilities)*100:.1f}%, {latency} мс)")
        
    def save_result(self):
        """Сохранение результата классификации в файл"""
        if self.last_result is None:
            messagebox.showwarning("Предупреждение", "Нет результатов для сохранения")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".json",
            filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.last_result, f, ensure_ascii=False, indent=2)
                self.add_log(f"Результат сохранён: {file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить результат: {str(e)}")
        
    def fetch_classes(self):
        """Получение списка классов с сервера"""
        try:
            response = requests.get(API_CLASSES, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'classes' in data:
                    global CLASS_NAMES
                    CLASS_NAMES = data['classes']
                    self.add_log("Список классов загружен с сервера")
        except Exception as e:
            self.add_log(f"Не удалось загрузить список классов: {str(e)}")
            
    def check_server_connection(self):
        """Проверка доступности сервера"""
        try:
            response = requests.get(API_HEALTH, timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.server_available = True
                self.server_status_var.set("Доступен ✓")
                self.status_indicator.config(fg='#2ecc71')
                self.add_log("Сервер доступен")
                
                # Показ информации о модели
                config_info = data.get('config', {})
                self.add_log(f"Модель: PatchTST (окно {config_info.get('WINDOW_LENGTH', '?')})")
            else:
                self.server_available = False
                self.server_status_var.set("Ошибка ✗")
                self.status_indicator.config(fg='#e74c3c')
        except:
            self.server_available = False
            self.server_status_var.set("Недоступен ✗")
            self.status_indicator.config(fg='#e74c3c')
            
        # Повторная проверка через 5 секунд
        self.root.after(5000, self.check_server_connection)
        
    def add_log(self, message):
        """Добавление сообщения в лог"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        # Ограничение длины лога (последние 10000 символов)
        if len(self.log_text.get(1.0, tk.END)) > 10000:
            self.log_text.delete(1.0, 2.0)


def main():
    root = tk.Tk()
    app = OperatorClient(root)
    root.mainloop()


if __name__ == "__main__":
    main()