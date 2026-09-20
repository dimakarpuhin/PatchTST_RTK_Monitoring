# database.py
# Модуль для работы с SQLite базой данных

import sqlite3
import pandas as pd
import numpy as np
from config import Config
import os

DB_PATH = 'data/rtk_monitoring.db'

def get_connection():
    """Получение соединения с БД"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    """Создание таблиц при первом запуске"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Таблица для временных окон
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS windows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT,
            window_data TEXT,  -- JSON или список значений
            label INTEGER,
            created_at TEXT
        )
    ''')
    
    # Таблица для результатов классификации
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_id INTEGER,
            class_id INTEGER,
            class_name TEXT,
            probabilities TEXT,  -- JSON
            latency_ms REAL,
            timestamp TEXT,
            FOREIGN KEY (window_id) REFERENCES windows(id)
        )
    ''')
    
    # Таблица для моделей
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT,
            path TEXT,
            metrics TEXT,  -- JSON (accuracy, f1, etc.)
            created_at TEXT
        )
    ''')
    
    # Таблица для конфигурации
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ База данных инициализирована")

def add_window(session_id, timestamp, window_data, label):
    """Добавление временного окна"""
    conn = get_connection()
    cursor = conn.cursor()
    
    import json
    window_json = json.dumps(window_data.tolist() if isinstance(window_data, np.ndarray) else window_data)
    
    cursor.execute('''
        INSERT INTO windows (session_id, timestamp, window_data, label, created_at)
        VALUES (?, ?, ?, ?, datetime('now'))
    ''', (session_id, timestamp, window_json, label))
    
    window_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return window_id

def add_result(window_id, class_id, class_name, probabilities, latency_ms):
    """Добавление результата классификации"""
    conn = get_connection()
    cursor = conn.cursor()
    
    import json
    probs_json = json.dumps(probabilities)
    
    cursor.execute('''
        INSERT INTO results (window_id, class_id, class_name, probabilities, latency_ms, timestamp)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
    ''', (window_id, class_id, class_name, probs_json, latency_ms))
    
    result_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return result_id

def get_all_windows():
    """Получение всех окон"""
    conn = get_connection()
    df = pd.read_sql_query('SELECT * FROM windows ORDER BY id DESC', conn)
    conn.close()
    return df

def get_all_results():
    """Получение всех результатов"""
    conn = get_connection()
    df = pd.read_sql_query('''
        SELECT r.*, w.session_id, w.label as true_label
        FROM results r
        JOIN windows w ON r.window_id = w.id
        ORDER BY r.id DESC
    ''', conn)
    conn.close()
    return df

def delete_window(window_id):
    """Удаление окна и связанных результатов"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Удаляем связанные результаты
    cursor.execute('DELETE FROM results WHERE window_id = ?', (window_id,))
    # Удаляем окно
    cursor.execute('DELETE FROM windows WHERE id = ?', (window_id,))
    
    conn.commit()
    conn.close()

def get_stats():
    """Получение статистики по БД"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM windows')
    windows_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM results')
    results_count = cursor.fetchone()[0]
    
    conn.close()
    return {
        'windows': windows_count,
        'results': results_count
    }

def export_to_csv():
    """Экспорт всей БД в CSV (для совместимости)"""
    windows_df = get_all_windows()
    results_df = get_all_results()
    
    os.makedirs('data/export', exist_ok=True)
    windows_df.to_csv('data/export/windows_export.csv', index=False)
    results_df.to_csv('data/export/results_export.csv', index=False)
    print("✅ Данные экспортированы в data/export/")