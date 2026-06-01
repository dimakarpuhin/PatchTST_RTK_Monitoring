# Нейросетевой модуль классификации неопределённостей РТК


---

## Описание

Программная реализация модифицированной архитектуры PatchTST для классификации неопределённостей (напряжение, ток, температура, электромагнитные помехи) радиотехнического комплекса (РТК).

**Научная новизна (глава 2):**

- **Адаптивное позиционное кодирование** с учётом локальной дисперсии (формула 2.6):
  $$E_{\text{pos}}^{(l)}(k) = PE(k) \cdot \alpha^{(l)} \cdot \sigma(P_k)$$

- **Межканальный механизм внимания** (формула 2.7):
  $$Z_{\text{channel}} = Z \cdot \text{softmax}\left(\frac{Z^T Z}{\sqrt{d}}\right)$$

- **Контрастивная регуляризация** для few-shot обучения (формула 2.8):
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_1\|W\|_2^2 + \lambda_2 \mathcal{L}_{\text{contrast}}$$

---

## Структура проекта
project/
│
├── config.py # Конфигурация модели
├── model.py # Модифицированный PatchTST
├── synthetic_data.py # Генератор синтетических данных
├── data_loader.py # Загрузка и предобработка
├── train.py # Обучение модели
├── server.py # FastAPI сервер
├── client.py # GUI клиент оператора
├── admin.py # GUI администратора
│
├── requirements.txt # Зависимости
├── run_server.bat # Запуск сервера (Windows)
├── run_client.bat # Запуск клиента (Windows)
├── run_admin.bat # Запуск администратора (Windows)
├── run_all.bat # Запуск всех компонентов
├── run_train.bat # Обучение модели
│
├── models/ # Директория для сохранения моделей
├── data/ # Директория для данных
├── logs/ # Директория для логов
│
└── README.md # Документация


---

## Быстрый старт

### 1. Установка Python

Требуется **Python 3.9 или выше**. Скачайте с [python.org](https://www.python.org/downloads/).

### 2. Установка зависимостей

```bash
pip install -r requirements.txt

Для Windows с GPU (NVIDIA CUDA):

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

3. Генерация синтетических данных

python synthetic_data.py

4. Обучение модели

python train.py

5. Запуск сервера

python server.py

Сервер будет доступен на http://localhost:8000

Документация API: http://localhost:8000/docs

6. Запуск клиента (в другом окне терминала)

python client.py

7. Запуск панели администратора

python admin.py

Альтернативный запуск (Windows)

Используйте bat-файлы для быстрого запуска:

    run_server.bat — запуск сервера

    run_client.bat — запуск клиента оператора

    run_admin.bat — запуск панели администратора

    run_all.bat — запуск всех компонентов

    run_train.bat — обучение модели


    Конфигурация

Основные параметры задаются в config.py:
Параметр	Значение по умолчанию	Описание
WINDOW_LENGTH	512	Длина временного окна T (отсчётов)
NUM_CHANNELS	4	Число параметров d
NUM_CLASSES	5	Число классов C
PATCH_LEN	32	Длина патча L
PATCH_STRIDE	16	Шаг патча S
D_MODEL	64	Размерность эмбеддинга d_model
NUM_HEADS	4	Число голов внимания H
NUM_LAYERS	3	Число слоёв трансформера L
DROPOUT	0.1	Вероятность dropout
MASK_PROB	0.15	Вероятность маскирования патчей
LAMBDA_1	1e-4	Коэффициент L2-регуляризации λ₁
LAMBDA_2	0.1	Коэффициент контрастивной потери λ₂
TEMPERATURE	0.07	Температура для контрастивной потери τ


API Endpoints (сервер)
Endpoint	Метод	Описание
/health	GET	Проверка доступности сервера
/classes	GET	Получение списка классов
/classify	POST	Классификация одного временного окна
/classify_batch	POST	Пакетная классификация
/model/update	POST	Обновление весов модели


Пример запроса /classify

Вход (JSON):
{
  "window": [
    [220.0, 10.2, 50.0, 35.0],
    [221.0, 10.3, 50.1, 34.8],
    ...
  ]
}

Выход (JSON):

{
  "class_id": 2,
  "class_name": "Скачок напряжения",
  "probabilities": [0.02, 0.94, 0.01, 0.02, 0.01],
  "latency_ms": 18.2,
  "timestamp": "2026-01-15T10:30:00"
}

Результаты (глава 3)
Метрика	Значение
Macro-F1	0.932
ROC-AUC	0.956
Время инференса (CPU, Intel Core i5)	18.0 мс
Время инференса (ARM Cortex-A9)	28.0 мс (T=1024) / 9.0 мс (T=256)
Время реакции (оптимизированный режим)	64 мс
Потребление памяти	~340 МБ
Точность при SNR = 10 дБ	0.848 (Macro-F1)


Требования к системе

    Операционная система: Windows 10/11, Linux (Ubuntu 20.04+), macOS 11+

    Python: 3.9 или выше

    ОЗУ: 512 МБ (минимум), рекомендуется 4 ГБ

    Процессор: x86_64 или ARM64 (поддержка CPU-only)

    GPU (опционально): NVIDIA с CUDA 11.8+ (например, RTX 4060 Ti)




Устранение неполадок
Ошибка: "ModuleNotFoundError: No module named 'torch'"

Решение: Установите зависимости:
 pip install -r requirements.txt

Ошибка: "CUDA not available"

Решение: Установите PyTorch с поддержкой CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118



Ошибка: "Port 8000 already in use"

Решение: Закройте другой процесс, использующий порт 8000, или измените порт в server.py.
Сервер не отвечает на запросы

Решение: Проверьте, что сервер запущен, и выполните:

curl http://localhost:8000/health


Ссылки

    Документация PyTorch

    Документация FastAPI

    Оригинальная статья PatchTST (Nie et al., 2023)


Лицензия

Данное программное обеспечение разработано в рамках диссертационного исследования по специальности 2.3.5.
Контакты

По вопросам, связанным с программной реализацией, обращайтесь к автору диссертации.
Благодарности

Разработка выполнена с использованием библиотек PyTorch, FastAPI, scikit-learn, matplotlib и других open-source проектов.

© 2026 | Диссертация по специальности 2.3.5


