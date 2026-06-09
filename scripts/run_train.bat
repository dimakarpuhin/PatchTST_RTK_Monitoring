@echo off
echo ============================================================
echo   ОБУЧЕНИЕ МОДЕЛИ PATCHTST
echo   Специальность 2.3.5
echo ============================================================
echo.

:: Проверка наличия Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ОШИБКА] Python не найден.
    pause
    exit /b 1
)

:: Активация виртуального окружения
if exist venv\Scripts\activate (
    call venv\Scripts\activate
    echo [ИНФО] Виртуальное окружение активировано
)

:: Проверка установки зависимостей
python -c "import torch" >nul 2>nul
if %errorlevel% neq 0 (
    echo [ОШИБКА] Зависимости не установлены.
    echo Выполните: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo [ИНФО] Запуск обучения...
echo [ИНФО] Процесс может занять несколько минут.
echo.

python train.py

echo.
echo [ИНФО] Обучение завершено.
pause