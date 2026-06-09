@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   ЗАПУСК НЕЙРОСЕТЕВОГО СЕРВЕРА
echo   Специальность 2.3.5 - Математическое и программное обеспечение
echo ============================================================
echo.

:: Проверка наличия Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ОШИБКА] Python не найден.
    echo Установите Python 3.9 или выше с официального сайта.
    echo.
    pause
    exit /b 1
)

:: Проверка версии Python
python -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" >nul 2>nul
if %errorlevel% neq 0 (
    echo [ОШИБКА] Требуется Python 3.9 или выше.
    echo Текущая версия:
    python --version
    echo.
    pause
    exit /b 1
)

:: Активация виртуального окружения (если существует)
if exist venv\Scripts\activate (
    call venv\Scripts\activate
    echo [ИНФО] Виртуальное окружение активировано
)

:: Проверка установки зависимостей
python -c "import torch" >nul 2>nul
if %errorlevel% neq 0 (
    echo [ПРЕДУПРЕЖДЕНИЕ] Зависимости не установлены.
    echo Выполните: pip install -r requirements.txt
    echo.
)

:: Проверка, что порт 8000 не занят
netstat -an | find "8000" >nul 2>nul
if %errorlevel% equ 0 (
    echo [ОШИБКА] Порт 8000 уже занят.
    echo Возможно, сервер уже запущен в другом окне.
    echo.
    pause
    exit /b 1
)

echo.
echo [ИНФО] Запуск сервера на http://localhost:8000
echo [ИНФО] Документация API: http://localhost:8000/docs
echo.
echo [ИНФО] Для остановки нажмите Ctrl+C
echo.
echo ============================================================
echo.

:: Запуск сервера
python server.py

:: Если сервер завершился с ошибкой
if %errorlevel% neq 0 (
    echo.
    echo [ОШИБКА] Сервер завершился с кодом ошибки %errorlevel%
    echo.
)

pause