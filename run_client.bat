@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   ЗАПУСК КЛИЕНТА ОПЕРАТОРА
echo   Нейросетевой модуль классификации неопределённостей РТК
echo ============================================================
echo.

:: Проверка наличия Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ОШИБКА] Python не найден.
    echo Установите Python 3.9 или выше.
    echo.
    pause
    exit /b 1
)

:: Активация виртуального окружения (если существует)
if exist venv\Scripts\activate (
    call venv\Scripts\activate
    echo [ИНФО] Виртуальное окружение активировано
)

:: Проверка доступности сервера
echo [ИНФО] Проверка доступности сервера...
python -c "import requests; requests.get('http://localhost:8000/health', timeout=3)" >nul 2>nul
if %errorlevel% neq 0 (
    echo [ПРЕДУПРЕЖДЕНИЕ] Сервер не отвечает на http://localhost:8000
    echo Убедитесь, что запущен run_server.bat в отдельном окне
    echo.
    echo [ИНФО] Клиент будет запущен, но может работать некорректно
    echo.
    timeout /t 2 /nobreak >nul
) else (
    echo [ИНФО] Сервер доступен
    echo.
)

echo ============================================================
echo.

:: Запуск клиента
python client.py

:: Если клиент завершился с ошибкой
if %errorlevel% neq 0 (
    echo.
    echo [ОШИБКА] Клиент завершился с кодом ошибки %errorlevel%
    echo.
)

pause