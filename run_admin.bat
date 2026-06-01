@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   ЗАПУСК ПАНЕЛИ АДМИНИСТРАТОРА
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
    echo [ИНФО] Панель администратора будет запущена, но функции обновления модели
    echo [ИНФО] и управления порогами могут быть недоступны.
    echo.
    timeout /t 2 /nobreak >nul
) else (
    echo [ИНФО] Сервер доступен
    echo.
)

:: Проверка наличия обученной модели
if not exist models\patchtst_model.pth (
    echo [ПРЕДУПРЕЖДЕНИЕ] Обученная модель не найдена.
    echo Выполните: python train.py
    echo.
)

:: Проверка наличия базы данных
if not exist data\synthetic_data.csv (
    echo [ПРЕДУПРЕЖДЕНИЕ] База данных не найдена.
    echo Будет создана новая база при первом добавлении образцов.
    echo.
)

echo ============================================================
echo.

:: Запуск панели администратора
python admin.py

:: Если администратор завершился с ошибкой
if %errorlevel% neq 0 (
    echo.
    echo [ОШИБКА] Панель администратора завершилась с кодом ошибки %errorlevel%
    echo.
)

pause