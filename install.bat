@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo   УСТАНОВКА ЗАВИСИМОСТЕЙ ДЛЯ PATCHTST
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

:: Проверка наличия pip
where pip >nul 2>nul
if %errorlevel% neq 0 (
    echo [ОШИБКА] pip не найден.
    echo Обновите Python или установите pip вручную.
    echo.
    pause
    exit /b 1
)

echo [ИНФО] Python найден:
python --version
echo.

:: Обновление pip
echo [1/4] Обновление pip...
python -m pip install --upgrade pip
echo.

:: Установка PyTorch для RTX 5060 Ti (CUDA 12.8)
echo [2/4] Установка PyTorch с поддержкой CUDA (для RTX 5060 Ti)...
echo       Размер ~2.8 ГБ, пожалуйста, подождите...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось установить PyTorch.
    echo Попробуйте позже или проверьте интернет-соединение.
    pause
    exit /b 1
)
echo.

:: Установка остальных зависимостей через зеркало Tsinghua
echo [3/4] Установка остальных зависимостей (зеркало Tsinghua)...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось установить зависимости.
    echo Попробуйте запустить скрипт ещё раз.
    pause
    exit /b 1
)
echo.

:: Проверка установки GPU
echo [4/4] Проверка установки...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)} if torch.cuda.is_available() else None')" 2>nul
if %errorlevel% equ 0 (
    echo [УСПЕХ] Все зависимости установлены!
    echo.
    echo GPU готов к работе ^(RTX 5060 Ti^)
) else (
    echo [ПРЕДУПРЕЖДЕНИЕ] Проверка GPU не удалась, но библиотеки установлены.
    echo Модель будет работать на CPU.
)
echo.

echo ============================================================
echo   УСТАНОВКА ЗАВЕРШЕНА
echo ============================================================
echo.
echo Следующие шаги:
echo   1. Запустите: python synthetic_data.py  (генерация данных)
echo   2. Запустите: python train.py            (обучение модели)
echo   3. Запустите: python server.py           (запуск сервера)
echo   4. Запустите: python client.py           (клиент оператора)
echo.

pause