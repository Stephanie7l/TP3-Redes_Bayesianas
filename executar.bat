@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Execute instalar.bat antes de executar a analise.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,13) else 1)" >nul 2>&1
if errorlevel 1 (
    echo O ambiente virtual nao usa Python 3.13. Execute instalar.bat para conferir.
    pause
    exit /b 1
)
set "PYTHONHASHSEED=42"
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
".venv\Scripts\python.exe" tp3.py
if errorlevel 1 (
    echo.
    echo A analise nao terminou. Confira a mensagem de erro acima.
    pause
    exit /b 1
)
echo.
echo Analise concluida. Consulte a pasta resultados.
start "" "%~dp0resultados"
pause
