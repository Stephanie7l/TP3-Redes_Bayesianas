@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "TP3_PYTHON="
py -3.13 --version >nul 2>&1
if not errorlevel 1 set "TP3_PYTHON=py -3.13"
if not defined TP3_PYTHON (
    python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,13) else 1)" >nul 2>&1
    if not errorlevel 1 set "TP3_PYTHON=python"
)
if not defined TP3_PYTHON (
    echo Instale o Python 3.13 de 64 bits e execute este arquivo novamente.
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,13) else 1)" >nul 2>&1
    if errorlevel 1 (
        echo O ambiente .venv existente usa outra versao do Python.
        echo Renomeie a pasta .venv e execute instalar.bat novamente.
        pause
        exit /b 1
    )
)
if not exist ".venv\Scripts\python.exe" (
    %TP3_PYTHON% -m venv .venv
    if errorlevel 1 goto erro
)
echo Instalando dependencias. A primeira instalacao pode demorar.
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto erro
".venv\Scripts\python.exe" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto erro
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto erro
echo.
echo Instalacao concluida. Agora abra executar.bat.
pause
exit /b 0
:erro
echo.
echo A instalacao nao terminou. Confira a mensagem acima e a conexao com a internet.
pause
exit /b 1
