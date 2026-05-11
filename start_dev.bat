@echo off
:: start_dev.bat — Inicia el backend MedAI en Windows
:: Ejecutar desde la carpeta MedAI_Backend

cd /d "%~dp0"

if not exist venv (
    echo ERROR: No se encontro el entorno virtual.
    echo Ejecuta primero el setup ^(ver README.md^):
    echo   python -m venv venv
    echo   venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist models\triage_model.joblib (
    echo Modelo no encontrado. Entrenando ahora ^(~5 minutos^)...
    venv\Scripts\python train_model.py
    if errorlevel 1 (
        echo ERROR: Fallo el entrenamiento del modelo.
        pause
        exit /b 1
    )
)

echo.
echo Iniciando MedAI Backend en http://localhost:8000
echo Documentacion API: http://localhost:8000/docs
echo Presiona Ctrl+C para detener.
echo.
venv\Scripts\uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
