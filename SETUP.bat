@echo off
chcp 65001 >nul
echo =======================================================
echo   DermAI - Skin Disease Detection Setup
echo =======================================================
echo.
echo [1/3] Verifying and installing Python dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [!] Pip install encountered a warning, continuing...
)

echo.
echo [2/3] Generating synthetic dataset (1050 images across 7 classes)...
python prepare_dataset.py
if %errorlevel% neq 0 (
    echo [ERROR] Dataset generation failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Training AI/ML Models (Neural CNN + SVM Ensemble)...
python train_model.py
if %errorlevel% neq 0 (
    echo [ERROR] Model training failed.
    pause
    exit /b 1
)

echo.
echo =======================================================
echo   SETUP COMPLETED SUCCESSFULLY!
echo   Run START.bat to launch the application.
echo =======================================================
pause
