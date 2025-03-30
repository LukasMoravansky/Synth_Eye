@echo off
setlocal enabledelayedexpansion

:: ==============================
:: Conda Environment Setup Script
:: ==============================
:: Author: Roman Parak
:: ==============================
:: To run the installation script, it is necessary to use the command prompt as shown below:
::   1. Open the Command Prompt.
::   2. Navigate to the folder containing install.bat.
::   3. Run the script by typing:
::          C:\projects\{project_name}> .\install.bat
:: 
:: Description:
::  The script installs Miniconda (if necessary), sets up a Conda environment,
::  installs necessary Python packages, and verifies the installation.
:: ==============================

:: --- Set Path Variables ---
set ENV_NAME=env_synth_eye
set CONDA_PATH=%USERPROFILE%\Miniconda3
set CONDA_SCRIPTS=%CONDA_PATH%\Scripts
set CONDA_BIN=%CONDA_PATH%\Library\bin
set VERIFY_SCRIPT=%~dp0verify.py

:: --- Check if Miniconda is Installed ---
echo [%time%] [INFO] Checking if Miniconda is installed...
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [%time%] [ERROR] Miniconda not found. Please install it to proceed.
    exit /b 1
) else (
    echo [%time%] [INFO] Miniconda is already installed. Skipping installation.
)

:: --- Temporarily Add Conda to PATH ---
set "PATH=%CONDA_SCRIPTS%;%CONDA_BIN%;%CONDA_PATH%;%PATH%" >nul 2>nul
echo [%time%] [INFO] Conda PATH added.

:: --- Verify Conda ---
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [%time%] [ERROR] Conda is not recognized! Try restarting the Command Prompt.
    exit /b 1
)

:: --- Initialize Conda ---
call conda info --envs >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [%time%] [INFO] Conda not initialized. Running conda init...
    call conda init >nul 2>nul
    echo [%time%] [INFO] Conda initialization successful!
)

:: --- Check & Remove Existing Environment ---
echo [%time%] [INFO] Checking if environment "%ENV_NAME%" exists...
call conda env list | findstr /C:"%ENV_NAME%" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [%time%] [INFO] Removing existing environment "%ENV_NAME%"...
    call conda env remove --name %ENV_NAME% -y >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [%time%] [ERROR] Failed to remove existing environment "%ENV_NAME%"!
        exit /b 1
    )
    echo [%time%] [INFO] Environment "%ENV_NAME%" removed successfully.
)

:: --- Create Conda Environment ---
echo [%time%] [INFO] Creating Conda environment "%ENV_NAME%" with Python 3.9...
call conda create -y --name %ENV_NAME% python=3.9 >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [%time%] [ERROR] Failed to create environment "%ENV_NAME%"!
    exit /b 1
)
echo [%time%] [INFO] Environment %ENV_NAME% created successfully.

:: --- Find Python Path in New Environment ---
for /f "delims=" %%P in ('conda run -n %ENV_NAME% where python') do set CONDA_PYTHON=%%P
echo [%time%] [INFO] Python used: %CONDA_PYTHON%

:: --- Install Required Packages ---
echo [%time%] [INFO] Installing required Python packages...
call conda install -y -n %ENV_NAME% -c conda-forge numpy matplotlib pandas scipy opencv pyqt lxml albumentations labelImg sqlite pysqlite3 >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [%time%] [ERROR] Package installation failed!
    exit /b 1
)
echo [%time%] [INFO] Python packages installed successfully.

:: --- Install Additional Python Libraries via pip ---
for %%P in (ultralytics pypylon) do (
    echo [%time%] [INFO] Installing %%P using pip...
    call conda run -n %ENV_NAME% python -m pip install %%P >nul 2>nul
    if !ERRORLEVEL! NEQ 0 (
        echo [%time%] [ERROR] Failed to install %%P!
        exit /b 1
    )
    echo [%time%] [INFO] Package %%P installed successfully.
)

:: --- Install Jupyter and Extensions ---
echo [%time%] [INFO] Installing Jupyter and extensions...
call conda install -y -n %ENV_NAME% jupyterlab notebook ipywidgets widgetsnbextension ipykernel -c conda-forge >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [%time%] [ERROR] Jupyter and extensions installation failed!
    exit /b 1
)
echo [%time%] [INFO] Jupyter and extensions installed successfully.

:: --- Final Conda Update ---
echo [%time%] [INFO] Updating Conda packages...
call conda update --all -y -n %ENV_NAME% >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [%time%] [ERROR] Conda update failed!
    exit /b 1
)
echo [%time%] [INFO] Conda packages updated successfully.

:: --- Reinstall Required Packages ---
echo [%time%] [INFO] Reinstalling required packages to ensure proper setup...
call conda install -y -n %ENV_NAME% --force-reinstall sqlite >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [%time%] [ERROR] Package reinstallation failed!
    exit /b 1
)
echo [%time%] [INFO] Required packages reinstalled successfully.

:: --- Verify Package Installation ---
if exist "%VERIFY_SCRIPT%" (
    echo [%time%] [INFO] Running verification script: %VERIFY_SCRIPT%
    call conda run -n %ENV_NAME% python %VERIFY_SCRIPT%
    if %ERRORLEVEL% NEQ 0 (
        echo [%time%] [ERROR] Verification script execution failed!
        exit /b 1
    )
    echo [%time%] [INFO] Verification script executed successfully.
) else (
    echo [%time%] [WARNING] Verification script not found. Skipping verification.
)

:: --- Final Success Message ---
echo [%time%] [INFO] All packages installed successfully!
echo [%time%] [INFO] Conda environment "%ENV_NAME%" is ready!
exit /b 0