#!/bin/bash

# ==============================
# Conda Environment Setup Script
# ==============================
# Author: Roman Parak
# ==============================
# To use the installation script, it is necessary to run the following commands:
#   1. Open the terminal.
#   2. Navigate to the folder containing install.sh.
#   3. Make the script executable by typing:
#      user_name@hostname:~$ chmod +x install.sh
#   4. Run the script by typing:
#      user_name@hostname:~$ ./install.sh
#
# Description:
#  The script installs Miniconda (if necessary), sets up a Conda environment,
#  installs necessary Python packages, and verifies the installation.
# ==============================

# --- Set Path Variables ---
ENV_NAME="env_synth_eye"
CONDA_PATH="$HOME/miniconda3"
CONDA_SCRIPTS="$CONDA_PATH/bin"
CONDA_BIN="$CONDA_PATH/bin"
VERIFY_SCRIPT="$(dirname "$0")/verify.py"

# --- Check if Miniconda is Installed ---
echo "[$(date)] [INFO] Checking if Miniconda is installed..."
if ! command -v conda &> /dev/null; then
    echo "[$(date)] [ERROR] Miniconda not found. Please install it to proceed."
    exit 1
else
    echo "[$(date)] [INFO] Miniconda is already installed."
fi

# --- Temporarily Add Conda to PATH ---
export PATH="$CONDA_SCRIPTS:$CONDA_BIN:$CONDA_PATH:$PATH"
echo "[$(date)] [INFO] Conda PATH added."

# --- Verify Conda ---
if ! command -v conda &> /dev/null
then
    echo "[$(date)] [ERROR] Conda is not recognized! Try restarting the terminal."
    exit 1
fi

# --- Initialize Conda ---
conda info --envs &> /dev/null
if [ $? -ne 0 ]; then
    echo "[$(date)] [INFO] Conda not initialized. Running conda init..."
    conda init > /dev/null 2>&1
    echo "[$(date)] [INFO] Conda initialization successful!"
fi

# --- Check & Remove Existing Environment ---
echo "[$(date)] [INFO] Checking if environment \"$ENV_NAME\" exists..."
if conda env list | grep -q "$ENV_NAME"; then
    echo "[$(date)] [INFO] Removing existing environment \"$ENV_NAME\"..."
    conda env remove --name "$ENV_NAME" -y > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "[$(date)] [ERROR] Failed to remove existing environment \"$ENV_NAME\"!"
        exit 1
    fi
    echo "[$(date)] [INFO] Environment \"$ENV_NAME\" removed successfully."
fi

# --- Create Conda Environment ---
echo "[$(date)] [INFO] Creating Conda environment \"$ENV_NAME\" with Python 3.9..."
conda create -y --name "$ENV_NAME" python=3.9 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "[$(date)] [ERROR] Failed to create environment \"$ENV_NAME\"!"
    exit 1
fi
echo "[$(date)] [INFO] Environment \"$ENV_NAME\" created successfully."

# --- Find Python Path in New Environment ---
CONDA_PYTHON=$(conda run -n "$ENV_NAME" which python)
echo "[$(date)] [INFO] Python used: $CONDA_PYTHON"

# --- Install Required Packages ---
echo "[$(date)] [INFO] Installing required Python packages..."
conda install -y -n "$ENV_NAME" -c conda-forge numpy matplotlib pandas scipy opencv pyqt lxml albumentations labelImg sqlite pysqlite3 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "[$(date)] [ERROR] Package installation failed!"
    exit 1
fi
echo "[$(date)] [INFO] Python packages installed successfully."

# --- Install Additional Python Libraries via pip ---
for package in ultralytics pypylon; do
    echo "[$(date)] [INFO] Installing $package using pip..."
    conda run -n "$ENV_NAME" python -m pip install "$package" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "[$(date)] [ERROR] Failed to install $package!"
        exit 1
    fi
    echo "[$(date)] [INFO] Package $package installed successfully."
done

# --- Install Jupyter and Extensions ---
echo "[$(date)] [INFO] Installing Jupyter and extensions..."
conda install -y -n "$ENV_NAME" jupyterlab notebook ipywidgets widgetsnbextension ipykernel -c conda-forge > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "[$(date)] [ERROR] Jupyter and extensions installation failed!"
    exit 1
fi
echo "[$(date)] [INFO] Jupyter and extensions installed successfully."

# --- Final Conda Update ---
echo "[$(date)] [INFO] Updating Conda packages..."
conda update --all -y -n "$ENV_NAME" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "[$(date)] [ERROR] Conda update failed!"
    exit 1
fi
echo "[$(date)] [INFO] Conda packages updated successfully."

# --- Reinstall Required Packages ---
echo "[$(date)] [INFO] Reinstalling required packages to ensure proper setup..."
conda install -y -n "$ENV_NAME" --force-reinstall sqlite > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "[$(date)] [ERROR] Package reinstallation failed!"
    exit 1
fi
echo "[$(date)] [INFO] Required packages reinstalled successfully."

# --- Verify Package Installation ---
if [ -f "$VERIFY_SCRIPT" ]; then
    echo "[$(date)] [INFO] Running verification script: $VERIFY_SCRIPT"
    conda run -n "$ENV_NAME" python "$VERIFY_SCRIPT"
    if [ $? -ne 0 ]; then
        echo "[$(date)] [ERROR] Verification script execution failed!"
        exit 1
    fi
    echo "[$(date)] [INFO] Verification script executed successfully."
else
    echo "[$(date)] [WARNING] Verification script not found. Skipping verification."
fi

# --- Final Success Message ---
echo "[$(date)] [INFO] All packages installed successfully!"
echo "[$(date)] [INFO] Conda environment \"$ENV_NAME\" is ready!"
exit 0