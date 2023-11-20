#!/bin/bash

# Create a Python virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install packages from requirements.txt
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate

echo "Virtual environment created and packages installed."
