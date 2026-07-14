#!/bin/bash
# Script to set up environment for Brain Tumor CNN project on Linux/Colab

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "Environment setup complete!"
