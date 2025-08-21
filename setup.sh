#!/bin/bash

# Setup script for realestate listing scraper

echo "Setting up realestate listing scraper..."

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
if [ -z "$python_version" ]; then
    echo "Error: Python 3.8+ is required but not found"
    exit 1
fi

echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "s1-env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv s1-env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source s1-env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete! To activate the environment, run:"
echo "source s1-env/bin/activate"
echo ""
echo "To run the app:"
echo "streamlit run scrape_redfin_Copy1.py"
