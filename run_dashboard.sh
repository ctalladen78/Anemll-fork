#!/bin/bash
# Move to script's directory safely
cd "$(dirname "$0")"

# Activate the virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "ERROR: Virtual environment not found. Please setup '.venv' first."
    exit 1
fi

echo "🚀 Launching Llama-CPP TurboQuant Dashboard..."
python3 dashboard.py
