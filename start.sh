#!/bin/bash
# Startup script for Render deployment

echo "[START] Initializing FinSight..."

# Check if database exists, if not create it
if [ ! -f "finsight.db" ]; then
    echo "[INIT] Database not found, creating..."
    python setup.py
else
    echo "[INFO] Database already exists"
fi

echo "[START] Starting Gunicorn..."
gunicorn app:app --bind 0.0.0.0:$PORT --log-level info
