#!/bin/bash
# Start script for Railway deployment
# Uses PORT environment variable with fallback to 8000

PORT=${PORT:-8000}
exec uvicorn main:app --host 0.0.0.0 --port $PORT
