#!/bin/bash
# Start uvicorn in background
/app/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

# Wait for server to be ready
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Run mcp-proxy pointing at the local server
exec mcp-proxy http://localhost:8000/mcp --transport streamablehttp
