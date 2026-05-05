#!/bin/bash
# Start Llama.cpp General Purpose Proxy (CLI Modus)

echo "=== Llama.cpp General Purpose Proxy CLI Start ==="
echo ""

# Stop systemd service if active
echo "=== Prüfe systemd Service ==="
if systemctl is-active --quiet llama-embeddings-proxy.service; then
    echo "Stoppe systemd Service..."
    systemctl stop llama-embeddings-proxy.service
fi

# Kill any existing proxy processes on port 8001
echo "=== Prüfe und stoppe Prozesse auf Port 8001 ==="
PID=$(lsof -t -i:8001 2>/dev/null)
if [ -n "$PID" ]; then
    echo "Stoppe Prozess auf Port 8001 (PID: $PID)..."
    kill -9 $PID 2>/dev/null
    sleep 1
fi

# Kill any existing llama-server processes started by proxy (ports 8012-8017)
echo "=== Prüfe und stoppe llama-server Prozesse (Host) ==="
for PORT in 8012 8013 8014 8015 8016 8017; do
    PIDS=$(lsof -t -i:$PORT 2>/dev/null)
    if [ -n "$PIDS" ]; then
        echo "Stoppe Prozesse auf Port $PORT..."
        for pid in $PIDS; do
            kill -9 $pid 2>/dev/null
        done
    fi
done

# Kill llama-server processes inside the container
echo "=== Prüfe und stoppe llama-server Prozesse (Container) ==="
podman exec llama-vulkan-radv pkill -9 llama-server 2>/dev/null || true
sleep 1

echo "=== Starte Proxy ==="
cd /home/damian/services/llama.cpp-embeddings-proxy

# Start proxy in foreground (CLI Output)
exec .venv/bin/python -u proxy.py