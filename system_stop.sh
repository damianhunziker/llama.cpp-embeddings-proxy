#!/bin/bash
set -e

echo "=== Llama.cpp General Purpose Proxy - Stop ==="
echo ""

# Check systemd service status
echo "=== Service Status ==="
if systemctl is-active --quiet llama-embeddings-proxy.service; then
    echo "Stopping llama-embeddings-proxy.service..."
    systemctl stop llama-embeddings-proxy.service
    echo "Service stopped."
else
    echo "llama-embeddings-proxy.service: already stopped"
fi