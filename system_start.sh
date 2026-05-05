#!/bin/bash
set -e

echo "=== Llama.cpp Embeddings Proxy ==="
echo ""

# Check systemd service status
echo "=== Service Status ==="
if systemctl is-active --quiet llama-embeddings-proxy.service; then
    echo "llama-embeddings-proxy.service: ACTIVE"
    systemctl status llama-embeddings-proxy.service --no-pager --short
else
    echo "llama-embeddings-proxy.service: INACTIVE"
fi