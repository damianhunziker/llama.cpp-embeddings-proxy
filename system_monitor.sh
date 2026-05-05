#!/bin/bash
set -e

# Get timestamp
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Check systemd service status
if systemctl is-active --quiet llama-embeddings-proxy.service; then
    SERVICE_STATUS="running"
else
    SERVICE_STATUS="stopped"
fi

# Get system metrics
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
MEMORY_USAGE=$(free -h | awk '/^Mem:/ {print $3 "/" $2}')
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}')

# Output JSON
cat << EOF
{
  "timestamp": "$TIMESTAMP",
  "services": [
    {"name": "llama-embeddings-proxy", "status": "$SERVICE_STATUS", "type": "systemd-service", "url": "http://localhost:8001/health"}
  ],
  "system": {
    "cpu_usage": "${CPU_USAGE}%",
    "memory_usage": "${MEMORY_USAGE}",
    "disk_usage": "${DISK_USAGE}"
  }
}
EOF