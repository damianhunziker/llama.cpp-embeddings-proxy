# System Monitor JSON Schema

## Structure
```json
{
  "timestamp": "2025-11-24T21:09:50Z",
  "services": [
    {
      "name": "service-name",
      "status": "running|stopped",
      "pid": number|null,
      "type": "mcp-server",
      "url": "http://localhost:23423"
    }
  ],
  "system": {
    "cpu_usage": "percentage",
    "memory_usage": "size",
    "disk_usage": "percentage"
  }
}
```

## Fields
- **timestamp**: ISO 8601 UTC timestamp
- **services**: Array of service objects
  - **name**: Service identifier
  - **status**: "running" or "stopped"
  - **pid**: Process ID or null if stopped
  - **type**: Service type ("mcp-server")
  - **url**: Url if any where the service is accessible ("http://localhost:23423")
- **system**: System metrics
  - **cpu_usage**: Current CPU usage percentage
  - **memory_usage**: Current memory usage
  - **disk_usage**: Root disk usage percentage