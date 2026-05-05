# Llama.cpp General Purpose Proxy

Manages llama-server instances in toolbox container and forwards requests for any model type. Supports embeddings, chat completions, and text completions in parallel.

## Setup

```bash
cd services/llama.cpp-embeddings-proxy
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Start

```bash
.venv/bin/python proxy.py
```

The proxy runs on port 8001.

## How It Works

### Architecture Overview

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────────────────────────┐
│   Client    │────▶│  Proxy :8001    │────▶│  Llama-server instances (ports)     │
│             │◀────│  (FastAPI)      │◀────│                                      │
└─────────────┘     └─────────────────┘     │  ┌─────────┐  ┌─────────┐          │
                                             │  │Model:0  │  │Model:1  │  ...      │
                                             │  │port 8018│  │port 8019│          │
                                             │  └─────────┘  └─────────┘          │
                                             │       Container: llama-vulkan-radv│
                                             └──────────────────────────────────────┘
```

### Request Flow

1. Client sends request to `localhost:8001` with model name + optional instance
2. Proxy extracts `model` and `instance` parameters from request body
3. Proxy calculates target port: `base_port + instance`
4. If model instance not running, proxy starts new llama-server in container
5. Proxy forwards request to internal llama-server on calculated port
6. Proxy returns response to client
7. After 15 minutes of inactivity, instance is stopped to free VRAM

### Instance-Based Parallelization

Each model can have multiple instances running simultaneously on different ports:

| Model | Base Port | Instances | Ports |
|-------|-----------|-----------|-------|
| bge-m3 (embedding) | 8012 | 1 | 8012 |
| Qwen3.5-9B-uncensored-1 | 8018 | 5 | 8018-8022 |

**Port Calculation:** `actual_port = base_port + instance_number`

### Model Types

| Type | Flags | Endpoints | Description |
|------|-------|-----------|-------------|
| `embedding` | `--embedding` | `/v1/embeddings` | Text embedding models |
| `chat` | (none, server handles) | `/v1/chat/completions` | Conversational models |
| `completion` | (none, server handles) | `/v1/completions` | Text completion models |

All types use `--parallel 8` for batch efficiency.

## API

- `GET /health` - Health check
- `GET /v1/models` - List available models (includes `max_instances` per model)
- `POST /v1/embeddings` - Generate embeddings
- `POST /v1/chat/completions` - Chat completions
- `POST /v1/completions` - Text completions

### Request Format

All endpoints accept a JSON body with:

```json
{
    "model": "model-name",
    "instance": 0,           // optional, default: 0
    ...                      // model-specific parameters
}
```

#### Embeddings Request

```json
{
    "model": "bge-m3",
    "instance": 0,
    "input": "The quick brown fox"
}
```

#### Chat Completions Request

```json
{
    "model": "Qwen3.5-9B-uncensored-1",
    "instance": 1,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
}
```

#### Text Completions Request

```json
{
    "model": "Qwen3.5-9B-uncensored-1",
    "instance": 2,
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.8
}
```

## Configuration

Edit `proxy.py` to add or modify models in `MODELS_CONFIG`:

```python
MODELS_CONFIG = {
    "model-name": {
        "path": "/run/host/data/models/model.gguf",
        "base_port": 8020,       # Starting port for instances
        "max_instances": 4,      # Number of parallel instances (1-4)
        "model_type": "chat",    # "embedding" | "chat" | "completion"
        "args": []               # optional: extra llama-server flags
    }
}
```

### Adding a New Chat Model

```python
"my-chat-model": {
    "path": "/run/host/data/models/chat/my-model-Q4_K_M.gguf",
    "base_port": 8030,
    "max_instances": 3,
    "model_type": "chat",
    "args": ["--ctx-size", "8192"]  # optional: custom context size
}
```

This allocates ports 8030, 8031, 8032 for instances 0, 1, 2.

## Lifecycle Management

### Startup Behavior

- Proxy starts background cleanup task (runs every 60 seconds)
- Model instances are started on-demand per request
- First request to an instance triggers llama-server startup
- Subsequent requests to same instance reuse running server (and reset timeout)

### Timeout & Cleanup

- **Timeout:** 15 minutes of inactivity per instance
- **Check interval:** Every 60 seconds
- **On timeout:** Process is terminated, VRAM is released
- **Graceful shutdown:** SIGTERM sent first, SIGKILL after 5s

### Instance Startup Locking

- Uses asyncio.Lock per `model:instance` key
- Prevents duplicate startup when concurrent requests target same instance
- Re-checks server health after acquiring lock to handle race conditions

### Orphaned Server Handling

If a llama-server process exists on the port but is not tracked:

1. Health check validates if it's responding
2. If responding: proxy adopts it (tracks and reuses)
3. If not responding: proxy kills it via `pkill -f llama-server.*{port}`

## Parallel Execution Example

```bash
# Start instance 0
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.5-9B-uncensored-1", "instance": 0, "messages": [...]}'

# Start instance 1 (runs in parallel, different port)
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.5-9B-uncensored-1", "instance": 1, "messages": [...]}'

# All 5 instances can run simultaneously
# Ports: 8018, 8019, 8020, 8021, 8022
```

## Systemd Service

Install as service:

```bash
sudo cp llama-embeddings-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable llama-embeddings-proxy.service
sudo systemctl start llama-embeddings-proxy.service
```

## Container

The proxy uses `podman exec` into container `llama-rocm7-nightlies` to manage llama-server processes.

## Note

All timing behavior, cleanup, and parallelization remain identical to the original embeddings-only proxy. Only the endpoint support and instance model have been extended.
# llama.cpp-embeddings-proxy
