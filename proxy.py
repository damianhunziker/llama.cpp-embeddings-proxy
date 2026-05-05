"""
Llama.cpp General Purpose Proxy
Manages llama-server instances in toolbox container and forwards requests for any model type.
Supports embeddings, chat completions, and text completions in parallel.
Multiple instances of the same model can run simultaneously on different ports.
"""
import asyncio
import os
import time
import subprocess
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any

app = FastAPI(title="Llama.cpp General Purpose Proxy")

# Configuration: Each model gets a base port and supports multiple instances
# model_type: "embedding" | "chat" | "completion"
# For embedding models, --embedding flag is used
# For chat/completion models, --parallel is used for batch efficiency
MODELS_CONFIG = {
    # Embedding models (existing) - single instance only for embeddings
    "bge-m3": {
        "path": "/run/host/data/models/embeddings/bge-m3-Q4_K_M.gguf",
        "base_port": 8012,
        "max_instances": 1,
        "model_type": "embedding"
    },
    "bge-large-en-v1.5": {
        "path": "/run/host/data/models/embeddings/bge-large-en-v1.5-Q5_K_M.gguf",
        "base_port": 8013,
        "max_instances": 1,
        "model_type": "embedding"
    },
    "all-MiniLM-L6-v2": {
        "path": "/run/host/data/models/embeddings/all-MiniLM-L6-v2-f16.gguf",
        "base_port": 8014,
        "max_instances": 1,
        "model_type": "embedding"
    },
    "jina-embeddings-v2-base-en": {
        "path": "/run/host/data/models/embeddings/jina-embeddings-v2-base-en-Q4_K_M.gguf",
        "base_port": 8015,
        "max_instances": 1,
        "model_type": "embedding"
    },
    "nomic-embed-text-v1.5": {
        "path": "/run/host/data/models/embeddings/nomic-embed-text-v1.5-f16.gguf",
        "base_port": 8016,
        "max_instances": 1,
        "model_type": "embedding"
    },
    "bge-base-en-v1.5": {
        "path": "/run/host/data/models/embeddings/bge-base-en-v1.5-Q4_K_M.gguf",
        "base_port": 8017,
        "max_instances": 1,
        "model_type": "embedding",
        "args": ["--pooling", "mean"]
    },
    # Chat models (multi-instance capable)
    "Qwen3.5-9B-uncensored-1": {
        "path": "/run/host/data/models/uncensored/DavidAU/Qwen3.5-9B-Claude-4.6-OS-AV-H-UNCENSORED-THINK-D_AU-Q6_K-imat.gguf",
        "base_port": 8018,
        "max_instances": 5,
        "model_type": "chat"
    }
}

TOOLBOX_CONTAINER = "llama-vulkan-radv"
TIMEOUT_SECONDS = 15 * 60  # 15 minutes
active_models = {}  # Key: f"{model_name}:{instance}"
model_locks = {}  # Locks per model:instance to prevent duplicate startup


def get_model_instance_key(model_name: str, instance: int) -> str:
    """Generate the key for active_models dict."""
    return f"{model_name}:{instance}"


def calc_port_for_instance(base_port: int, instance: int) -> int:
    """Calculate the actual port for a given model instance."""
    return base_port + instance


async def wait_for_server(port, timeout=60):
    """Wait for the internal llama-server to be ready."""
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                res = await client.get(f"http://127.0.0.1:{port}/health", timeout=5.0)
                if res.status_code == 200:
                    return True
            except (httpx.RequestError, httpx.TimeoutException):
                pass
            await asyncio.sleep(0.5)
    return False


async def ensure_container_running():
    """Ensure the toolbox container is running, start if needed."""
    result = subprocess.run(
        ["podman", "inspect", "--format={{.State.Running}}", TOOLBOX_CONTAINER],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0 or "true" not in result.stdout.lower():
        print(f"[Proxy] -> Starting container '{TOOLBOX_CONTAINER}'...")
        subprocess.run(["podman", "start", TOOLBOX_CONTAINER], check=True, timeout=60)
        await asyncio.sleep(3)  # Give container time to initialize


def build_llama_server_cmd(config: Dict[str, Any], port: int, instance: int) -> List[str]:
    """Build the llama-server command based on model configuration."""
    cmd = [
        "podman", "exec", "-d", TOOLBOX_CONTAINER,
        "/usr/bin/llama-server",  # Full path required
        "-m", config["path"],
        "--port", str(port),
        "--host", "127.0.0.1",
        "-ngl", "99",
        "--sleep-idle-seconds", str(TIMEOUT_SECONDS),
        "--parallel", "8",
        "--ctx-size", "128000"
    ]
    
    model_type = config.get("model_type", "embedding")
    
    if model_type == "embedding":
        cmd.append("--embedding")
    # For chat/completion models, no special flag needed - server handles both
    
    # Add any extra args from config
    if "args" in config:
        cmd.extend(config["args"])
    
    return cmd


async def ensure_model_instance_running(model_name: str, instance: int):
    """Start the llama-server for the model instance on its internal port if not already running."""
    if model_name not in MODELS_CONFIG:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not configured.")
    
    config = MODELS_CONFIG[model_name]
    base_port = config["base_port"]
    max_instances = config.get("max_instances", 1)
    model_type = config.get("model_type", "embedding")
    
    # Validate instance number
    if instance < 0 or instance >= max_instances:
        raise HTTPException(
            status_code=400,
            detail=f"Instance {instance} invalid for model '{model_name}'. Valid range: 0-{max_instances - 1}"
        )
    
    port = calc_port_for_instance(base_port, instance)
    key = get_model_instance_key(model_name, instance)
    
    # Ensure container is running before any podman exec commands
    await ensure_container_running()
    
    # Use lock to prevent duplicate startup for same model:instance
    if key not in model_locks:
        model_locks[key] = asyncio.Lock()
    
    async with model_locks[key]:
        # Re-check after acquiring lock - another request may have started it
        if key in active_models:
            proc = active_models[key]["process"]
            if proc is None:
                if await wait_for_server(port, timeout=5):
                    active_models[key]["last_used"] = time.time()
                    return port
                else:
                    print(f"[Proxy] -> Orphaned server on port {port} not responding, restarting...")
                    del active_models[key]
            elif proc.poll() is None:
                if await wait_for_server(port, timeout=5):
                    active_models[key]["last_used"] = time.time()
                    return port
                else:
                    print(f"[Proxy] -> Server on port {port} not responding, restarting...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    del active_models[key]
            else:
                del active_models[key]
        
        # Check if a server is already listening on the port (orphaned process)
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
            if res.status_code == 200:
                # Orphaned server found - track it and reuse
                print(f"[Proxy] -> Found existing server on port {port}, using it.")
                active_models[key] = {
                    "process": None,  # Managed externally
                    "last_used": time.time(),
                    "port": port,
                    "model_type": model_type,
                    "model_name": model_name,
                    "instance": instance
                }
                return port
    except:
        pass
    
    # Kill any orphaned llama-server processes on this port (inside container)
    try:
        result = subprocess.run(
            ["podman", "exec", TOOLBOX_CONTAINER, "pkill", "-f", f"llama-server.*{port}"],
            capture_output=True, timeout=10, text=True
        )
        print(f"[Proxy] -> pkill output: {result.stdout}, error: {result.stderr}")
    except Exception as e:
        print(f"[Proxy] -> pkill error: {e}")
    
    print(f"[Proxy] -> Starting llama-server for '{model_name}' instance {instance} (type: {model_type}) on internal port {port}...")
    
    # Build command
    cmd = build_llama_server_cmd(config, port, instance)
    
    # Inherit environment (no custom env vars needed - CLI args are used)
    env = os.environ.copy()
    
    # Start the process - log to file for debugging
    log_file = open(f"/home/damian/services/llama.cpp-embeddings-proxy/llama-{port}.log", "w")
    log_file.write(f"[DEBUG] Starting: {' '.join(cmd)}\n")
    log_file.flush()
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env
    )
    print(f"[Proxy] -> Started process PID {process.pid} for '{model_name}' instance {instance} on port {port}")
    
    # Wait for server to be ready
    if not await wait_for_server(port):
        process.kill()
        raise HTTPException(status_code=500, detail="Timeout starting internal llama-server")
    
    active_models[key] = {
        "process": process,
        "last_used": time.time(),
        "port": port,
        "model_type": model_type,
        "model_name": model_name,
        "instance": instance
    }
    print(f"[Proxy] -> '{model_name}' instance {instance} ready on internal port {port}")
    return port


async def cleanup_inactive_models():
    """Stop model instances that haven't been used for TIMEOUT_SECONDS."""
    while True:
        await asyncio.sleep(60)
        current_time = time.time()
        for key in list(active_models.keys()):
            if current_time - active_models[key]["last_used"] > TIMEOUT_SECONDS:
                proc = active_models[key]["process"]
                model_name = active_models[key]["model_name"]
                instance = active_models[key]["instance"]
                if proc is not None:
                    print(f"[Proxy] -> Stopping '{model_name}' instance {instance} (15 min inactive). VRAM released.")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                else:
                    # Orphaned server - try to kill it via pkill
                    port = active_models[key]["port"]
                    print(f"[Proxy] -> Stopping orphaned server on port {port} (15 min inactive).")
                    try:
                        subprocess.run(
                            ["podman", "exec", TOOLBOX_CONTAINER, 
                             "pkill", "-f", f"llama-server.*{port}"],
                            capture_output=True, timeout=10
                        )
                    except:
                        pass
                del active_models[key]


@app.on_event("startup")
async def startup_event():
    """Start background cleanup task."""
    asyncio.create_task(cleanup_inactive_models())
    model_summary = []
    for name, cfg in MODELS_CONFIG.items():
        max_inst = cfg.get("max_instances", 1)
        model_summary.append(f"{name} (instances 0-{max_inst-1})")
    print(f"[Proxy] -> Started. Configured models: {model_summary}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up all llama-server processes on shutdown."""
    print("[Proxy] -> Shutting down all internal llama-servers...")
    for key, data in active_models.items():
        proc = data["process"]
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


# --- Public Endpoints (Port 8001) ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "proxy": "llama.cpp-general-proxy"}


@app.get("/v1/models")
async def list_models():
    """List available models (proxy reports configured models)."""
    return {
        "object": "list",
        "data": [
            {
                "id": k,
                "object": "model",
                "owned_by": "local-gateway",
                "model_type": v.get("model_type", "embedding"),
                "max_instances": v.get("max_instances", 1)
            }
            for k, v in MODELS_CONFIG.items()
        ]
    }


@app.post("/v1/embeddings")
async def proxy_embeddings(request: Request):
    """Start model if needed, forward embedding request to internal server."""
    body = await request.json()
    model_name = body.get("model")
    
    if not model_name:
        raise HTTPException(status_code=400, detail="Field 'model' is required in request body.")
    
    if model_name not in MODELS_CONFIG:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not configured.")
    
    config = MODELS_CONFIG[model_name]
    if config.get("model_type") != "embedding":
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not an embedding model.")
    
    # Instance parameter (default 0 for backward compatibility)
    instance = body.pop("instance", 0)
    
    # Ensure model instance is running (starts if not, updates timeout if running)
    internal_port = await ensure_model_instance_running(model_name, instance)
    
    # Forward request to internal llama-server
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"http://127.0.0.1:{internal_port}/v1/embeddings",
                json=body,
                timeout=120.0  # Extended timeout for large batches
            )
            return JSONResponse(status_code=response.status_code, content=response.json())
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Internal server error: {str(e)}")


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    """Start model if needed, forward chat completions request to internal server."""
    body = await request.json()
    model_name = body.get("model")
    
    if not model_name:
        raise HTTPException(status_code=400, detail="Field 'model' is required in request body.")
    
    if model_name not in MODELS_CONFIG:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not configured.")
    
    config = MODELS_CONFIG[model_name]
    if config.get("model_type") not in ("chat", "embedding"):
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' does not support chat completions.")
    
    # Instance parameter (default 0 for backward compatibility)
    instance = body.pop("instance", 0)
    
    # Ensure model instance is running (starts if not, updates timeout if running)
    internal_port = await ensure_model_instance_running(model_name, instance)
    
    # Forward request to internal llama-server
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"http://127.0.0.1:{internal_port}/v1/chat/completions",
                json=body,
                timeout=300.0  # 5 min timeout for chat completions
            )
            return JSONResponse(status_code=response.status_code, content=response.json())
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Internal server error: {str(e)}")


@app.post("/v1/completions")
async def proxy_completions(request: Request):
    """Start model if needed, forward text completions request to internal server."""
    body = await request.json()
    model_name = body.get("model")
    
    if not model_name:
        raise HTTPException(status_code=400, detail="Field 'model' is required in request body.")
    
    if model_name not in MODELS_CONFIG:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not configured.")
    
    config = MODELS_CONFIG[model_name]
    if config.get("model_type") not in ("completion", "embedding"):
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' does not support text completions.")
    
    # Instance parameter (default 0 for backward compatibility)
    instance = body.pop("instance", 0)
    
    # Ensure model instance is running (starts if not, updates timeout if running)
    internal_port = await ensure_model_instance_running(model_name, instance)
    
    # Forward request to internal llama-server
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"http://127.0.0.1:{internal_port}/v1/completions",
                json=body,
                timeout=300.0  # 5 min timeout for completions
            )
            return JSONResponse(status_code=response.status_code, content=response.json())
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
