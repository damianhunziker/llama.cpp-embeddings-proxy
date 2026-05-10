"""
Llama.cpp General Purpose Proxy
Manages llama-server instances in toolbox container and forwards requests for any model type.
Supports embeddings, chat completions, and text completions in parallel.
Multiple instances of the same model can run simultaneously on different ports.
VRAM-based auto-scaling: automatically starts new instances when model is busy and VRAM available.
"""
import asyncio
import json
import logging
import os
import time
import subprocess
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any

logger = logging.getLogger("llama-proxy")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

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
        "max_instances": 8,
        "model_type": "chat",
        "stream": False,
        "vram_mb": 6000  # Estimated VRAM per instance in MB
    }
}

TOOLBOX_CONTAINER = "llama-vulkan-radv"  # ROCm container with proper user mapping
TIMEOUT_SECONDS = 15 * 60  # 15 minutes for embedding models
TIMEOUT_SECONDS_CHAT = 180  # 180 seconds for chat/completion models (prompt cache retention)

# Global flag to disable streaming for all chat/completion requests
# Set to true to force disable streaming, false to allow client control
DISABLE_STREAMING_GLOBAL = True


def kill_llama_server(port: int):
    """Kill llama-server processes on the specified port inside the container.
    Uses pkill -9 inside the container to kill the process and free VRAM.
    """
    try:
        # pkill -9 sends SIGKILL to the process, ensuring VRAM is freed
        result = subprocess.run(
            ["podman", "exec", TOOLBOX_CONTAINER, "pkill", "-9", "-f", f"llama-server.*{port}"],
            capture_output=True, text=True, timeout=30
        )
        # pkill returns 0 if processes were found and killed, 1 if none found
        if result.returncode == 0:
            logger.info("Killed llama-server on port %d", port)
        elif result.returncode == 1:
            logger.info("No llama-server found on port %d", port)
        else:
            logger.warning("pkill error on port %d: %s", port, result.stderr)
        
        time.sleep(0.3)  # Allow VRAM to be freed
        return True
    except subprocess.TimeoutExpired:
        logger.warning("pkill timed out for port %d", port)
    except Exception as e:
        logger.error("pkill error on port %d: %s", port, e)
    
    return False


VRAM_RESERVE_MB = 500  # Keep 500MB reserve

# Pending instances - being started by another request
# Key: model_name:instance, Value: asyncio.Event to wait on
pending_instances: Dict[str, asyncio.Event] = {}

active_models = {}  # Key: f"{model_name}:{instance}"
model_locks = {}  # Locks per model:instance to prevent duplicate startup
model_scale_locks = {}  # Locks per model to prevent race in instance selection


def get_model_instance_key(model_name: str, instance: int) -> str:
    """Generate the key for active_models dict."""
    return f"{model_name}:{instance}"


def calc_port_for_instance(base_port: int, instance: int) -> int:
    """Calculate the actual port for a given model instance."""
    return base_port + instance


async def get_vram_info() -> tuple[int, int]:
    """Get VRAM usage from rocm-smi. Returns (used_mb, total_mb)."""
    try:
        result = subprocess.run(
            ["podman", "exec", TOOLBOX_CONTAINER, "rocm-smi", "--showusedvram", "--json"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            logger.warning("rocm-smi error: %s", result.stderr)
            return 0, 0
        
        data = json.loads(result.stdout)
        # Handle both single GPU and multi-GPU output
        if isinstance(data, list):
            data = data[0]
        elif isinstance(data, dict) and "cards" in data:
            data = data["cards"][0]
        
        used_str = data.get("used_vram", "0M")
        total_str = data.get("total_vram", "0M")
        
        used_mb = int(used_str.replace("MiB", "").replace("M", "").replace("GiB", "").replace("G", ""))
        total_mb = int(total_str.replace("MiB", "").replace("M", "").replace("GiB", "").replace("G", ""))
        
        # Convert GiB to MiB if necessary
        if "GiB" in used_str or "G" in used_str:
            used_mb *= 1024
        if "GiB" in total_str or "G" in total_str:
            total_mb *= 1024
            
        return used_mb, total_mb
    except Exception as e:
        logger.warning("Failed to get VRAM info: %s", e)
        return 0, 0


async def get_available_vram() -> int:
    """Get available VRAM in MB."""
    used_mb, total_mb = await get_vram_info()
    available = total_mb - used_mb - VRAM_RESERVE_MB
    return max(0, available)


def get_model_vram_requirement(model_name: str) -> int:
    """Get VRAM requirement for model in MB."""
    config = MODELS_CONFIG[model_name]
    if "vram_mb" in config:
        return config["vram_mb"]
    
    # Fallback: estimate from file size
    path = config["path"]
    try:
        size_bytes = os.path.getsize(path)
        size_mb = size_bytes // (1024 * 1024)
        # Rough estimation: ~2.5x the file size for Q4_K_M models
        return int(size_mb * 2.5)
    except:
        return 4000  # Default fallback


async def can_start_new_instance(model_name: str) -> tuple[bool, str]:
    """Check if a new instance can be started. Returns (can_start, reason)."""
    config = MODELS_CONFIG[model_name]
    max_instances = config.get("max_instances", 1)
    
    # Count current instances
    current_instances = 0
    for key in active_models:
        if key.startswith(f"{model_name}:"):
            current_instances += 1
    
    if current_instances >= max_instances:
        return False, f"Max instances ({max_instances}) reached"
    
    # Check VRAM
    required_mb = get_model_vram_requirement(model_name)
    available_mb = await get_available_vram()
    
    if available_mb < required_mb:
        return False, f"Insufficient VRAM (need {required_mb}MB, have {available_mb}MB)"
    
    return True, "OK"


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
        logger.info("Starting container '%s'...", TOOLBOX_CONTAINER)
        subprocess.run(["podman", "start", TOOLBOX_CONTAINER], check=True, timeout=60)
        await asyncio.sleep(3)  # Give container time to initialize


def build_llama_server_cmd(config: Dict[str, Any], port: int, instance: int) -> List[str]:
    """Build the llama-server command based on model configuration."""
    model_type = config.get("model_type", "embedding")
    
    # Use 1s timeout for chat/completion models (close immediately when idle)
    # Use longer timeout for embedding models
    timeout = TIMEOUT_SECONDS_CHAT if model_type in ("chat", "completion") else TIMEOUT_SECONDS
    
    # Parallel slots: embedding models benefit from batch parallelism,
    # chat/completion models use 1 slot so each request gets the full --ctx-size.
    # llama.cpp divides --ctx-size equally among --parallel slots.
    parallel_slots = 1 if model_type in ("chat", "completion") else 8
    
    cmd = [
        "/usr/bin/llama-server",  # Full path inside ROCm container
        "-m", config["path"],
        "--port", str(port),
        "--host", "127.0.0.1",
        "-ngl", "99",
        "--sleep-idle-seconds", str(timeout),
        "--parallel", str(parallel_slots),
        "--ctx-size", "128000"
    ]
    
    if model_type == "embedding":
        cmd.append("--embedding")
    # For chat/completion models, no special flag needed - server handles both
    
    # Add any extra args from config
    if "args" in config:
        cmd.extend(config["args"])
    
    return cmd


async def find_idle_or_new_instance(model_name: str) -> tuple[int, str]:
    """Find an idle instance or determine if a new one can be started.
    Returns (instance_number, key).
    Raises HTTPException with status 503 if scaling is needed but not possible.
    
    This function uses a per-model lock to prevent race conditions where
    multiple requests select the same slot simultaneously.
    """
    config = MODELS_CONFIG[model_name]
    max_instances = config.get("max_instances", 1)
    base_port = config["base_port"]
    
    # Get or create model-level lock for instance selection
    if model_name not in model_scale_locks:
        model_scale_locks[model_name] = asyncio.Lock()
    lock = model_scale_locks[model_name]
    
    await lock.acquire()
    try:
        # First pass: find idle instance
        for i in range(max_instances):
            key = get_model_instance_key(model_name, i)
            if key in active_models and not active_models[key].get("busy", False):
                # Check if server is still responsive
                port = calc_port_for_instance(base_port, i)
                if await wait_for_server(port, timeout=2):
                    # Mark as busy immediately to prevent other requests from selecting
                    active_models[key]["busy"] = True
                    return i, key
                # Unresponsive instance will be cleaned up, remove from active
                logger.info("Removing unresponsive instance %d from active_models", i)
                del active_models[key]
        
        # Second pass: find unused slot
        for i in range(max_instances):
            key = get_model_instance_key(model_name, i)
            if key not in active_models:
                # Check if something is already listening on this port
                # If so, it's likely being started by another request - skip
                port = calc_port_for_instance(base_port, i)
                try:
                    async with httpx.AsyncClient() as client:
                        res = await client.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
                        if res.status_code == 200:
                            # Something is already on this port - likely being started
                            # Register as pending and wait for it
                            if key not in pending_instances:
                                pending_instances[key] = asyncio.Event()
                            event = pending_instances[key]
                            # Release lock while waiting, then re-acquire after
                            lock.release()
                            try:
                                # Wait for the other request to finish starting
                                await asyncio.wait_for(event.wait(), timeout=60)
                            except asyncio.TimeoutError:
                                raise HTTPException(status_code=503, detail=f"Timeout waiting for instance {i}")
                            # Re-acquire lock and mark as busy
                            await lock.acquire()
                            # Double-check it's still valid and not busy
                            if key in active_models and not active_models[key].get("busy", False):
                                active_models[key]["busy"] = True
                                return i, key
                            # If it became busy or was removed, try finding another
                            continue
                except:
                    # Port is free - we can use this slot
                    pass
                # Mark the slot as pending (being started by us)
                active_models[key] = {
                    "process": None,  # Will be set after starting
                    "last_used": time.time(),
                    "port": port,
                    "model_type": config.get("model_type", "embedding"),
                    "model_name": model_name,
                    "instance": i,
                    "busy": True  # Mark busy immediately
                }
                return i, key
        
        # All slots full, check if we can auto-scale
        can_start, reason = await can_start_new_instance(model_name)
        if not can_start:
            raise HTTPException(
                status_code=503,
                detail=f"No available instances for '{model_name}': {reason}"
            )
        
        # Return first slot - will be started fresh, mark as busy
        i = 0
        key = get_model_instance_key(model_name, i)
        port = calc_port_for_instance(base_port, i)
        active_models[key] = {
            "process": None,
            "last_used": time.time(),
            "port": port,
            "model_type": config.get("model_type", "embedding"),
            "model_name": model_name,
            "instance": i,
            "busy": True  # Mark busy immediately
        }
        return i, key
    finally:
        if lock.locked():
            lock.release()


async def ensure_model_instance_running(model_name: str, instance: int):
    """Start the llama-server for the model instance on its internal port if not already running.
    If instance is -1, auto-detects: tries idle instance first, then starts new if VRAM allows.
    Returns (port, key) tuple.
    """
    if model_name not in MODELS_CONFIG:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not configured.")
    
    config = MODELS_CONFIG[model_name]
    base_port = config["base_port"]
    max_instances = config.get("max_instances", 1)
    model_type = config.get("model_type", "embedding")
    
    # Auto-detect instance if not specified
    if instance == -1:
        instance, key = await find_idle_or_new_instance(model_name)
        logger.info("Auto-selected instance %d for '%s'", instance, model_name)
    else:
        # Validate instance number for explicit instance
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
        # Also check if this instance is currently being started by another request
        if key in active_models:
            proc = active_models[key]["process"]
            if proc is None:
                # Could be an orphaned server or one being started - check health
                if await wait_for_server(port, timeout=5):
                    active_models[key]["last_used"] = time.time()
                    return port, key
                else:
                    logger.warning("Orphaned server on port %d not responding, restarting...", port)
                    del active_models[key]
            elif proc.poll() is None:
                if await wait_for_server(port, timeout=5):
                    active_models[key]["last_used"] = time.time()
                    return port, key
                else:
                    logger.warning("Server on port %d not responding, restarting...", port)
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    del active_models[key]
            else:
                del active_models[key]
        
        # Check if a server is already listening on the port (orphaned process)
        # This check is INSIDE the lock to prevent two requests from starting
        # two different servers on the same port
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
                if res.status_code == 200:
                    # Orphaned server found - track it and reuse
                    logger.info("Found existing server on port %d, using it.", port)
                    active_models[key] = {
                        "process": None,  # Managed externally
                        "last_used": time.time(),
                        "port": port,
                        "model_type": model_type,
                        "model_name": model_name,
                        "instance": instance,
                        "busy": False
                    }
                    return port, key
        except Exception:
            pass
        
        # Kill any orphaned llama-server processes on this port (running inside container)
        # Only if no valid server was found above
        try:
            kill_llama_server(port)
        except subprocess.TimeoutExpired:
            logger.warning("pkill timed out for port %d, continuing anyway", port)
        except Exception as e:
            logger.error("pkill error: %s", e)
    
    logger.info("Starting llama-server for '%s' instance %d (type: %s) on internal port %d...", model_name, instance, model_type, port)
    
    # Build command
    # Log file path inside container for this instance
    container_log = f"/tmp/llama-{port}.log"
    
    # Build command with log file
    cmd = build_llama_server_cmd(config, port, instance)
    cmd.extend(["-v", "--log-file", container_log])
    
    # Inherit environment (no custom env vars needed - CLI args are used)
    env = os.environ.copy()
    
    # Start the process inside the container with logging
    # Use > for redirect to log file, nohup to ignore SIGHUP, & to run in background
    container_cmd = f"nohup {' '.join(cmd)} >> {container_log} 2>&1 &"
    process = subprocess.Popen(
        ["podman", "exec", TOOLBOX_CONTAINER, "sh", "-c", container_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
    logger.info("Started llama-server for '%s' instance %d on port %d (inside container)", model_name, instance, port)
    logger.info("View logs: podman exec llama-vulkan-radv cat /tmp/llama-%d.log", port)
    
    # Wait for server to be ready (up to 3 min for large models)
    if not await wait_for_server(port, timeout=180):
        # Kill any stray processes on this port inside container
        kill_llama_server(port)
        raise HTTPException(status_code=500, detail="Timeout starting internal llama-server")
    
    active_models[key] = {
        "process": process,
        "last_used": time.time(),
        "port": port,
        "model_type": model_type,
        "model_name": model_name,
        "instance": instance,
        "busy": False
    }
    logger.info("'%s' instance %d ready on internal port %d", model_name, instance, port)
    
    # Signal any waiting requests that this instance is ready
    if key in pending_instances:
        pending_instances[key].set()
        del pending_instances[key]
    
    return port, key


def mark_instance_busy(key: str):
    """Mark an instance as busy (processing a request)."""
    if key in active_models:
        active_models[key]["busy"] = True


def mark_instance_idle(key: str):
    """Mark an instance as idle (done processing)."""
    if key in active_models:
        active_models[key]["busy"] = False


async def stop_instance(key: str):
    """Stop a specific llama-server instance and remove from active_models."""
    if key not in active_models:
        return
    
    data = active_models[key]
    port = data["port"]
    model_name = data["model_name"]
    instance = data["instance"]
    
    logger.info("Stopping '%s' instance %d on port %d.", model_name, instance, port)
    kill_llama_server(port)
    del active_models[key]


async def cleanup_inactive_models():
    """Stop model instances that haven't been used for their timeout period or are idle.
    Chat/completion models: TIMEOUT_SECONDS_CHAT (180s) for prompt cache retention.
    Embedding models: TIMEOUT_SECONDS (15 min) for long batch processing.
    """
    while True:
        await asyncio.sleep(60)
        current_time = time.time()
        for key in list(active_models.keys()):
            data = active_models[key]
            
            # Skip if busy (currently processing a request)
            if data.get("busy", False):
                continue
            
            # Use model-type-specific timeout
            model_type = data.get("model_type", "embedding")
            timeout = TIMEOUT_SECONDS_CHAT if model_type in ("chat", "completion") else TIMEOUT_SECONDS
            
            if current_time - data["last_used"] > timeout:
                model_name = data["model_name"]
                instance = data["instance"]
                port = data["port"]
                logger.info("Stopping '%s' instance %d (timeout after %ds). VRAM released.", model_name, instance, timeout)
                kill_llama_server(port)
                del active_models[key]


@app.on_event("startup")
async def startup_event():
    """Start background cleanup task."""
    asyncio.create_task(cleanup_inactive_models())
    model_summary = []
    for name, cfg in MODELS_CONFIG.items():
        max_inst = cfg.get("max_instances", 1)
        model_summary.append(f"{name} (instances 0-{max_inst-1})")
    logger.info("Started. Configured models: %s", model_summary)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up all llama-server processes on shutdown."""
    logger.info("Shutting down all internal llama-servers...")
    for key, data in active_models.items():
        port = data["port"]
        kill_llama_server(port)


# --- Public Endpoints (Port 8001) ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "proxy": "llama.cpp-general-proxy"}


@app.get("/vram")
async def get_vram_status():
    """Get current VRAM usage status."""
    used_mb, total_mb = await get_vram_info()
    available_mb = total_mb - used_mb - VRAM_RESERVE_MB
    
    instance_summary = []
    for key, data in active_models.items():
        instance_summary.append({
            "key": key,
            "busy": data.get("busy", False),
            "last_used": data["last_used"]
        })
    
    return {
        "used_mb": used_mb,
        "total_mb": total_mb,
        "available_mb": max(0, available_mb),
        "reserve_mb": VRAM_RESERVE_MB,
        "active_instances": instance_summary
    }


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
                "max_instances": v.get("max_instances", 1),
                "vram_mb": v.get("vram_mb", "estimated")
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
    
    # Instance parameter: -1 for auto-select, 0+ for specific instance
    instance = body.pop("instance", -1)
    
    # Ensure model instance is running (starts if not, updates timeout if running)
    internal_port, key = await ensure_model_instance_running(model_name, instance)
    
    try:
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
    finally:
        # Mark instance as idle when done
        mark_instance_idle(key)


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    """Start model if needed, forward chat completions request to internal server.
    For chat models: instance stays alive for TIMEOUT_SECONDS_CHAT (180s) to enable prompt caching.
    """
    body = await request.json()
    model_name = body.get("model")
    
    if not model_name:
        raise HTTPException(status_code=400, detail="Field 'model' is required in request body.")
    
    if model_name not in MODELS_CONFIG:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not configured.")
    
    config = MODELS_CONFIG[model_name]
    if config.get("model_type") not in ("chat", "embedding"):
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' does not support chat completions.")
    
    # Instance parameter: -1 for auto-select, 0+ for specific instance
    instance = body.pop("instance", -1)
    
    # Global streaming disable flag
    if DISABLE_STREAMING_GLOBAL and body.get("stream"):
        body["stream"] = False
    
    # Ensure model instance is running (starts if not)
    internal_port, key = await ensure_model_instance_running(model_name, instance)
    
    try:
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
    finally:
        # Mark instance as idle when done - let cleanup task handle removal after timeout
        mark_instance_idle(key)


@app.post("/v1/completions")
async def proxy_completions(request: Request):
    """Start model if needed, forward text completions request to internal server.
    For completion models: instance stays alive for TIMEOUT_SECONDS_CHAT (180s) to enable prompt caching.
    """
    body = await request.json()
    model_name = body.get("model")
    
    if not model_name:
        raise HTTPException(status_code=400, detail="Field 'model' is required in request body.")
    
    if model_name not in MODELS_CONFIG:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not configured.")
    
    config = MODELS_CONFIG[model_name]
    if config.get("model_type") not in ("completion", "embedding"):
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' does not support text completions.")
    
    # Instance parameter: -1 for auto-select, 0+ for specific instance
    instance = body.pop("instance", -1)
    
    # Global streaming disable flag
    if DISABLE_STREAMING_GLOBAL and body.get("stream"):
        body["stream"] = False
    
    # Ensure model instance is running (starts if not)
    internal_port, key = await ensure_model_instance_running(model_name, instance)
    
    try:
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
    finally:
        # Mark instance as idle when done - let cleanup task handle removal after timeout
        mark_instance_idle(key)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
