"""
Parallel Chat Completion Test

Tests multiple parallel instances of chat completion running simultaneously.
Each instance handles a different question independently.
"""
import asyncio
import httpx
import time
import sys
from typing import List, Dict, Any

PROXY_URL = "http://localhost:8001"
MODEL_NAME = "Qwen3.5-9B-uncensored-1"
MAX_INSTANCES = 5

# Simple questions for parallel testing
TEST_QUESTIONS = [
    {"role": "system", "content": "You are a helpful assistant. Keep responses short and direct."},
    {"role": "user", "content": "What is the capital of France?"}
]

TEST_COMPLETIONS = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Keep responses very brief."},
            {"role": "user", "content": "What is 2+2?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Keep responses very brief."},
            {"role": "user", "content": "What is the capital of Japan?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Keep responses very brief."},
            {"role": "user", "content": "What is the speed of light?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Keep responses very brief."},
            {"role": "user", "content": "What is H2O?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Keep responses very brief."},
            {"role": "user", "content": "What year is it?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
]


async def check_proxy_health() -> bool:
    """Check if proxy is running."""
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{PROXY_URL}/health", timeout=5.0)
            return res.status_code == 200
    except:
        return False


async def get_vram_status() -> Dict[str, Any]:
    """Get VRAM status."""
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{PROXY_URL}/vram", timeout=5.0)
            return res.json()
    except:
        return {}


async def list_models() -> List[Dict[str, Any]]:
    """List available models."""
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{PROXY_URL}/v1/models", timeout=5.0)
            return res.json().get("data", [])
    except:
        return []


async def chat_completion(instance: int, question_data: Dict[str, Any]) -> Dict[str, Any]:
    """Send a single chat completion request to a specific instance."""
    start_time = time.time()
    
    body = {
        "model": MODEL_NAME,
        "instance": instance,
        **question_data
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{PROXY_URL}/v1/chat/completions",
                json=body,
                timeout=300.0
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {
                    "instance": instance,
                    "status": "success",
                    "elapsed": elapsed,
                    "response": content[:100] + "..." if len(content) > 100 else content
                }
            else:
                return {
                    "instance": instance,
                    "status": "error",
                    "elapsed": elapsed,
                    "error": f"HTTP {response.status_code}: {response.text[:100]}"
                }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "instance": instance,
            "status": "error",
            "elapsed": elapsed,
            "error": str(e)[:100]
        }


async def test_parallel_instances(num_instances: int = 3):
    """Test parallel chat completion across multiple instances."""
    print(f"\n{'='*60}")
    print(f"PARALLEL CHAT COMPLETION TEST")
    print(f"{'='*60}")
    
    # Check proxy health
    print("\n[1] Checking proxy health...")
    if not await check_proxy_health():
        print("ERROR: Proxy is not running on port 8001")
        print("Start with: .venv/bin/python proxy.py")
        return False
    print("✓ Proxy is healthy")
    
    # Get VRAM status
    print("\n[2] VRAM Status:")
    vram = await get_vram_status()
    if vram:
        print(f"   Used: {vram.get('used_mb', 'N/A')} MB")
        print(f"   Total: {vram.get('total_mb', 'N/A')} MB")
        print(f"   Available: {vram.get('available_mb', 'N/A')} MB")
    
    # List models
    print("\n[3] Available Models:")
    models = await list_models()
    for m in models:
        print(f"   - {m['id']} ({m.get('model_type', 'unknown')}) max_instances={m.get('max_instances', 1)}")
    
    # Run parallel tests
    print(f"\n[4] Running parallel test with {num_instances} instances...")
    print(f"    Model: {MODEL_NAME}")
    print(f"    Instances: 0 to {num_instances - 1}")
    print("-" * 60)
    
    # Prepare tasks - use first num_instances completions
    tasks = []
    for i in range(num_instances):
        task = asyncio.create_task(chat_completion(i, TEST_COMPLETIONS[i]))
        tasks.append((i, task))
    
    # Execute all in parallel
    overall_start = time.time()
    results = await asyncio.gather(*[t[1] for t in tasks])
    overall_elapsed = time.time() - overall_start
    
    # Print results
    print("\n[5] RESULTS:")
    success_count = 0
    for i, result in enumerate(results):
        status_icon = "✓" if result["status"] == "success" else "✗"
        print(f"\n   Instance {result['instance']}: {status_icon} {result['status']}")
        print(f"   Time: {result['elapsed']:.1f}s")
        if result["status"] == "success":
            print(f"   Response: {result['response']}")
            success_count += 1
        else:
            print(f"   Error: {result.get('error', 'N/A')}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total instances tested: {num_instances}")
    print(f"Successful: {success_count}")
    print(f"Failed: {num_instances - success_count}")
    print(f"Total parallel time: {overall_elapsed:.1f}s")
    print(f"Average time per instance: {overall_elapsed/num_instances:.1f}s")
    
    # Final VRAM status
    print("\n[6] Final VRAM Status:")
    vram = await get_vram_status()
    if vram:
        print(f"   Used: {vram.get('used_mb', 'N/A')} MB")
        print(f"   Available: {vram.get('available_mb', 'N/A')} MB")
    
    return success_count == num_instances


async def stress_test(num_requests: int = 10):
    """Stress test - fire many concurrent requests."""
    print(f"\n{'='*60}")
    print(f"STRESS TEST - {num_requests} Concurrent Requests")
    print(f"{'='*60}")
    
    if not await check_proxy_health():
        print("ERROR: Proxy is not running")
        return False
    
    print(f"\nLaunching {num_requests} concurrent chat completion requests...")
    
    tasks = []
    for i in range(num_requests):
        instance = i % MAX_INSTANCES
        task = asyncio.create_task(chat_completion(instance, TEST_COMPLETIONS[i % len(TEST_COMPLETIONS)]))
        tasks.append(task)
    
    overall_start = time.time()
    results = await asyncio.gather(*tasks)
    overall_elapsed = time.time() - overall_start
    
    success_count = sum(1 for r in results if r["status"] == "success")
    
    print(f"\n{'='*60}")
    print(f"STRESS TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total requests: {num_requests}")
    print(f"Successful: {success_count}")
    print(f"Failed: {num_requests - success_count}")
    print(f"Total time: {overall_elapsed:.1f}s")
    print(f"Requests/sec: {num_requests/overall_elapsed:.2f}")
    
    return True


async def continuous_load_test(duration_seconds: int = 60, concurrency: int = 3):
    """Continuous load test for a duration."""
    print(f"\n{'='*60}")
    print(f"CONTINUOUS LOAD TEST")
    print(f"{'='*60}")
    print(f"Duration: {duration_seconds}s")
    print(f"Concurrency: {concurrency} parallel requests")
    
    if not await check_proxy_health():
        print("ERROR: Proxy is not running")
        return False
    
    start_time = time.time()
    request_count = 0
    success_count = 0
    error_count = 0
    
    async def worker(worker_id: int):
        nonlocal request_count, success_count, error_count
        while time.time() - start_time < duration_seconds:
            instance = worker_id % MAX_INSTANCES
            result = await chat_completion(instance, TEST_COMPLETIONS[worker_id % len(TEST_COMPLETIONS)])
            request_count += 1
            if result["status"] == "success":
                success_count += 1
            else:
                error_count += 1
            await asyncio.sleep(0.1)  # Brief pause between requests
    
    tasks = [asyncio.create_task(worker(i)) for i in range(concurrency)]
    await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"LOAD TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total requests: {request_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Requests/sec: {request_count/elapsed:.2f}")
    print(f"Success rate: {success_count/request_count*100:.1f}%")
    
    return True


def print_usage():
    """Print usage information."""
    print("\nUsage:")
    print("  python test_parallel_chat.py parallel [num_instances]")
    print("                              - Test parallel instances (default: 3)")
    print("  python test_parallel_chat.py stress [num_requests]")
    print("                              - Stress test with concurrent requests (default: 10)")
    print("  python test_parallel_chat.py load [duration] [concurrency]")
    print("                              - Continuous load test (default: 60s, 3 parallel)")
    print("  python test_parallel_chat.py all")
    print("                              - Run all tests")
    print("\nExamples:")
    print("  python test_parallel_chat.py parallel 5")
    print("  python test_parallel_chat.py stress 20")
    print("  python test_parallel_chat.py load 120 5")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LLAMA.CPP PROXY - PARALLEL CHAT COMPLETION TEST")
    print("="*60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "parallel":
            num = int(sys.argv[2]) if len(sys.argv) > 2 else 3
            asyncio.run(test_parallel_instances(num))
        elif command == "stress":
            num = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            asyncio.run(stress_test(num))
        elif command == "load":
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            conc = int(sys.argv[3]) if len(sys.argv) > 3 else 3
            asyncio.run(continuous_load_test(duration, conc))
        elif command == "all":
            print("\nRunning ALL tests...\n")
            asyncio.run(test_parallel_instances(3))
            asyncio.run(stress_test(5))
            asyncio.run(continuous_load_test(30, 2))
        else:
            print(f"Unknown command: {command}")
            print_usage()
    else:
        print_usage()
