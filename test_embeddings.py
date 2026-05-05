#!/usr/bin/env python3
"""
Test script for llama.cpp embeddings proxy.
Tests all configured embedding models sequentially.
"""
import time
import httpx

PROXY_URL = "http://localhost:8001"
MODELS = [
    "bge-m3",
    "bge-large-en-v1.5",
    "bge-base-en-v1.5",
    "all-MiniLM-L6-v2",
    "jina-embeddings-v2-base-en",
    "nomic-embed-text-v1.5"
]

TEST_PROMPTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Natural language processing enables computers to understand human language",
    "Embeddings are vector representations of text"
]


def test_model(model_name: str) -> dict:
    """Test a single model with multiple prompts."""
    results = {"model": model_name, "tests": [], "success": 0, "failed": 0}
    
    for i, prompt in enumerate(TEST_PROMPTS):
        try:
            start = time.time()
            response = httpx.post(
                f"{PROXY_URL}/v1/embeddings",
                json={"model": model_name, "input": prompt},
                timeout=120.0
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                embedding_dim = len(data.get("data", [{}])[0].get("embedding", []))
                results["tests"].append({
                    "prompt": prompt[:30] + "..." if len(prompt) > 30 else prompt,
                    "status": "OK",
                    "dim": embedding_dim,
                    "time_ms": round(elapsed * 1000, 2)
                })
                results["success"] += 1
            else:
                results["tests"].append({
                    "prompt": prompt[:30] + "...",
                    "status": f"HTTP {response.status_code}",
                    "error": response.text[:100]
                })
                results["failed"] += 1
        except Exception as e:
            results["tests"].append({
                "prompt": prompt[:30] + "...",
                "status": "ERROR",
                "error": str(e)[:100]
            })
            results["failed"] += 1
    
    return results


def main():
    print("=" * 70)
    print("Embeddings Proxy Test Suite")
    print("=" * 70)
    print(f"Proxy: {PROXY_URL}")
    print(f"Models: {len(MODELS)}")
    print(f"Prompts per model: {len(TEST_PROMPTS)}")
    print("=" * 70)
    
    # Check health
    try:
        resp = httpx.get(f"{PROXY_URL}/health", timeout=10)
        print(f"\n✓ Proxy health: {resp.json()}")
    except Exception as e:
        print(f"\n✗ Proxy health check failed: {e}")
        return
    
    # List models
    try:
        resp = httpx.get(f"{PROXY_URL}/v1/models", timeout=10)
        models = resp.json().get("data", [])
        print(f"✓ Available models: {[m['id'] for m in models]}")
    except Exception as e:
        print(f"✗ Failed to list models: {e}")
    
    print("\n" + "=" * 70)
    print("Running tests...")
    print("=" * 70)
    
    total_success = 0
    total_failed = 0
    
    for model in MODELS:
        print(f"\n[{MODELS.index(model) + 1}/{len(MODELS)}] Testing: {model}")
        print("-" * 50)
        
        result = test_model(model)
        
        for test in result["tests"]:
            status = "✓" if test["status"] == "OK" else "✗"
            if test["status"] == "OK":
                print(f"  {status} {test['prompt']:35} dim={test['dim']:4} time={test['time_ms']:7.2f}ms")
            else:
                print(f"  {status} {test['prompt']:35} {test.get('status', 'ERROR')}")
        
        print(f"  Result: {result['success']}/{len(TEST_PROMPTS)} passed")
        total_success += result["success"]
        total_failed += result["failed"]
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total tests: {total_success + total_failed}")
    print(f"Passed: {total_success}")
    print(f"Failed: {total_failed}")
    print(f"Success rate: {total_success / (total_success + total_failed) * 100:.1f}%")
    
    # Check llama-server processes
    print("\n" + "=" * 70)
    print("Running llama-server instances:")
    print("=" * 70)


if __name__ == "__main__":
    main()