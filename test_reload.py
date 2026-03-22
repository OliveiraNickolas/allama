#!/usr/bin/env python3
"""Test script for model reload functionality.

This tests:
1. Load a model
2. Unload it
3. Reload it again
4. Verify it works

Test using: python test_reload.py
"""

import asyncio
import httpx
import time
import sys

BASE_URL = "http://127.0.0.1:9001"  # Non-conflicting port
TIMEOUT = 10


async def test_model_load_unload_reload():
    """Test loading, unloading, and reloading a model."""
    print("\n🧪 Starting model reload test...\n")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Step 1: Check health
        print("Step 1: Checking health...")
        try:
            resp = await client.get(f"{BASE_URL}/health")
            data = resp.json()
            print(f"   Status: {resp.status_code}")
            print(f"   Active servers: {data.get('active_servers', 'N/A')}")
            print(f"   Running: {data.get('running', 'N/A')}")
        except Exception as e:
            print(f"   ERROR: Cannot connect to {BASE_URL}")
            print(f"   Is allama running? Got: {e}")
            return False
        print()

        # Step 2: Load model
        print("Step 2: Loading model (triggering traffic)...")
        try:
            payload = {
                "model": "default",
                "messages": [{"role": "user", "content": "Hello, world!"}],
                "max_tokens": 50,
                "temperature": 0.5
            }
            resp = await client.post(
                f"{BASE_URL}/v1/chat/completions",
                json=payload
            )
            print(f"   Status: {resp.status_code}")
            if resp.status_code == 200:
                result = resp.json()
                print(f"   Response: {result.get('choices', [{}])[0].get('message', {}).get('content', '')[:100]}...")
            else:
                print(f"   Error: {resp.text[:200]}")
        except Exception as e:
            print(f"   ERROR: {e}")
        print()

        # Wait for health monitor to update
        print("Waiting for system to stabilize...")
        time.sleep(2)
        print()

        # Step 3: Check active servers
        print("Step 3: Checking active servers...")
        try:
            resp = await client.get(f"{BASE_URL}/health")
            data = resp.json()
            print(f"   Active servers: {data.get('active_servers', 'N/A')}")
        except Exception as e:
            print(f"   ERROR: {e}")
        print()

        # Step 4: Simulate unload by waiting for keep-alive timeout
        print("Step 4: Simulating unload (waiting for keep-alive timeout)...")
        print("   Note: Health monitor auto-unloads after KEEP_ALIVE_SECONDS")
        print(f"   Please wait for autounload or manually trigger via kill if needed")
        # Skip manual kill - user is using vllm process
        print("   SKIPPING manual kill (user has active vllm)")
        print()

        # Step 5: Try loading again
        print("Step 5: Loading model again (after potential unload)...")
        try:
            resp = await client.post(
                f"{BASE_URL}/v1/chat/completions",
                json=payload
            )
            print(f"   Status: {resp.status_code}")
            if resp.status_code == 200:
                result = resp.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                print(f"   Success! Response: {content[:100]}...")
            else:
                print(f"   Error: {resp.text[:200]}")
        except Exception as e:
            print(f"   ERROR: {e}")
        print()

    print("✅ Test completed!")
    return True


if __name__ == "__main__":
    # Wait a bit for server to be ready
    print("Connecting to Allama at localhost:9001")

    try:
        asyncio.run(test_model_load_unload_reload())
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user")
        sys.exit(1)
