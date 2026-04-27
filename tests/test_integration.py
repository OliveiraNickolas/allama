"""
Integration tests — require allma running at http://127.0.0.1:9000.
Tests real model loading with vLLM and llama.cpp backends.

Run only when the server is up:
    allma serve && pytest tests/test_integration.py -v -s
"""
import time
import pytest

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

BASE_URL = "http://127.0.0.1:9000"
LOAD_TIMEOUT = 600   # 10 min — large model loading
SHORT_TIMEOUT = 30

PROMPT_SIMPLE = [{"role": "user", "content": "Responda apenas: OK"}]


# ==============================================================================
# Helpers
# ==============================================================================
def health_check(client: "httpx.Client") -> dict:
    r = client.get(f"{BASE_URL}/health", timeout=SHORT_TIMEOUT)
    r.raise_for_status()
    return r.json()


def load_and_query(client: "httpx.Client", model: str, messages: list, max_tokens: int = 256) -> "httpx.Response":
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    return client.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=LOAD_TIMEOUT)


def extract_content(body: dict) -> str:
    msg = body["choices"][0]["message"]
    return (msg.get("content") or "").strip() or (msg.get("reasoning") or "").strip()


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture
def client():
    """HTTP client per test — server state (loaded models) persists between tests."""
    if not _HTTPX_AVAILABLE:
        pytest.skip("httpx not installed — run: pip install httpx")
    with httpx.Client() as c:
        try:
            c.get(f"{BASE_URL}/health", timeout=5)
        except Exception:
            pytest.skip("Allma is not running at http://127.0.0.1:9000 — run: allma serve")
        yield c


# ==============================================================================
# Test 1 — Server health
# ==============================================================================
class TestServerHealth:
    def test_health_endpoint(self, client):
        data = health_check(client)
        assert data["status"] == "healthy"

    def test_models_list_not_empty(self, client):
        r = client.get(f"{BASE_URL}/v1/models", timeout=SHORT_TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        assert "data" in data
        assert len(data["data"]) > 0, "No profiles found — check configs/profile/"

    def test_ps_endpoint(self, client):
        r = client.get(f"{BASE_URL}/v1/ps", timeout=SHORT_TIMEOUT)
        assert r.status_code == 200

    def test_404_on_unknown_route(self, client):
        r = client.get(f"{BASE_URL}/nonexistent", timeout=SHORT_TIMEOUT)
        assert r.status_code == 404


# ==============================================================================
# Test 2 — Model loading (vLLM, first available 27b profile)
# ==============================================================================
class TestModelLoad:
    MODEL = "Qwen3.6:27b-Instruct"

    def test_model_in_list(self, client):
        r = client.get(f"{BASE_URL}/v1/models", timeout=SHORT_TIMEOUT)
        ids = [m["id"] for m in r.json().get("data", [])]
        if self.MODEL not in ids:
            pytest.skip(f"{self.MODEL} not configured — available: {ids}")

    def test_load_and_respond(self, client):
        r = client.get(f"{BASE_URL}/v1/models", timeout=SHORT_TIMEOUT)
        ids = [m["id"] for m in r.json().get("data", [])]
        if self.MODEL not in ids:
            pytest.skip(f"{self.MODEL} not configured")

        print(f"\n  Loading {self.MODEL}...")
        t0 = time.time()
        r = load_and_query(client, self.MODEL, PROMPT_SIMPLE)
        elapsed = time.time() - t0

        print(f"  Status: {r.status_code} | Time: {elapsed:.1f}s")
        assert r.status_code == 200, f"Error: {r.text[:300]}"
        content = extract_content(r.json())
        print(f"  Response: {content[:80]!r}")
        assert content.strip(), "Empty response"

    def test_second_request_is_faster(self, client):
        r = client.get(f"{BASE_URL}/v1/models", timeout=SHORT_TIMEOUT)
        ids = [m["id"] for m in r.json().get("data", [])]
        if self.MODEL not in ids:
            pytest.skip(f"{self.MODEL} not configured")

        t0 = time.time()
        r = load_and_query(client, self.MODEL, PROMPT_SIMPLE)
        elapsed = time.time() - t0

        assert r.status_code == 200
        assert elapsed < 120, f"Second request took {elapsed:.0f}s — expected reuse < 120s"
        assert extract_content(r.json()).strip(), "Empty response on reuse"

    def test_model_active_after_load(self, client):
        r = client.get(f"{BASE_URL}/v1/models", timeout=SHORT_TIMEOUT)
        ids = [m["id"] for m in r.json().get("data", [])]
        if self.MODEL not in ids:
            pytest.skip(f"{self.MODEL} not configured")
        data = health_check(client)
        assert data["active_servers"] >= 1


# ==============================================================================
# Test 3 — OpenAI API compatibility
# ==============================================================================
class TestOpenAICompat:
    def test_chat_completions_schema(self, client):
        """Response follows OpenAI chat completions schema."""
        r = client.get(f"{BASE_URL}/v1/models", timeout=SHORT_TIMEOUT)
        models = [m["id"] for m in r.json().get("data", [])]
        if not models:
            pytest.skip("No models configured")

        model = models[0]
        resp = load_and_query(client, model, PROMPT_SIMPLE, max_tokens=64)
        if resp.status_code != 200:
            pytest.skip(f"Model {model} not loadable: {resp.text[:100]}")

        body = resp.json()
        assert "choices" in body
        assert "model" in body
        assert "usage" in body
        assert len(body["choices"]) > 0
        assert "message" in body["choices"][0]
        assert "role" in body["choices"][0]["message"]

    def test_streaming_response(self, client):
        """Streaming mode returns chunked SSE events."""
        r = client.get(f"{BASE_URL}/v1/models", timeout=SHORT_TIMEOUT)
        models = [m["id"] for m in r.json().get("data", [])]
        if not models:
            pytest.skip("No models configured")

        model = models[0]
        payload = {
            "model": model,
            "messages": PROMPT_SIMPLE,
            "max_tokens": 32,
            "stream": True,
        }
        with client.stream("POST", f"{BASE_URL}/v1/chat/completions",
                           json=payload, timeout=LOAD_TIMEOUT) as resp:
            if resp.status_code != 200:
                pytest.skip(f"Model {model} not loadable")
            chunks = []
            for line in resp.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(line)
        assert len(chunks) > 0, "No SSE chunks received in streaming mode"

    def test_invalid_model_returns_404_or_422(self, client):
        payload = {
            "model": "NonExistentModel:does-not-exist",
            "messages": PROMPT_SIMPLE,
            "max_tokens": 32,
        }
        r = client.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=SHORT_TIMEOUT)
        assert r.status_code in (404, 422, 400), f"Expected 4xx, got {r.status_code}"
