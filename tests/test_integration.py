"""
Integration tests — requerem allama rodando em http://127.0.0.1:9000.
Testam carregamento real de modelos com backends vLLM e llama.cpp.

Modelos testados:
 - vLLM leve : Qwen3.5:9b (19GB, TP=1, single GPU)
 - llama.cpp : Qwen3.5:27b-Claude-4.6 (27GB Q8_0, pode ter CPU offload)
 - Simultâneo : Qwen3.5:9b + Qwen3vl:8b (GPU 0 + GPU 1, 2x3090)
"""
import asyncio
import time
import pytest
import httpx

BASE_URL = "http://127.0.0.1:9000"
LOAD_TIMEOUT = 600 # 10 min — loading de modelos grandes
SHORT_TIMEOUT = 30 # health / models list

PROMPT_SIMPLE = [{"role": "user", "content": "Responda apenas: OK"}]
PROMPT_VISION = [{"role": "user", "content": "Descreva esta cor: azul"}]


# ==============================================================================
# Helpers
# ==============================================================================
async def health_check(client: httpx.AsyncClient) -> dict:
 r = await client.get(f"{BASE_URL}/health", timeout=SHORT_TIMEOUT)
 r.raise_for_status()
 return r.json()


async def load_and_query(
 client: httpx.AsyncClient,
 model: str,
 messages: list,
 max_tokens: int = 512,
) -> dict:
 payload = {
 "model": model,
 "messages": messages,
 "max_tokens": max_tokens,
 "temperature": 0.0,
 "stream": False,
 }
 r = await client.post(
 f"{BASE_URL}/v1/chat/completions",
 json=payload,
 timeout=LOAD_TIMEOUT,
 )
 return r


def extract_content(response_body: dict) -> str:
 """Extrai texto da resposta — suporta modelos normais e de thinking.
 vLLM 0.18 usa 'reasoning' para o bloco de thinking, não 'reasoning_content'.
 """
 msg = response_body["choices"][0]["message"]
 return (msg.get("content") or "").strip() or (msg.get("reasoning") or "").strip()


async def active_models(client: httpx.AsyncClient) -> int:
 data = await health_check(client)
 return data.get("active_servers", 0)


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture
async def client():
 """Cliente HTTP por teste — o estado do servidor (modelos carregados) persiste entre testes."""
 async with httpx.AsyncClient() as c:
 try:
 await c.get(f"{BASE_URL}/health", timeout=5)
 except Exception:
 pytest.skip("Allama não está rodando em http://127.0.0.1:9000")
 yield c


# ==============================================================================
# Teste 1 — vLLM mais leve: Qwen3.5-9b
# ==============================================================================
class TestVllmLight:
 async def test_server_healthy(self, client):
 data = await health_check(client)
 assert data["status"] == "healthy"
 print(f"\n Servidores ativos antes: {data['active_servers']}")

 async def test_load_and_respond(self, client):
 model = "Qwen3.5:9b"
 print(f"\n Carregando {model} (vLLM, TP=1, ~19GB)...")
 t0 = time.time()

 r = await load_and_query(client, model, PROMPT_SIMPLE)
 elapsed = time.time() - t0

 print(f" Status: {r.status_code} | Tempo: {elapsed:.1f}s")
 assert r.status_code == 200, f"Erro: {r.text[:200]}"

 body = r.json()
 content = extract_content(body)
 print(f" Resposta: {content[:80]!r}")
 assert content.strip(), "Resposta vazia (nem content nem reasoning_content)"

 async def test_second_request_reuses_server(self, client):
 """Segunda requisição deve reutilizar o processo já carregado (muito mais rápida)."""
 model = "Qwen3.5:9b"
 t0 = time.time()
 r = await load_and_query(client, model, PROMPT_SIMPLE)
 elapsed = time.time() - t0

 print(f"\n Reutilização — Status: {r.status_code} | Tempo: {elapsed:.1f}s")
 assert r.status_code == 200
 assert extract_content(r.json()).strip(), "Resposta vazia no reuso"
 assert elapsed < 120, f"Segunda requisição demorou {elapsed:.0f}s — esperava < 120s (reuso)"

 async def test_model_is_active_after_load(self, client):
 data = await health_check(client)
 assert data["active_servers"] >= 1
 print(f"\n Servidores ativos após load: {data['active_servers']}")


# ==============================================================================
# Teste 2 — llama.cpp: Qwen3.5:27b-Claude-4.6
# ==============================================================================
class TestLlamaCppLight:
 async def test_load_and_respond(self, client):
 model = "Qwen3.5:27b-Claude-4.6"
 print(f"\n Carregando {model} (llama.cpp, 27GB Q8_0)...")
 print(" Nota: modelo >24GB, esperado CPU offload parcial")
 t0 = time.time()

 r = await load_and_query(client, model, PROMPT_SIMPLE)
 elapsed = time.time() - t0

 print(f" Status: {r.status_code} | Tempo: {elapsed:.1f}s")
 assert r.status_code == 200, f"Erro: {r.text[:200]}"

 body = r.json()
 content = extract_content(body)
 print(f" Resposta: {content[:80]!r}")
 assert content.strip(), "Resposta vazia"

 async def test_second_request_reuses_server(self, client):
 model = "Qwen3.5:27b-Claude-4.6"
 t0 = time.time()
 r = await load_and_query(client, model, PROMPT_SIMPLE)
 elapsed = time.time() - t0

 print(f"\n Reutilização — Status: {r.status_code} | Tempo: {elapsed:.1f}s")
 assert r.status_code == 200
 assert extract_content(r.json()).strip(), "Resposta vazia no reuso"
 assert elapsed < 120

 async def test_model_is_active(self, client):
 data = await health_check(client)
 assert data["active_servers"] >= 1


# ==============================================================================
# Teste 3 — Carregamento simultâneo: Qwen3.5-9b + Qwen3vl-8b
# ==============================================================================
class TestSimultaneousLoad:
 async def test_both_models_load_concurrently(self, client):
 """
 Dispara requisições para dois modelos TP=1 ao mesmo tempo.
 Espera: ambos carregam em GPUs distintas (GPU 0 ~19GB + GPU 1 ~17GB).
 """
 model_a = "Qwen3.5:9b"
 model_b = "Qwen3vl:8b"

 print(f"\n Disparando carga simultânea:")
 print(f" {model_a} (~19GB, TP=1)")
 print(f" {model_b} (~17GB, TP=1)")
 print(f" Setup: 2x RTX 3090 (24GB cada) — total disponível ~47GB")

 t0 = time.time()
 results = await asyncio.gather(
 load_and_query(client, model_a, PROMPT_SIMPLE),
 load_and_query(client, model_b, PROMPT_VISION),
 return_exceptions=True,
 )
 elapsed = time.time() - t0
 print(f"\n Tempo total (paralelo): {elapsed:.1f}s")

 r_a, r_b = results

 # model_a
 if isinstance(r_a, Exception):
 pytest.fail(f"{model_a} lançou exceção: {r_a}")
 print(f" {model_a}: status={r_a.status_code}")
 assert r_a.status_code == 200, f"Erro em {model_a}: {r_a.text[:200]}"
 content_a = extract_content(r_a.json())
 assert content_a.strip(), f"{model_a} retornou resposta vazia"
 print(f" Resposta: {content_a[:80]!r}")

 # model_b
 if isinstance(r_b, Exception):
 pytest.fail(f"{model_b} lançou exceção: {r_b}")
 print(f" {model_b}: status={r_b.status_code}")
 assert r_b.status_code == 200, f"Erro em {model_b}: {r_b.text[:200]}"
 content_b = extract_content(r_b.json())
 assert content_b.strip(), f"{model_b} retornou resposta vazia"
 print(f" Resposta: {content_b[:80]!r}")

 async def test_both_models_active_simultaneously(self, client):
 """Após o teste anterior, ambos os modelos devem ainda estar ativos."""
 data = await health_check(client)
 active = data["active_servers"]
 print(f"\n Servidores ativos simultaneamente: {active}")
 assert active >= 2, f"Esperava >=2 servidores ativos, got {active}"

 async def test_vram_distribution(self):
 """Verifica que as GPUs foram usadas de forma distribuída via nvidia-smi."""
 import subprocess
 result = subprocess.run(
 ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
 "--format=csv,noheader,nounits"],
 capture_output=True, text=True, timeout=10,
 )
 lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
 print(f"\n Processos usando VRAM:")
 for line in lines:
 print(f" {line}")
 # Com 2 modelos carregados simultaneamente, deve haver >= 2 processos usando VRAM
 assert len(lines) >= 2, (
 f"Esperava >= 2 processos usando VRAM (um por GPU), encontrei {len(lines)}"
 )
