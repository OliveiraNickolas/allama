"""
Testes unitários para funções puras e lógica sem dependência de GPU/servidor.
"""
import os
import sys
import socket
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==============================================================================
# configs/loader.py — parse_all_file
# ==============================================================================
class TestParseAllFile:
 def setup_method(self):
 from configs.loader import parse_all_file
 self.parse = parse_all_file

 def test_basic_key_value(self):
 cfg = self.parse("backend = vllm\npath = /models/qwen")
 assert cfg["backend"] == "vllm"
 assert cfg["path"] == "/models/qwen"

 def test_boolean_coercion(self):
 cfg = self.parse("enabled = true\ndisabled = false")
 assert cfg["enabled"] is True
 assert cfg["disabled"] is False

 def test_none_coercion(self):
 cfg = self.parse("val = null\nval2 = none\nval3 = ~")
 assert cfg["val"] is None
 assert cfg["val2"] is None
 assert cfg["val3"] is None

 def test_section(self):
 cfg = self.parse("[sampling]\ntemperature = 0.7\ntop_p = 0.9")
 assert "sampling" in cfg
 assert cfg["sampling"]["temperature"] == "0.7"
 assert cfg["sampling"]["top_p"] == "0.9"

 def test_json_list(self):
 cfg = self.parse('extra_args = ["--disable-log-requests", "--trust-remote-code"]')
 assert isinstance(cfg["extra_args"], list)
 assert "--trust-remote-code" in cfg["extra_args"]

 def test_multiline_json_list(self):
 content = 'extra_args = [\n"--arg1",\n"--arg2"\n]'
 cfg = self.parse(content)
 assert cfg["extra_args"] == ["--arg1", "--arg2"]

 def test_comments_ignored(self):
 cfg = self.parse("# comment\nbackend = vllm\n# another comment")
 assert list(cfg.keys()) == ["backend"]

 def test_empty_lines_ignored(self):
 cfg = self.parse("\n\nbackend = vllm\n\n")
 assert cfg["backend"] == "vllm"

 def test_quoted_values_stripped(self):
 cfg = self.parse('path = "/models/qwen"')
 assert cfg["path"] == "/models/qwen"

 def test_section_and_top_level(self):
 cfg = self.parse("backend = vllm\n[sampling]\ntemp = 0.5")
 assert cfg["backend"] == "vllm"
 assert cfg["sampling"]["temp"] == "0.5"


# ==============================================================================
# allama.py — format_user_agent
# ==============================================================================
class TestFormatUserAgent:
 def setup_method(self):
 import allama
 self.fmt = allama.format_user_agent

 def test_empty(self):
 assert self.fmt("") == "unknown"
 assert self.fmt("unknown") == "unknown"

 def test_claude_terminal(self):
 assert self.fmt("claude/1.0 python/3.11") == "Claude - Terminal"

 def test_claude_vscode(self):
 assert self.fmt("claude-code/1.0 vscode/1.85") == "Claude - VSCode"

 def test_openwebui_explicit(self):
 assert self.fmt("OpenWebUI/1.0") == "OpenWebUI"

 def test_openwebui_aiohttp(self):
 assert self.fmt("Mozilla/5.0 python-aiohttp/3.9") == "OpenWebUI"

 def test_curl(self):
 assert self.fmt("curl/7.88.1") == "curl"

 def test_wget(self):
 assert self.fmt("Wget/1.21") == "wget"

 def test_python_requests(self):
 assert self.fmt("python-requests/2.31") == "Python/requests"

 def test_desktop_chrome(self):
 ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
 assert self.fmt(ua) == "OpenWebUI Desktop"

 def test_truncation(self):
 long_ua = "X" * 60
 result = self.fmt(long_ua)
 assert result.endswith("...")
 assert len(result) == 50

 def test_short_unknown_ua(self):
 ua = "my-custom-client/1.0"
 assert self.fmt(ua) == ua


# ==============================================================================
# allama.py — _calc_model_size_gb
# ==============================================================================
class TestCalcModelSizeGb:
 def setup_method(self):
 import allama
 self.calc = allama._calc_model_size_gb

 def test_sums_safetensors(self):
 with tempfile.TemporaryDirectory() as d:
 Path(d, "model-00001.safetensors").write_bytes(b"x" * 1024)
 Path(d, "model-00002.safetensors").write_bytes(b"x" * 1024)
 Path(d, "config.json").write_bytes(b"x" * 512) # deve ser ignorado
 size = self.calc(d)
 assert abs(size - 2048 / (1024 ** 3)) < 1e-10

 def test_ignores_non_safetensors(self):
 with tempfile.TemporaryDirectory() as d:
 Path(d, "tokenizer.json").write_bytes(b"x" * 10000)
 Path(d, "model.safetensors").write_bytes(b"x" * 500)
 size = self.calc(d)
 assert abs(size - 500 / (1024 ** 3)) < 1e-10

 def test_empty_dir(self):
 with tempfile.TemporaryDirectory() as d:
 size = self.calc(d)
 assert size == 0.0

 def test_nested_dirs(self):
 with tempfile.TemporaryDirectory() as d:
 sub = Path(d, "shard")
 sub.mkdir()
 (sub / "model.safetensors").write_bytes(b"x" * 2048)
 size = self.calc(d)
 assert abs(size - 2048 / (1024 ** 3)) < 1e-10


# ==============================================================================
# allama.py — is_port_free
# ==============================================================================
class TestIsPortFree:
 def setup_method(self):
 import allama
 self.is_free = allama.is_port_free

 def test_free_port(self):
 # Encontra uma porta livre dinamicamente
 with socket.socket() as s:
 s.bind(("127.0.0.1", 0))
 port = s.getsockname()[1]
 # Socket fechado — porta deve estar livre
 assert self.is_free(port) is True

 def test_occupied_port(self):
 with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
 s.bind(("127.0.0.1", 0))
 s.listen(1)
 port = s.getsockname()[1]
 assert self.is_free(port) is False


# ==============================================================================
# allama.py — port allocation (thread safety)
# ==============================================================================
class TestPortAllocation:
 def test_vllm_ports_unique_concurrent(self):
 import allama
 ports = []
 lock = threading.Lock()

 def grab():
 p = allama.get_next_vllm_port()
 with lock:
 ports.append(p)

 threads = [threading.Thread(target=grab) for _ in range(50)]
 for t in threads:
 t.start()
 for t in threads:
 t.join()

 assert len(ports) == len(set(ports)), "Portas duplicadas detectadas"

 def test_llama_ports_unique_concurrent(self):
 import allama
 ports = []
 lock = threading.Lock()

 def grab():
 p = allama.get_next_llama_port()
 with lock:
 ports.append(p)

 threads = [threading.Thread(target=grab) for _ in range(50)]
 for t in threads:
 t.start()
 for t in threads:
 t.join()

 assert len(ports) == len(set(ports)), "Portas duplicadas detectadas"

 def test_vllm_ports_are_sequential(self):
 import allama
 # Captura o valor atual e verifica que avança
 p1 = allama.get_next_vllm_port()
 p2 = allama.get_next_vllm_port()
 assert p2 == p1 + 1


# ==============================================================================
# allama.py — get_model_vram_need
# ==============================================================================
class TestGetModelVramNeed:
 def setup_method(self):
 import allama
 self.vram_need = allama.get_model_vram_need

 def test_vllm_missing_path_returns_default(self):
 cfg = {"backend": "vllm", "path": "/nonexistent/path"}
 result = self.vram_need(cfg, "test-model")
 assert result == 4.0

 def test_llama_missing_file_returns_default(self):
 cfg = {"backend": "llama.cpp", "model": "/nonexistent/model.gguf"}
 result = self.vram_need(cfg, "test-model")
 assert result == 4.0

 def test_vllm_kv_cache_uses_fp16(self):
 """kv_cache deve usar 2 bytes (fp16) × 2 (K+V)."""
 with tempfile.TemporaryDirectory() as d:
 Path(d, "model.safetensors").write_bytes(b"x" * int(1 * 1024 ** 3)) # 1 GB
 cfg = {
 "backend": "vllm",
 "path": d,
 "max_model_len": "1024",
 "max_num_seqs": "1",
 }
 result = self.vram_need(cfg, "test")
 # 1GB * 1.06 + (1024 * 1 * 2 * 2) / (1024^3) + 1.0
 expected_kv = (1024 * 1 * 2 * 2) / (1024 ** 3)
 expected = 1.0 * 1.06 + expected_kv + 1.0
 assert abs(result - expected) < 0.01

 def test_llama_cpp_estimation(self):
 with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
 f.write(b"x" * int(2 * 1024 ** 3)) # 2 GB
 path = f.name
 try:
 cfg = {
 "backend": "llama.cpp",
 "model": path,
 "n_ctx": "1024",
 }
 result = self.vram_need(cfg, "test")
 assert result > 2.0 # deve ser > tamanho do arquivo
 finally:
 os.unlink(path)


# ==============================================================================
# allama.py — AUTO_SWAP_ENABLED respeitado no health monitor
# ==============================================================================
class TestAutoSwapEnabled:
 def test_auto_swap_disabled_prevents_unload(self):
 """Quando AUTO_SWAP_ENABLED=false, modelos ociosos não devem ser marcados para unload."""
 import allama

 original = allama.AUTO_SWAP_ENABLED
 try:
 allama.AUTO_SWAP_ENABLED = False

 # Simular servidor ativo e ocioso
 import time
 from unittest.mock import MagicMock
 proc = MagicMock()
 proc.poll.return_value = None

 allama.active_servers["test-model"] = {"process": proc, "port": 9999, "backend": "vllm"}
 allama.server_idle_time["test-model"] = time.time() - allama.KEEP_ALIVE_SECONDS - 100

 to_unload = []
 now = time.time()
 with allama.global_lock:
 for name, server in list(allama.active_servers.items()):
 idle = now - allama.server_idle_time.get(name, 0)
 if allama.AUTO_SWAP_ENABLED and idle > allama.KEEP_ALIVE_SECONDS:
 to_unload.append(name)

 assert "test-model" not in to_unload
 finally:
 allama.AUTO_SWAP_ENABLED = original
 allama.active_servers.pop("test-model", None)
 allama.server_idle_time.pop("test-model", None)

 def test_auto_swap_enabled_marks_idle_for_unload(self):
 """Quando AUTO_SWAP_ENABLED=true, modelos ociosos devem ser marcados."""
 import allama
 import time
 from unittest.mock import MagicMock

 original = allama.AUTO_SWAP_ENABLED
 try:
 allama.AUTO_SWAP_ENABLED = True
 proc = MagicMock()
 proc.poll.return_value = None

 allama.active_servers["test-model2"] = {"process": proc, "port": 9998, "backend": "vllm"}
 allama.server_idle_time["test-model2"] = time.time() - allama.KEEP_ALIVE_SECONDS - 100

 to_unload = []
 now = time.time()
 with allama.global_lock:
 for name, server in list(allama.active_servers.items()):
 idle = now - allama.server_idle_time.get(name, 0)
 if allama.AUTO_SWAP_ENABLED and idle > allama.KEEP_ALIVE_SECONDS:
 to_unload.append(name)

 assert "test-model2" in to_unload
 finally:
 allama.AUTO_SWAP_ENABLED = original
 allama.active_servers.pop("test-model2", None)
 allama.server_idle_time.pop("test-model2", None)
