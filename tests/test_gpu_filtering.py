"""
Testes para o sistema de filtragem de GPUs via ALLAMA_VISIBLE_DEVICES.
Usa mock de nvidia-smi para rodar sem hardware real.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


FAKE_NVIDIA_SMI_OUTPUT = "0, 24576, 20000\n1, 24576, 8000\n2, 16384, 16000\n"


def _mock_run_nvidia(*args, **kwargs):
 from unittest.mock import MagicMock
 result = MagicMock()
 result.stdout = FAKE_NVIDIA_SMI_OUTPUT
 result.returncode = 0
 return result


def test_all_gpus_no_filter():
 """Sem ALLAMA_VISIBLE_DEVICES, todas as GPUs são retornadas."""
 import allama
 env = {k: v for k, v in os.environ.items() if k != "ALLAMA_VISIBLE_DEVICES"}
 with patch("allama.subprocess.run", side_effect=_mock_run_nvidia):
 with patch.dict(os.environ, env, clear=True):
 gpus = allama.get_all_gpus()
 assert len(gpus) == 3
 assert [g["index"] for g in gpus] == [0, 1, 2]


def test_all_gpus_single_filter():
 """ALLAMA_VISIBLE_DEVICES=1 retorna apenas GPU 1."""
 import allama
 with patch("allama.subprocess.run", side_effect=_mock_run_nvidia):
 with patch.dict(os.environ, {"ALLAMA_VISIBLE_DEVICES": "1"}):
 gpus = allama.get_all_gpus()
 assert len(gpus) == 1
 assert gpus[0]["index"] == 1


def test_all_gpus_multi_filter():
 """ALLAMA_VISIBLE_DEVICES=0,2 retorna GPUs 0 e 2."""
 import allama
 with patch("allama.subprocess.run", side_effect=_mock_run_nvidia):
 with patch.dict(os.environ, {"ALLAMA_VISIBLE_DEVICES": "0,2"}):
 gpus = allama.get_all_gpus()
 assert len(gpus) == 2
 assert [g["index"] for g in gpus] == [0, 2]


def test_all_gpus_invalid_filter():
 """ALLAMA_VISIBLE_DEVICES com GPU inexistente retorna lista vazia."""
 import allama
 with patch("allama.subprocess.run", side_effect=_mock_run_nvidia):
 with patch.dict(os.environ, {"ALLAMA_VISIBLE_DEVICES": "9"}):
 gpus = allama.get_all_gpus()
 assert len(gpus) == 0


def test_gpu_memory_values():
 """Valores de VRAM são convertidos corretamente de MB para GB."""
 import allama
 env = {k: v for k, v in os.environ.items() if k != "ALLAMA_VISIBLE_DEVICES"}
 with patch("allama.subprocess.run", side_effect=_mock_run_nvidia):
 with patch.dict(os.environ, env, clear=True):
 gpus = allama.get_all_gpus()
 gpu0 = next(g for g in gpus if g["index"] == 0)
 assert abs(gpu0["free_gb"] - 20000 / 1024) < 0.01
 assert abs(gpu0["total_gb"] - 24576 / 1024) < 0.01


def test_get_best_gpu_picks_most_free():
 """get_best_gpu retorna a GPU com mais memória livre."""
 import allama
 env = {k: v for k, v in os.environ.items() if k != "ALLAMA_VISIBLE_DEVICES"}

 def mock_free(*args, **kwargs):
 from unittest.mock import MagicMock
 r = MagicMock()
 r.stdout = "20000, 0\n8000, 1\n16000, 2\n"
 r.returncode = 0
 return r

 with patch("allama.subprocess.run", side_effect=mock_free):
 with patch.dict(os.environ, env, clear=True):
 best = allama.get_best_gpu()
 assert best == 0 # GPU 0 tem 20000 MB livres
