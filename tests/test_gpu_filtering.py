"""
Tests for GPU filtering via ALLMA_VISIBLE_DEVICES.
Uses mocked nvidia-smi to run without real hardware.
"""
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.gpu as gpu_mod

# nvidia-smi output format: index, total_mb, free_mb
FAKE_NVIDIA_SMI_OUTPUT = "0, 24576, 20000\n1, 24576, 8000\n2, 16384, 16000\n"


def _mock_run(*args, **kwargs):
    r = MagicMock()
    r.stdout = FAKE_NVIDIA_SMI_OUTPUT
    r.returncode = 0
    return r


def test_all_gpus_no_filter():
    """Without ALLMA_VISIBLE_DEVICES, all GPUs are returned."""
    env = {k: v for k, v in os.environ.items() if k != "ALLMA_VISIBLE_DEVICES"}
    with patch.object(gpu_mod, "subprocess") as mock_sub:
        mock_sub.run.side_effect = _mock_run
        with patch.dict(os.environ, env, clear=True):
            gpus = gpu_mod.get_all_gpus()
    assert len(gpus) == 3
    assert [g["index"] for g in gpus] == [0, 1, 2]


def test_all_gpus_single_filter():
    """ALLMA_VISIBLE_DEVICES=1 returns only GPU 1."""
    with patch.object(gpu_mod, "subprocess") as mock_sub:
        mock_sub.run.side_effect = _mock_run
        with patch.dict(os.environ, {"ALLMA_VISIBLE_DEVICES": "1"}):
            gpus = gpu_mod.get_all_gpus()
    assert len(gpus) == 1
    assert gpus[0]["index"] == 1


def test_all_gpus_multi_filter():
    """ALLMA_VISIBLE_DEVICES=0,2 returns GPUs 0 and 2."""
    with patch.object(gpu_mod, "subprocess") as mock_sub:
        mock_sub.run.side_effect = _mock_run
        with patch.dict(os.environ, {"ALLMA_VISIBLE_DEVICES": "0,2"}):
            gpus = gpu_mod.get_all_gpus()
    assert len(gpus) == 2
    assert [g["index"] for g in gpus] == [0, 2]


def test_all_gpus_invalid_filter():
    """ALLMA_VISIBLE_DEVICES with non-existent GPU index returns empty list."""
    with patch.object(gpu_mod, "subprocess") as mock_sub:
        mock_sub.run.side_effect = _mock_run
        with patch.dict(os.environ, {"ALLMA_VISIBLE_DEVICES": "9"}):
            gpus = gpu_mod.get_all_gpus()
    assert len(gpus) == 0


def test_gpu_memory_values():
    """VRAM values are correctly converted from MB to GB."""
    env = {k: v for k, v in os.environ.items() if k != "ALLMA_VISIBLE_DEVICES"}
    with patch.object(gpu_mod, "subprocess") as mock_sub:
        mock_sub.run.side_effect = _mock_run
        with patch.dict(os.environ, env, clear=True):
            gpus = gpu_mod.get_all_gpus()
    gpu0 = next(g for g in gpus if g["index"] == 0)
    assert abs(gpu0["free_gb"] - 20000 / 1024) < 0.01
    assert abs(gpu0["total_gb"] - 24576 / 1024) < 0.01


def test_get_best_gpu_picks_most_free():
    """get_best_gpu returns the GPU with the most free memory."""
    # get_best_gpu calls get_free_gpu_memory which uses format: free_mb, index
    def mock_free(*args, **kwargs):
        r = MagicMock()
        r.stdout = "20000, 0\n8000, 1\n16000, 2\n"
        r.returncode = 0
        return r

    with patch.object(gpu_mod, "subprocess") as mock_sub:
        mock_sub.run.side_effect = mock_free
        best = gpu_mod.get_best_gpu()
    assert best == 0  # GPU 0 has 20000 MB free


def test_all_gpus_subprocess_error_returns_empty():
    """When nvidia-smi fails, get_all_gpus returns empty list."""
    with patch.object(gpu_mod, "subprocess") as mock_sub:
        mock_sub.run.side_effect = Exception("nvidia-smi not found")
        gpus = gpu_mod.get_all_gpus()
    assert gpus == []


def test_visible_devices_comma_with_spaces():
    """ALLMA_VISIBLE_DEVICES handles spaces around commas."""
    with patch.object(gpu_mod, "subprocess") as mock_sub:
        mock_sub.run.side_effect = _mock_run
        with patch.dict(os.environ, {"ALLMA_VISIBLE_DEVICES": "0, 2"}):
            gpus = gpu_mod.get_all_gpus()
    assert len(gpus) == 2
    assert {g["index"] for g in gpus} == {0, 2}
