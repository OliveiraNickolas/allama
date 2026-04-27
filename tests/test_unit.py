"""
Unit tests — pure logic, no GPU or server required.
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

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
        assert cfg["sampling"]["temperature"] == 0.7
        assert cfg["sampling"]["top_p"] == 0.9

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
        assert cfg["sampling"]["temp"] == 0.5

    def test_tilde_expansion(self):
        cfg = self.parse("path = ~/AI/Models/MyModel")
        assert not cfg["path"].startswith("~")
        assert cfg["path"].endswith("/AI/Models/MyModel")

    def test_integer_coercion(self):
        cfg = self.parse("max_model_len = 65536\nmax_num_seqs = 8")
        assert cfg["max_model_len"] == 65536
        assert cfg["max_num_seqs"] == 8

    def test_float_coercion(self):
        cfg = self.parse("gpu_memory_utilization = 0.90")
        assert cfg["gpu_memory_utilization"] == 0.90


# ==============================================================================
# configs/loader.py — load_models_from_configs
# ==============================================================================
class TestLoadModelsFromConfigs:
    def setup_method(self):
        from configs.loader import load_models_from_configs
        self.load = load_models_from_configs

    def test_nonexistent_dir_returns_empty(self):
        base, profiles = self.load("/nonexistent/configs/dir")
        assert base == {}
        assert profiles == {}

    def test_loads_base_with_backend(self, tmp_path):
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        (base_dir / "TestModel.allm").write_text(
            'backend = "vllm"\npath = "/models/test"\n'
        )
        base, _ = self.load(str(tmp_path))
        assert "TestModel" in base
        assert base["TestModel"]["backend"] == "vllm"

    def test_skips_base_without_backend(self, tmp_path):
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        (base_dir / "NoBackend.allm").write_text("path = /models/test\n")
        base, _ = self.load(str(tmp_path))
        assert "NoBackend" not in base

    def test_loads_profile_with_base(self, tmp_path):
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        (base_dir / "MyBase.allm").write_text('backend = "vllm"\npath = "/m"\n')

        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        (profile_dir / "MyProfile.allm").write_text(
            'name = "My:Profile"\nbase = "MyBase"\n'
        )
        base, profiles = self.load(str(tmp_path))
        assert "My:Profile" in profiles
        assert profiles["My:Profile"]["base"] == "MyBase"

    def test_skips_profile_without_base(self, tmp_path):
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        (profile_dir / "Orphan.allm").write_text('name = "Orphan"\n')
        _, profiles = self.load(str(tmp_path))
        assert "Orphan" not in profiles

    def test_uses_filename_as_name_when_no_name_field(self, tmp_path):
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        (base_dir / "UnnamedModel.allm").write_text('backend = "llama.cpp"\nmodel = "/m.gguf"\n')
        base, _ = self.load(str(tmp_path))
        assert "UnnamedModel" in base


# ==============================================================================
# core/gpu.py — _calc_model_size_gb
# ==============================================================================
class TestCalcModelSizeGb:
    def setup_method(self):
        from core.gpu import _calc_model_size_gb
        self.calc = _calc_model_size_gb

    def test_sums_safetensors(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "model-00001.safetensors").write_bytes(b"x" * 1024)
            Path(d, "model-00002.safetensors").write_bytes(b"x" * 1024)
            Path(d, "config.json").write_bytes(b"x" * 512)  # ignored
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
            assert self.calc(d) == 0.0

    def test_nested_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d, "shard")
            sub.mkdir()
            (sub / "model.safetensors").write_bytes(b"x" * 2048)
            size = self.calc(d)
            assert abs(size - 2048 / (1024 ** 3)) < 1e-10


# ==============================================================================
# core/gpu.py — get_model_vram_need
# ==============================================================================
class TestGetModelVramNeed:
    def setup_method(self):
        from core.gpu import get_model_vram_need
        self.vram_need = get_model_vram_need

    def test_vllm_missing_path_returns_default(self):
        cfg = {"backend": "vllm", "path": "/nonexistent/path"}
        assert self.vram_need(cfg, "test-model") == 4.0

    def test_llama_missing_file_returns_default(self):
        cfg = {"backend": "llama.cpp", "model": "/nonexistent/model.gguf"}
        assert self.vram_need(cfg, "test-model") == 4.0

    def test_vllm_returns_more_than_model_size(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "model.safetensors").write_bytes(b"x" * int(1 * 1024 ** 3))  # 1 GB
            cfg = {
                "backend": "vllm",
                "path": d,
                "max_model_len": "1024",
                "max_num_seqs": "1",
            }
            result = self.vram_need(cfg, "test")
            assert result > 1.0  # overhead brings it above raw model size

    def test_llama_cpp_returns_more_than_file_size(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"x" * int(2 * 1024 ** 3))  # 2 GB
            path = f.name
        try:
            cfg = {"backend": "llama.cpp", "model": path, "n_ctx": "1024"}
            result = self.vram_need(cfg, "test")
            assert result > 2.0
        finally:
            os.unlink(path)

    def test_unknown_backend_returns_default(self):
        cfg = {"backend": "unknown_backend"}
        assert self.vram_need(cfg, "test") == 4.0


# ==============================================================================
# core/gpu.py — get_all_gpus (mocked subprocess)
# ==============================================================================
class TestGetAllGpus:
    FAKE_OUTPUT = "0, 24576, 20000\n1, 24576, 8000\n2, 16384, 16000\n"

    def _mock_run(self, *args, **kwargs):
        r = MagicMock()
        r.stdout = self.FAKE_OUTPUT
        r.returncode = 0
        return r

    def test_returns_all_gpus_without_filter(self):
        import core.gpu as gpu_mod
        env = {k: v for k, v in os.environ.items() if k != "ALLMA_VISIBLE_DEVICES"}
        with patch.object(gpu_mod, "subprocess") as mock_sub:
            mock_sub.run.side_effect = self._mock_run
            with patch.dict(os.environ, env, clear=True):
                gpus = gpu_mod.get_all_gpus()
        assert len(gpus) == 3
        assert [g["index"] for g in gpus] == [0, 1, 2]

    def test_single_device_filter(self):
        import core.gpu as gpu_mod
        with patch.object(gpu_mod, "subprocess") as mock_sub:
            mock_sub.run.side_effect = self._mock_run
            with patch.dict(os.environ, {"ALLMA_VISIBLE_DEVICES": "1"}):
                gpus = gpu_mod.get_all_gpus()
        assert len(gpus) == 1
        assert gpus[0]["index"] == 1

    def test_multi_device_filter(self):
        import core.gpu as gpu_mod
        with patch.object(gpu_mod, "subprocess") as mock_sub:
            mock_sub.run.side_effect = self._mock_run
            with patch.dict(os.environ, {"ALLMA_VISIBLE_DEVICES": "0,2"}):
                gpus = gpu_mod.get_all_gpus()
        assert len(gpus) == 2
        assert [g["index"] for g in gpus] == [0, 2]

    def test_invalid_filter_returns_empty(self):
        import core.gpu as gpu_mod
        with patch.object(gpu_mod, "subprocess") as mock_sub:
            mock_sub.run.side_effect = self._mock_run
            with patch.dict(os.environ, {"ALLMA_VISIBLE_DEVICES": "9"}):
                gpus = gpu_mod.get_all_gpus()
        assert len(gpus) == 0

    def test_memory_values_converted_to_gb(self):
        import core.gpu as gpu_mod
        env = {k: v for k, v in os.environ.items() if k != "ALLMA_VISIBLE_DEVICES"}
        with patch.object(gpu_mod, "subprocess") as mock_sub:
            mock_sub.run.side_effect = self._mock_run
            with patch.dict(os.environ, env, clear=True):
                gpus = gpu_mod.get_all_gpus()
        gpu0 = next(g for g in gpus if g["index"] == 0)
        assert abs(gpu0["free_gb"] - 20000 / 1024) < 0.01
        assert abs(gpu0["total_gb"] - 24576 / 1024) < 0.01

    def test_subprocess_failure_returns_empty(self):
        import core.gpu as gpu_mod
        with patch.object(gpu_mod, "subprocess") as mock_sub:
            mock_sub.run.side_effect = Exception("nvidia-smi not found")
            gpus = gpu_mod.get_all_gpus()
        assert gpus == []


# ==============================================================================
# core/gpu.py — get_best_gpu (mocked)
# ==============================================================================
class TestGetBestGpu:
    def test_picks_gpu_with_most_free_memory(self):
        import core.gpu as gpu_mod

        def mock_run(*args, **kwargs):
            r = MagicMock()
            # format: free_mb, index
            r.stdout = "20000, 0\n8000, 1\n16000, 2\n"
            r.returncode = 0
            return r

        with patch.object(gpu_mod, "subprocess") as mock_sub:
            mock_sub.run.side_effect = mock_run
            best = gpu_mod.get_best_gpu()
        assert best == 0  # GPU 0 has most free memory

    def test_returns_0_when_no_gpus(self):
        import core.gpu as gpu_mod
        with patch.object(gpu_mod, "subprocess") as mock_sub:
            mock_sub.run.side_effect = Exception("no nvidia-smi")
            best = gpu_mod.get_best_gpu()
        assert best == 0
