"""Tests for refract.backends: registry, base ABC defaults, llamacpp adapter,
mlx KV translation + import-gate, vllm skeleton."""

from __future__ import annotations

from pathlib import Path

import pytest

from refract.backends import auto_backend, get_backend
from refract.backends.base import (
    Backend,
    BackendCapabilityError,
    CompletionResult,
    KLDResult,
    TrajectoryResult,
)
from refract.backends.llamacpp import LlamaCppBackend
from refract.backends.vllm import VLLMBackend


# --- registry -------------------------------------------------------------


def test_get_backend_llamacpp():
    assert isinstance(get_backend("llamacpp"), LlamaCppBackend)


def test_get_backend_case_insensitive():
    assert isinstance(get_backend("LLAMACPP"), LlamaCppBackend)


def test_get_backend_vllm():
    assert isinstance(get_backend("vllm"), VLLMBackend)


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("nothing")


def test_auto_backend_gguf_picks_llamacpp(tmp_path, monkeypatch):
    monkeypatch.delenv("REFRACT_BACKEND", raising=False)
    p = tmp_path / "model.gguf"
    p.write_text("x")
    bk = auto_backend(p)
    assert bk.name == "llamacpp"


def test_auto_backend_directory_with_config_picks_mlx(tmp_path, monkeypatch):
    monkeypatch.delenv("REFRACT_BACKEND", raising=False)
    d = tmp_path / "mlx_model"
    d.mkdir()
    (d / "config.json").write_text("{}")
    bk = auto_backend(d)
    assert bk.name == "mlx"


def test_auto_backend_unknown_falls_back_to_vllm(tmp_path, monkeypatch):
    monkeypatch.delenv("REFRACT_BACKEND", raising=False)
    p = tmp_path / "weird.bin"
    p.write_text("x")
    bk = auto_backend(p)
    assert bk.name == "vllm"


def test_auto_backend_env_override_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("REFRACT_BACKEND", "vllm")
    p = tmp_path / "model.gguf"  # would normally be llamacpp
    p.write_text("x")
    bk = auto_backend(p)
    assert bk.name == "vllm"


# --- vllm skeleton --------------------------------------------------------


def test_vllm_run_completion_raises_not_implemented():
    bk = VLLMBackend()
    with pytest.raises(NotImplementedError):
        bk.run_completion()


def test_vllm_run_completion_trajectory_raises_not_implemented():
    bk = VLLMBackend()
    with pytest.raises(NotImplementedError):
        bk.run_completion_trajectory()


def test_vllm_run_kld_raises_not_implemented():
    bk = VLLMBackend()
    with pytest.raises(NotImplementedError):
        bk.run_kld()


def test_vllm_tokenize_to_ids_raises_not_implemented():
    bk = VLLMBackend()
    with pytest.raises(NotImplementedError):
        bk.tokenize_to_ids()


# --- base.Backend default detect_thinking_mode ----------------------------


class _ConcreteBackend(Backend):
    """Minimal concrete backend that lets the default detect_thinking_mode +
    model_metadata exercise their default code paths."""

    name = "test"

    def __init__(self, completion_text: str = "no thinking here"):
        self._text = completion_text

    def run_completion(self, **_kw) -> CompletionResult:
        return CompletionResult(text=self._text, n_tokens=0, metadata={})

    def run_completion_trajectory(self, **_kw) -> TrajectoryResult:
        return TrajectoryResult(token_ids=[], metadata={})

    def run_kld(self, **_kw) -> KLDResult:
        return KLDResult(mean_kld=0.0)

    def tokenize_to_ids(self, **_kw):
        return []


def test_base_detect_thinking_mode_no_markers():
    bk = _ConcreteBackend(completion_text="4")
    detected, markers = bk.detect_thinking_mode(model=Path("m"))
    assert detected is False
    assert markers == []


def test_base_detect_thinking_mode_finds_marker():
    bk = _ConcreteBackend(completion_text="<think>hmm</think> 4")
    detected, markers = bk.detect_thinking_mode(model=Path("m"))
    assert detected is True
    assert "<think>" in markers


def test_base_detect_thinking_mode_swallows_exception():
    class _Boom(_ConcreteBackend):
        def run_completion(self, **_kw):
            raise RuntimeError("boom")
    bk = _Boom()
    detected, markers = bk.detect_thinking_mode(model=Path("m"))
    assert detected is False
    assert markers == []


def test_base_model_metadata_default_shape():
    bk = _ConcreteBackend()
    info = bk.model_metadata(model=Path("/x/y.gguf"))
    assert info["backend"] == "test"
    assert info["model"] == "/x/y.gguf"


# --- backend dataclasses --------------------------------------------------


def test_completion_result_defaults():
    r = CompletionResult(text="hi", n_tokens=2)
    assert r.metadata == {}


def test_kld_result_defaults():
    r = KLDResult(mean_kld=0.5)
    assert r.ppl is None
    assert r.metadata == {}


def test_trajectory_result_defaults():
    r = TrajectoryResult(token_ids=[1, 2])
    assert r.metadata == {}


def test_backend_capability_error_is_runtime_error():
    assert issubclass(BackendCapabilityError, RuntimeError)


# --- llamacpp adapter (delegates to runner; mock the runner) --------------


def test_llamacpp_run_completion_delegates(monkeypatch):
    bk = LlamaCppBackend()
    captured = {}

    def fake_rc(*, model, prompt, kv, **kw):
        captured["kv_label"] = kv.label()
        captured["prompt"] = prompt
        return ("hello world", {"returncode": 0})

    monkeypatch.setattr("refract.runner.run_completion", fake_rc)
    res = bk.run_completion(
        model=Path("/m.gguf"), prompt="hi",
        kv_config_str="ctk=q8_0,ctv=q8_0",
    )
    assert res.text == "hello world"
    assert "ctk=q8_0" in captured["kv_label"]


def test_llamacpp_run_completion_trajectory_delegates(monkeypatch):
    bk = LlamaCppBackend()

    def fake_rct(*, model, prompt, kv, **kw):
        return ([1, 2, 3], {"returncode": 0, "n_tokens": 3})

    monkeypatch.setattr("refract.runner.run_completion_trajectory", fake_rct)
    res = bk.run_completion_trajectory(
        model=Path("/m.gguf"), prompt="hi", kv_config_str="ctk=f16,ctv=f16",
    )
    assert res.token_ids == [1, 2, 3]


def test_llamacpp_tokenize_delegates(monkeypatch):
    bk = LlamaCppBackend()
    monkeypatch.setattr("refract.runner.tokenize_to_ids",
                        lambda model, text, timeout=120.0: [4, 5, 6])
    assert bk.tokenize_to_ids(model=Path("/m"), text="x") == [4, 5, 6]


def test_llamacpp_model_metadata_includes_backend_name():
    bk = LlamaCppBackend()
    info = bk.model_metadata(model=Path("/m.gguf"))
    assert info["backend"] == "llamacpp"
    assert "llama_cpp_bin_dir" in info


# --- mlx backend (translate KV; import-gate) ------------------------------


def test_mlx_translate_kv_symmetric_q8():
    from refract.backends.mlx import _translate_kv_to_mlx
    out = _translate_kv_to_mlx("ctk=q8_0,ctv=q8_0")
    assert out["kv_bits"] == 8
    assert out["kv_group_size"] == 64


def test_mlx_translate_kv_f16_no_quant():
    from refract.backends.mlx import _translate_kv_to_mlx
    out = _translate_kv_to_mlx("ctk=f16,ctv=f16")
    assert out["kv_bits"] is None


def test_mlx_translate_kv_asymmetric_raises():
    from refract.backends.mlx import _translate_kv_to_mlx
    with pytest.raises(BackendCapabilityError):
        _translate_kv_to_mlx("ctk=q8_0,ctv=turbo4")


def test_mlx_translate_kv_turbo_raises():
    from refract.backends.mlx import _translate_kv_to_mlx
    with pytest.raises(BackendCapabilityError):
        _translate_kv_to_mlx("ctk=turbo4,ctv=turbo4")


def test_mlx_translate_kv_unknown_type_raises():
    from refract.backends.mlx import _translate_kv_to_mlx
    with pytest.raises(BackendCapabilityError):
        _translate_kv_to_mlx("ctk=q9_99,ctv=q9_99")


def test_mlx_require_mlx_raises_if_missing(monkeypatch):
    """When mlx-lm isn't installed, _require_mlx must raise BackendCapabilityError."""
    import sys
    from refract.backends import mlx as mlx_mod
    # Hide mlx + mlx_lm from sys.modules + meta_path so import fails.
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    with pytest.raises(BackendCapabilityError):
        mlx_mod._require_mlx()
