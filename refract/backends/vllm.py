"""vLLM backend for REFRACT — SKELETON.

Status: every method raises NotImplementedError with concrete pointers
to vLLM's Python API. Interface contract matches Backend so the rest of
REFRACT doesn't change when this is filled in.

What's needed to plug this in
-----------------------------

  pip install vllm

vLLM is server-mode preferred; the Python `LLM` class can also drive
single-process inference. For each method:

  1. ``run_completion``: instantiate ``vllm.LLM(model=str(model))`` once
     per backend instance (cache across calls), apply the model's chat
     template via ``llm.get_tokenizer().apply_chat_template``, then
     ``llm.generate(prompt, sampling_params)`` with greedy params
     (temperature=0). Return decoded text.

  2. ``run_completion_trajectory``: use the ``logprobs`` field on the
     ``RequestOutput`` to recover decode-time token IDs. Each
     ``CompletionOutput.token_ids`` is a list of ints — exactly what
     REFRACT trajectory wants. No binary patches needed; vLLM gives
     this natively.

  3. ``run_kld``: tokenize the corpus, run two vllm.LLM forwards (one
     per KV config) capturing logprobs at every position via
     ``SamplingParams(prompt_logprobs=N)``, compute
     ``KL(P_ref || P_cand)``. Note: vLLM has no built-in KL-divergence
     analog so this is bespoke (~50 lines).

  4. ``tokenize_to_ids``: ``llm.get_tokenizer().encode(text)``.

  5. ``detect_thinking_mode``: default base probe works as-is.

KV-config translation
---------------------

vLLM's KV quantization is exposed via ``LLM(kv_cache_dtype="...")`` and
related constructors. Map llama.cpp's ``ctk/ctv`` syntax onto vLLM's
``kv_cache_dtype`` (fp8_e5m2, fp8_e4m3, etc.). For TurboQuant-specific
schemes, the implementer should pull from TheTom/vllm
``feature/turboquant-kv-cache``.

GPU vs CPU
----------

vLLM defaults to GPU; on a Mac, the CPU backend is slow but functional.
For matrix runs, use a Linux box with CUDA/ROCm. The selftest should
detect platform and warn if running CPU-only on a non-trivial matrix.

Reference implementation skeleton
---------------------------------

```python
# import inside methods so users without vllm installed don't pay startup cost
import vllm

class VLLMBackend(Backend):
    _llm_cache = {}  # keyed by (model_path, kv_config_str) → LLM instance

    def _get_llm(self, model, kv_config_str):
        key = (str(model), kv_config_str)
        if key not in self._llm_cache:
            self._llm_cache[key] = vllm.LLM(
                model=str(model),
                kv_cache_dtype=self._translate_kv(kv_config_str),
                ...
            )
        return self._llm_cache[key]

    def run_completion(self, *, model, prompt, kv_config_str, ...):
        llm = self._get_llm(model, kv_config_str)
        sampling = vllm.SamplingParams(temperature=temperature, max_tokens=n_predict)
        if apply_chat_template:
            tok = llm.get_tokenizer()
            messages = [{"role": "user", "content": prompt}]
            if system:
                messages.insert(0, {"role": "system", "content": system})
            templated = tok.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)
        else:
            templated = prompt
        outs = llm.generate([templated], sampling)
        return CompletionResult(text=outs[0].outputs[0].text, n_tokens=...)
```
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import Backend, BackendCapabilityError, CompletionResult, KLDResult, TrajectoryResult


_NOT_IMPL_MSG = (
    "VLLMBackend is a v0.3.1 SKELETON. The interface is pinned but "
    "implementation is pending. See refract/backends/vllm.py docstring "
    "for a step-by-step implementation guide using vllm. Until then, "
    "set REFRACT_BACKEND=llamacpp or pass --backend llamacpp."
)


class VLLMBackend(Backend):
    name = "vllm"

    def run_completion(self, **_kwargs) -> CompletionResult:
        raise NotImplementedError(_NOT_IMPL_MSG)

    def run_completion_trajectory(self, **_kwargs) -> TrajectoryResult:
        raise NotImplementedError(_NOT_IMPL_MSG)

    def run_kld(self, **_kwargs) -> KLDResult:
        raise NotImplementedError(_NOT_IMPL_MSG)

    def tokenize_to_ids(self, **_kwargs) -> list[int]:
        raise NotImplementedError(_NOT_IMPL_MSG)
