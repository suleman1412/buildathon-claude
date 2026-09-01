"""
Pluggable LLM backend used by every pipeline stage.

Swap backends via `get_llm_client(backend, ...)`:
  - "mock":      canned responses, no model, no network. Used for local
                 wiring tests on a machine with no GPU (e.g. this laptop).
  - "llama_cpp": local quantized GGUF model (e.g. Qwen2.5-Instruct) run via
                 llama-cpp-python. Intended to run ONLY inside the
                 Kaggle/Colab notebook (notebooks/kaggle_colab_runner.ipynb),
                 where a GPU and enough RAM are available.
  - "anthropic": placeholder for swapping to the Claude API once keys are
                 available. Not implemented in v0.

Every pipeline stage talks to LLMClient.complete() / .complete_json() only,
so changing the backend never touches pipeline logic.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        ...

    def complete_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> dict:
        raw = self.complete(system_prompt, user_prompt, max_tokens=max_tokens)
        return _parse_json_loose(raw)


def _escape_stray_backslashes(s: str) -> str:
    """Small local models sometimes emit invalid JSON escapes (e.g. a
    hyphenation artifact like 'LAN-\\GUAGE' from PDF text, where \\G isn't
    a valid JSON escape). Double any backslash not already starting a
    legal escape sequence so json.loads can parse it."""
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s)


def _parse_json_loose(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    candidates = [raw] + ([match.group(0)] if match else [])
    candidates += [_escape_stray_backslashes(c) for c in candidates]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON from LLM response:\n{raw!r}")


class MockLLMClient(LLMClient):
    """Returns pre-queued responses in order. For local, model-free testing
    of pipeline wiring only -- never used for real extraction or eval."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self._responses:
            raise RuntimeError("MockLLMClient ran out of queued responses")
        return self._responses.pop(0)


class LlamaCppLLMClient(LLMClient):
    """Local quantized GGUF model via llama-cpp-python.

    Instantiate this ONLY inside the Kaggle/Colab notebook where
    llama-cpp-python is installed and a GPU is available. llama_cpp is
    imported lazily so merely importing this module elsewhere (e.g. on a
    laptop with no GPU) never pulls it in.
    """

    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = -1, verbose: bool = False):
        from llama_cpp import Llama  # lazy import: only required on Kaggle/Colab

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,  # -1 = offload all layers to GPU if available
            verbose=verbose,
        )

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        result = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return result["choices"][0]["message"]["content"]


def get_llm_client(backend: str, **kwargs) -> LLMClient:
    if backend == "mock":
        return MockLLMClient(kwargs.get("responses", []))
    if backend == "llama_cpp":
        return LlamaCppLLMClient(
            model_path=kwargs["model_path"],
            n_ctx=kwargs.get("n_ctx", 8192),
            n_gpu_layers=kwargs.get("n_gpu_layers", -1),
            verbose=kwargs.get("verbose", False),
        )
    raise ValueError(f"Unknown LLM backend: {backend!r}")
