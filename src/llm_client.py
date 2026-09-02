"""
Pluggable LLM backend used by every pipeline stage.

Swap backends via `get_llm_client(backend, ...)`:
  - "mock":      canned responses, no model, no network. Used for local
                 wiring tests on a machine with no GPU (e.g. this laptop).
  - "llama_cpp": local quantized GGUF model (e.g. Qwen2.5-Instruct) run via
                 llama-cpp-python. Intended to run ONLY inside the
                 Kaggle/Colab notebook (notebooks/kaggle_colab_runner.ipynb),
                 where a GPU and enough RAM are available.
  - "anthropic": Claude API. Defaults to Haiku (cheap) and a $2/instance
                 spend cap -- pass model="claude-sonnet-5" for a specific
                 hard-debugging attempt, not as the default. Reads
                 ANTHROPIC_API_KEY from the environment unless api_key is
                 passed explicitly. Needs `pip install anthropic`.

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


def _extract_balanced_objects(raw: str) -> list[str]:
    """Find every top-level {...} span via brace counting (not greedy regex,
    which mashes multiple objects in the text into one invalid span). Skips
    braces inside string literals so quoted '}' doesn't miscount."""
    objects = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(raw):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(raw[start : i + 1])
    return objects


def _parse_json_loose(raw: str) -> dict:
    # Prefer the LAST balanced object: models that ramble tend to correct
    # themselves and give their final answer last.
    candidates = list(reversed(_extract_balanced_objects(raw))) + [raw]
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


class SpendCapExceeded(RuntimeError):
    pass


class AnthropicLLMClient(LLMClient):
    """Claude API. anthropic is imported lazily so this module stays
    importable without the package installed until this backend is
    actually selected.

    Defaults to Haiku (cheap) rather than Sonnet -- with a single API key
    for a time-boxed build session, cost discipline matters more than
    squeezing out extra reasoning quality on every call. Pass
    model="claude-sonnet-5" explicitly for a specific hard-debugging
    attempt, not as the default.

    Enforces a hard per-instance spend cap (default $4 -- raised from an
    initial $2 once real usage data showed actual runs cost ~$0.005-0.02
    even with retries, and the session moved to experimenting with Sonnet
    at ~3x Haiku's per-token rate; still generous headroom relative to the
    ~$0.02-0.08/run this pipeline normally costs -- see
    validator_project_summary.md's own cost table) using the *actual*
    token usage the API returns, not an estimate from prompt length. This
    catches a runaway retry loop or unexpectedly huge context before it
    can eat a meaningful chunk of a $49 budget. Rates below are
    approximate placeholders -- check console.anthropic.com/settings/billing
    for current pricing if the exact number matters.
    """

    _RATES_USD_PER_MTOK = {  # (input, output)
        "claude-haiku-4-5-20251001": (1.00, 5.00),
        "claude-sonnet-5": (3.00, 15.00),
        "claude-opus-5": (15.00, 75.00),
    }
    _DEFAULT_RATE = (3.00, 15.00)  # used for any model name not in the table above

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_spend_usd: float = 4.0,
    ):
        import anthropic  # lazy import: only required for this backend

        self._client = anthropic.Anthropic(api_key=api_key)  # falls back to ANTHROPIC_API_KEY env var
        self._model = model
        self._max_spend_usd = max_spend_usd
        self.spend_usd = 0.0  # cumulative for this instance's lifetime; public so callers can log it

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        if self.spend_usd >= self._max_spend_usd:
            raise SpendCapExceeded(
                f"AnthropicLLMClient spend cap hit: ~${self.spend_usd:.4f} >= "
                f"${self._max_spend_usd:.2f} cap. Stopping before another API call. "
                f"Pass a higher max_spend_usd if this run genuinely needs more."
            )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            # thinking disabled: models with it on by default (e.g. claude-sonnet-5)
            # can burn the entire max_tokens budget on thinking and leave zero
            # room for the actual answer (a response with only a ThinkingBlock,
            # no text at all) -- these calls just need a direct JSON answer, no
            # exposed reasoning needed
            thinking={"type": "disabled"},
            # no explicit temperature: newer models reject it outright
            # ("temperature is deprecated for this model")
        )
        in_rate, out_rate = self._RATES_USD_PER_MTOK.get(self._model, self._DEFAULT_RATE)
        self.spend_usd += (
            response.usage.input_tokens * in_rate + response.usage.output_tokens * out_rate
        ) / 1_000_000
        # some models (e.g. claude-sonnet-5) can emit a ThinkingBlock before
        # the TextBlock -- content[0] isn't reliably the answer, so find the
        # actual text block(s) instead of assuming position
        text_blocks = [block.text for block in response.content if block.type == "text"]
        if not text_blocks:
            raise ValueError(f"No text block in response.content: {response.content!r}")
        return "".join(text_blocks)


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
    if backend == "anthropic":
        return AnthropicLLMClient(
            model=kwargs.get("model", "claude-haiku-4-5-20251001"),
            api_key=kwargs.get("api_key"),
            max_spend_usd=kwargs.get("max_spend_usd", 2.0),
        )
    raise ValueError(f"Unknown LLM backend: {backend!r}")
