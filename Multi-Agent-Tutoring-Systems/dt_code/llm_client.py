"""
llm_client.py

Thin, dependency-light wrapper around two FREE hosted LLM APIs:
  - Groq   (https://console.groq.com)
  - Gemini (https://aistudio.google.com)

Supports multiple API keys so long runs (e.g. all 516 proof states) can keep
going without waiting for a single key's rate/daily quota to reset.

Put keys in `.env` as either:
    GROQ_API_KEY=key1
    GROQ_API_KEYS=key2,key3          # comma/whitespace-separated extras
    GROQ_API_KEY_2=...               # or numbered keys
    GROQ_API_KEY_3=...

Same pattern for Gemini: GOOGLE_API_KEY / GOOGLE_API_KEYS / GOOGLE_API_KEY_N.

On a short-lived 429, the client rotates to the next key immediately.
On a daily quota hit for one key+model, that key is marked exhausted for that
model and the next key is tried. Only when every key is exhausted does the
caller see DailyQuotaExceeded.
"""

import os
import re
import time
import json
import threading
import requests

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_MISTRAL_MODEL = "mistral-large-latest"


class LLMError(Exception):
    pass


class DailyQuotaExceeded(LLMError):
    """Raised when every configured key for this provider has hit its daily
    cap for the requested model. Retrying cannot fix this until quota resets."""
    pass


def _mask_key(key):
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return "***"
    return f"...{key[-4:]}"


def _split_keys(raw):
    if not raw:
        return []
    parts = re.split(r"[\s,;]+", raw.strip())
    return [p for p in parts if p]


def _collect_keys(single_var, multi_var):
    """
    Gather unique keys from:
      - single var like GROQ_API_KEY
      - multi var like GROQ_API_KEYS (comma/whitespace-separated)
      - numbered vars like GROQ_API_KEY_1, GROQ_API_KEY_2, ...
    """
    keys = []
    seen = set()

    def add(key):
        key = (key or "").strip()
        if key and key not in seen:
            seen.add(key)
            keys.append(key)

    add(os.environ.get(single_var, ""))
    for k in _split_keys(os.environ.get(multi_var, "")):
        add(k)
    for i in range(1, 64):
        add(os.environ.get(f"{single_var}_{i}", ""))

    return keys


def load_provider_keys(provider):
    if provider == "groq":
        return _collect_keys("GROQ_API_KEY", "GROQ_API_KEYS")
    if provider == "gemini":
        return _collect_keys("GOOGLE_API_KEY", "GOOGLE_API_KEYS")
    if provider == "mistral":
        return _collect_keys("MISTRAL_API_KEY", "MISTRAL_API_KEYS")
    raise ValueError(f"Unknown provider: {provider}")



class ApiKeyPool:
    """Round-robin key pool with per-(key, model) daily-quota blacklisting."""

    def __init__(self, provider):
        self.provider = provider
        self.keys = load_provider_keys(provider)
        self._idx = 0
        self._lock = threading.Lock()
        # (key, model) pairs that hit daily quota
        self._daily_exhausted = set()

    def available(self, model):
        return [k for k in self.keys if (k, model) not in self._daily_exhausted]

    def acquire(self, model, reason=None):
        """Return the next usable key (round-robin), or None if all exhausted."""
        with self._lock:
            if not self.keys:
                return None
            usable_n = len(self.available(model))
            if usable_n == 0:
                return None
            for _ in range(len(self.keys)):
                key = self.keys[self._idx % len(self.keys)]
                self._idx = (self._idx + 1) % len(self.keys)
                if (key, model) not in self._daily_exhausted:
                    if reason:
                        print(f"  [{self.provider}] {reason}; using key {_mask_key(key)} "
                              f"({usable_n} keys usable for this model)")
                    return key
            return None

    def mark_daily_exhausted(self, key, model):
        with self._lock:
            self._daily_exhausted.add((key, model))
            left = len(self.available(model))
            print(f"  [{self.provider}] daily quota hit on key {_mask_key(key)} "
                  f"for model '{model}' -- {left} key(s) left for this model")
            return left

    def summary(self):
        return f"{len(self.keys)} key(s) loaded for {self.provider}"


_POOLS = {}
_POOLS_LOCK = threading.Lock()


def get_key_pool(provider):
    with _POOLS_LOCK:
        pool = _POOLS.get(provider)
        if pool is None:
            pool = ApiKeyPool(provider)
            _POOLS[provider] = pool
            if pool.keys:
                print(f"  [{provider}] loaded {len(pool.keys)} API key(s) for rotation: "
                      + ", ".join(_mask_key(k) for k in pool.keys))
        return pool


def reset_key_pools():
    """Test helper / force reload after editing env vars mid-process."""
    with _POOLS_LOCK:
        _POOLS.clear()


def _is_daily_quota_error(resp_text):
    t = resp_text.lower()
    return ("tokens per day" in t or "requests per day" in t
            or " tpd" in t or " rpd" in t or "daily" in t
            or "quota exceeded" in t or "exceeded your current quota" in t)


def _call_mistral(
    prompt,
    model=DEFAULT_MISTRAL_MODEL,
    temperature=0.2,
    max_tokens=2048,
    retries=3,
):
    """Call Mistral using the thread-safe multi-key pool.

    Behavior:

    - Round-robin across configured keys.
    - Rotate immediately on short-lived 429.
    - Mark a key exhausted for the requested model on daily quota.
    - Continue with remaining keys.
    - Raise DailyQuotaExceeded only when all keys are exhausted.
    """

    pool = get_key_pool(
        "mistral"
    )

    if not pool.keys:

        raise LLMError(
            "No Mistral API keys found. "
            "Set MISTRAL_API_KEY and/or "
            "MISTRAL_API_KEYS "
            "(comma-separated) in .env."
        )

    body = {

        "model":
            model,

        "messages": [
            {
                "role":
                    "user",

                "content":
                    prompt,
            }
        ],

        "temperature":
            temperature,

        "max_tokens":
            max_tokens,
    }

    # One attempt per key, plus additional
    # complete rotations.

    max_attempts = max(
        retries * max(1, len(pool.keys)),
        len(pool.keys),
    )

    last_err = None

    keys_tried_this_cycle = set()

    for attempt in range(
        max_attempts
    ):

        reason = (
            "rate limited"
            if keys_tried_this_cycle
            else None
        )

        api_key = pool.acquire(
            model,
            reason=reason,
        )

        if not api_key:

            raise DailyQuotaExceeded(

                "All Mistral API keys exhausted "
                f"daily quota for model '{model}'. "
                "Add more keys to MISTRAL_API_KEYS "
                "or wait for quota reset."
            )

        headers = {

            "Authorization":
                f"Bearer {api_key}",

            "Content-Type":
                "application/json",
        }

        try:

            resp = requests.post(

                MISTRAL_URL,

                headers=headers,

                json=body,

                timeout=90,
            )

        except requests.RequestException as exc:

            last_err = str(exc)

            # Network errors should allow another
            # configured key to be attempted.

            keys_tried_this_cycle.add(
                api_key
            )

            continue

        # ====================================================
        # SUCCESS
        # ====================================================

        if resp.status_code == 200:

            try:

                data = resp.json()

                content = (
                    data["choices"][0]
                    ["message"]["content"]
                )

            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:

                raise LLMError(
                    "Mistral returned an unexpected "
                    f"response format: {exc}"
                )

            if not content:

                finish_reason = (
                    data["choices"][0]
                    .get(
                        "finish_reason"
                    )
                )

                raise LLMError(
                    "Mistral returned empty content "
                    f"(finish_reason={finish_reason})"
                )

            return content

        # ====================================================
        # RATE LIMIT / QUOTA
        # ====================================================

        if resp.status_code == 429:

            last_err = resp.text

            # ------------------------------------------------
            # Daily quota
            # ------------------------------------------------

            if _is_daily_quota_error(
                resp.text
            ):

                left = (
                    pool.mark_daily_exhausted(
                        api_key,
                        model,
                    )
                )

                if left == 0:

                    raise DailyQuotaExceeded(

                        "All Mistral API keys exhausted "
                        f"daily quota for model '{model}': "
                        f"{resp.text}"
                    )

                # Try next key immediately.

                keys_tried_this_cycle.clear()

                continue

            # ------------------------------------------------
            # Short-lived rate limit
            # ------------------------------------------------

            keys_tried_this_cycle.add(
                api_key
            )

            usable = pool.available(
                model
            )

            if (
                len(keys_tried_this_cycle)
                >= len(usable)
                and len(usable) > 0
            ):

                wait = min(
                    3,
                    1
                    + attempt
                    // max(
                        1,
                        len(usable),
                    ),
                )

                print(
                    f"  [mistral] all "
                    f"{len(usable)} key(s) "
                    f"rate-limited once; "
                    f"brief wait {wait}s..."
                )

                time.sleep(
                    wait
                )

                keys_tried_this_cycle.clear()

            continue

        # ====================================================
        # OTHER API ERROR
        # ====================================================

        raise LLMError(

            "Mistral API error "
            f"{resp.status_code}: "
            f"{resp.text}"
        )

    raise LLMError(

        "Mistral API failed after "
        f"{max_attempts} attempts: "
        f"{last_err}"
    )


def _call_groq(prompt, model=DEFAULT_GROQ_MODEL, temperature=0.2, max_tokens=2048, retries=3):
    """
    openai/gpt-oss-* are reasoning models on Groq: they spend part of the
    token budget on hidden chain-of-thought. Cap reasoning_effort="low" so
    content is not empty.
    """
    pool = get_key_pool("groq")
    if not pool.keys:
        raise LLMError(
            "No Groq API keys found. Set GROQ_API_KEY and/or GROQ_API_KEYS "
            "(comma-separated) in .env. Get free keys at https://console.groq.com"
        )

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"

    # Allow one attempt per key, then a couple of full rotations for RPM 429s.
    max_attempts = max(retries * max(1, len(pool.keys)), len(pool.keys))
    last_err = None
    keys_tried_this_cycle = set()

    for attempt in range(max_attempts):
        reason = "rate limited" if keys_tried_this_cycle else None
        api_key = pool.acquire(model, reason=reason)
        if not api_key:
            raise DailyQuotaExceeded(
                f"All Groq API keys exhausted daily quota for model '{model}'. "
                f"Add more keys to GROQ_API_KEYS or wait for reset."
            )

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = requests.post(GROQ_URL, headers=headers, json=body, timeout=60)

        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if not content:
                finish_reason = data["choices"][0].get("finish_reason")
                reasoning_preview = (data["choices"][0]["message"].get("reasoning") or "")[:200]
                raise LLMError(
                    f"Groq returned empty content (finish_reason={finish_reason}) -- "
                    f"likely ran out of max_tokens during reasoning before writing the "
                    f"final answer. Consider raising max_tokens or lowering reasoning_effort "
                    f"further. Reasoning preview: {reasoning_preview!r}"
                )
            return content

        if resp.status_code == 429:
            last_err = resp.text
            if _is_daily_quota_error(resp.text):
                left = pool.mark_daily_exhausted(api_key, model)
                if left == 0:
                    raise DailyQuotaExceeded(
                        f"All Groq API keys exhausted daily quota for model '{model}': {resp.text}"
                    )
                keys_tried_this_cycle.clear()
                continue

            # Short-lived per-minute limit: rotate key immediately instead of sleeping.
            keys_tried_this_cycle.add(api_key)
            usable = pool.available(model)
            if len(keys_tried_this_cycle) >= len(usable) and len(usable) > 0:
                wait = min(3, 1 + attempt // max(1, len(usable)))
                print(f"  [groq] all {len(usable)} key(s) rate-limited once; brief wait {wait}s...")
                time.sleep(wait)
                keys_tried_this_cycle.clear()
            continue

        raise LLMError(f"Groq API error {resp.status_code}: {resp.text}")

    raise LLMError(f"Groq API failed after {max_attempts} attempts: {last_err}")


def _call_gemini(prompt, model=DEFAULT_GEMINI_MODEL, temperature=0.2, max_tokens=1536, retries=3):
    pool = get_key_pool("gemini")
    if not pool.keys:
        raise LLMError(
            "No Gemini API keys found. Set GOOGLE_API_KEY and/or GOOGLE_API_KEYS "
            "in .env. Get free keys at https://aistudio.google.com"
        )

    url = GEMINI_URL_TMPL.format(model=model)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }

    max_attempts = max(retries * max(1, len(pool.keys)), len(pool.keys))
    last_err = None
    keys_tried_this_cycle = set()

    for attempt in range(max_attempts):
        reason = "rate limited" if keys_tried_this_cycle else None
        api_key = pool.acquire(model, reason=reason)
        if not api_key:
            raise DailyQuotaExceeded(
                f"All Gemini API keys exhausted daily quota for model '{model}'. "
                f"Add more keys to GOOGLE_API_KEYS or wait for reset."
            )

        resp = requests.post(url, params={"key": api_key}, json=body, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

        if resp.status_code == 429:
            last_err = resp.text
            if _is_daily_quota_error(resp.text):
                left = pool.mark_daily_exhausted(api_key, model)
                if left == 0:
                    raise DailyQuotaExceeded(
                        f"All Gemini API keys exhausted daily quota for model '{model}': {resp.text}"
                    )
                keys_tried_this_cycle.clear()
                continue

            keys_tried_this_cycle.add(api_key)
            usable = pool.available(model)
            if len(keys_tried_this_cycle) >= len(usable) and len(usable) > 0:
                wait = min(3, 1 + attempt // max(1, len(usable)))
                print(f"  [gemini] all {len(usable)} key(s) rate-limited once; brief wait {wait}s...")
                time.sleep(wait)
                keys_tried_this_cycle.clear()
            continue

        raise LLMError(f"Gemini API error {resp.status_code}: {resp.text}")

    raise LLMError(f"Gemini API failed after {max_attempts} attempts: {last_err}")


def call_llm(prompt, provider="mistral", model=None, temperature=0.2, max_tokens=1536):
    """Unified entry point. provider in {'mistral', 'groq', 'gemini'}."""
    if provider == "mistral":
        return _call_mistral(prompt, model=model or DEFAULT_MISTRAL_MODEL,
                              temperature=temperature, max_tokens=max_tokens)
    if provider == "groq":
        return _call_groq(prompt, model=model or DEFAULT_GROQ_MODEL,
                           temperature=temperature, max_tokens=max_tokens)
    if provider == "gemini":
        return _call_gemini(prompt, model=model or DEFAULT_GEMINI_MODEL,
                             temperature=temperature, max_tokens=max_tokens)
    raise ValueError(f"Unknown provider: {provider}")


def extract_json_object(text):
    """
    Robust JSON extraction -- handles raw JSON, ```json fences, or JSON
    embedded in extra text. Strips <think>...</think> reasoning blocks first
    so stray braces inside thoughts cannot break the greedy regex.
    """
    if not text:
        return None
    if isinstance(text, dict):
        return text

    s = text.strip()
    s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    json_block_start = s.find("```json")
    if json_block_start != -1:
        block_end = s.find("```", json_block_start + 6)
        if block_end != -1:
            fenced_block = s[json_block_start:block_end]
            start = fenced_block.find("{")
            end = fenced_block.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(fenced_block[start:end + 1].strip())
                except Exception:
                    pass

    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None

if __name__ == "__main__":

    from utils.env_loader import load_env

    load_env()

    print(
        "=== API KEY POOLS ==="
    )

    for provider in (
        "mistral",
        "groq",
        "gemini",
    ):

        try:

            pool = get_key_pool(
                provider
            )

            print(
                f"{provider}: "
                f"{pool.summary()}"
            )

        except Exception as exc:

            print(
                f"{provider}: "
                f"ERROR - {exc}"
            )

    # --------------------------------------------------------
    # Optional live Mistral test
    # --------------------------------------------------------

    try:

        pool = get_key_pool(
            "mistral"
        )

        if pool.keys:

            out = call_llm(

                'Reply with exactly this JSON: {"ok": true}',

                provider="mistral",

                model=DEFAULT_MISTRAL_MODEL,

                temperature=0.0,

                max_tokens=100,
            )

            print(
                "\nMistral response:"
            )

            print(out)

            print(
                "\nParsed:"
            )

            print(
                extract_json_object(out)
            )

        else:

            print(
                "\nNo Mistral keys configured; "
                "skipping live test."
            )

    except Exception as exc:

        print(
            f"\nMistral test failed: "
            f"{type(exc).__name__}: {exc}"
        )
