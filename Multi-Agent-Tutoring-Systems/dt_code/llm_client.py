#llm_client.py

import os
import re
import time
import json
import threading
import requests
from dotenv import load_dotenv
load_dotenv()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

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


def _parse_duration_to_seconds(s):
    """Parse Groq's reset-header format ('7.66s', '2m59.56s', '1h2m3s') to float seconds."""
    if s is None:
        return None
    s = str(s).strip()
    # Plain float/int seconds (e.g. retry-after: "2")
    try:
        return float(s)
    except ValueError:
        pass
    total = 0.0
    for value, unit in re.findall(r"([\d.]+)\s*(h|m|s)", s):
        value = float(value)
        if unit == "h":
            total += value * 3600
        elif unit == "m":
            total += value * 60
        else:
            total += value
    return total if total > 0 else None


def _rate_limit_wait_seconds(resp, default=5.0, cap=75.0):
    """
    Groq tells you exactly how long to wait -- use that instead of guessing.
    Prefers `retry-after`, falls back to `x-ratelimit-reset-tokens` (TPM window,
    almost always the actual bottleneck for reasoning models), then a default.
    Capped so a bad/huge header can't stall a run for an unreasonable time.
    """
    wait = _parse_duration_to_seconds(resp.headers.get("retry-after"))
    if wait is None:
        wait = _parse_duration_to_seconds(resp.headers.get("x-ratelimit-reset-tokens"))
    if wait is None:
        wait = default
    return min(max(wait, 0.5) + 0.5, cap)  # small buffer past the reset boundary, then cap


def _call_groq(prompt, model=DEFAULT_GROQ_MODEL, temperature=0.2, max_tokens=2048, retries=3):
    """
    openai/gpt-oss-* are reasoning models on Groq: they spend part of the
    token budget on hidden chain-of-thought. Cap reasoning_effort="low" so
    content is not empty.

    NOTE: Groq rate limits (RPM/TPM/RPD/TPD) apply at the ORGANIZATION level,
    not per API key -- if all your keys belong to the same Groq account, they
    share one bucket and rotating between them does not raise your effective
    throughput. Multiple keys still help if they genuinely belong to
    different accounts, or as a fallback once one key's *daily* quota
    (RPD/TPD) is separately exhausted. For the common per-minute (RPM/TPM)
    429s, the fix is pacing correctly using the response headers below, not
    key-switching.
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
        # Groq's JSON Object Mode guarantees syntactically valid JSON output
        # (it 400s instead of returning prose) as long as the prompt itself
        # says "JSON" somewhere -- your prompts.py already does via
        # Response_Format. This is what actually fixes "could not be parsed"
        # failures; smaller/instant models are far less reliable than
        # gpt-oss-20b about following a plain-English "respond with only
        # JSON" instruction on their own.
        "response_format": {"type": "json_object"},
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
            # Proactively pace if we're about to run out of TPM budget for
            # this model, so the *next* call doesn't just 429 immediately.
            remaining_tokens = resp.headers.get("x-ratelimit-remaining-tokens")
            try:
                remaining_tokens = int(remaining_tokens) if remaining_tokens is not None else None
            except ValueError:
                remaining_tokens = None
            if remaining_tokens is not None and remaining_tokens < max_tokens:
                wait = _rate_limit_wait_seconds(resp, default=5.0, cap=65.0)
                print(f"  [groq] only {remaining_tokens} tokens left in this minute's budget "
                      f"for '{model}'; pausing {wait:.1f}s before the next call...")
                time.sleep(wait)
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

            # Short-lived per-minute (RPM/TPM) limit. Since keys on the same
            # org share one bucket, cycling through them fast just wastes
            # attempts -- wait out the window Groq actually reports instead.
            keys_tried_this_cycle.add(api_key)
            usable = pool.available(model)
            wait = _rate_limit_wait_seconds(resp)
            print(f"  [groq] rate limited on key {_mask_key(api_key)} for '{model}'; "
                  f"waiting {wait:.1f}s (per Groq's reported reset window)...")
            time.sleep(wait)
            if len(keys_tried_this_cycle) >= len(usable) and len(usable) > 0:
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
def _call_mistral(
    prompt,
    model=DEFAULT_MISTRAL_MODEL,
    temperature=0.2,
    max_tokens=2048,
    retries=3,
):
    pool = get_key_pool("mistral")

    if not pool.keys:
        raise LLMError(
            "No Mistral API keys found. Set MISTRAL_API_KEY and/or "
            "MISTRAL_API_KEYS in .env."
        )

    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    max_attempts = max(
        retries * max(1, len(pool.keys)),
        len(pool.keys),
    )

    last_err = None
    keys_tried_this_cycle = set()

    for attempt in range(max_attempts):
        reason = "rate limited" if keys_tried_this_cycle else None

        api_key = pool.acquire(model, reason=reason)

        if not api_key:
            raise DailyQuotaExceeded(
                f"All Mistral API keys exhausted daily quota "
                f"for model '{model}'. "
                f"Add more keys to MISTRAL_API_KEYS or wait for reset."
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(
            MISTRAL_URL,
            headers=headers,
            json=body,
            timeout=60,
        )

        if resp.status_code == 200:
            data = resp.json()

            content = data["choices"][0]["message"]["content"]

            if not content:
                raise LLMError(
                    "Mistral returned empty content."
                )

            return content

        if resp.status_code == 429:
            last_err = resp.text

            if _is_daily_quota_error(resp.text):
                left = pool.mark_daily_exhausted(api_key, model)

                if left == 0:
                    raise DailyQuotaExceeded(
                        f"All Mistral API keys exhausted daily quota "
                        f"for model '{model}': {resp.text}"
                    )

                keys_tried_this_cycle.clear()
                continue

            # Temporary rate limit -> rotate key
            keys_tried_this_cycle.add(api_key)

            usable = pool.available(model)

            if (
                len(keys_tried_this_cycle) >= len(usable)
                and len(usable) > 0
            ):
                wait = min(
                    3,
                    1 + attempt // max(1, len(usable))
                )

                print(
                    f"  [mistral] all {len(usable)} key(s) "
                    f"rate-limited once; brief wait {wait}s..."
                )

                time.sleep(wait)
                keys_tried_this_cycle.clear()

            continue

        raise LLMError(
            f"Mistral API error {resp.status_code}: {resp.text}"
        )

    raise LLMError(
        f"Mistral API failed after {max_attempts} attempts: {last_err}"
    )

def call_llm(prompt, provider="groq", model=None, temperature=0.2, max_tokens=1536):
    """Unified entry point. provider in {'groq', 'gemini'}."""
    if provider == "groq":
        return _call_groq(prompt, model=model or DEFAULT_GROQ_MODEL,
                           temperature=temperature, max_tokens=max_tokens)
    elif provider == "gemini":
        return _call_gemini(prompt, model=model or DEFAULT_GEMINI_MODEL,
                             temperature=temperature, max_tokens=max_tokens)
    elif provider == "mistral":
        return _call_mistral(prompt, model=model or DEFAULT_MISTRAL_MODEL,
                             temperature=temperature, max_tokens=max_tokens)
    else:
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
    pool = get_key_pool("groq")
    print(pool.summary())
    if pool.keys:
        out = call_llm("Reply with exactly this JSON: {\"ok\": true}")
        print("Groq response:", out)
        print("Parsed:", extract_json_object(out))
    else:
        print("No GROQ keys set -- skipping live call.")