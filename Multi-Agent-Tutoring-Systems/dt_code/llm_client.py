"""
llm_client.py

Thin, dependency-light wrapper around two FREE hosted LLM APIs:
  - Groq   (https://console.groq.com) -- fast, generous free tier, hosts
            open models like Llama-3.3-70B and Qwen. Use as your GROQ_MODEL.
  - Gemini (https://aistudio.google.com) -- free tier for Gemini Flash models.

Only `requests` is needed (no heavy SDKs). Both are plain REST calls, so your
laptop's CPU does nothing but send/receive JSON -- all the actual inference
runs on Groq's / Google's servers for free.

Set API keys as environment variables (e.g. in a .env file, loaded by
python-dotenv in run_baseline.py):
    GROQ_API_KEY=...
    GOOGLE_API_KEY=...      # optional, only needed if you use provider="gemini"

Recommended default split for genuine independence between Tutor and Verifier
in later steps: use a DIFFERENT model family for each (e.g. Groq/Llama for the
student simulator + Tutor, Gemini for the Verifier once you build Step 4).
"""

import os
import time
import json
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"  # llama-3.1-8b-instant and
                                            # llama-3.3-70b-versatile are BOTH
                                            # being shut down by Groq on
                                            # 08/16/2026 -- migrated to their
                                            # recommended replacements. If you
                                            # hit a 429 with an unfamiliar
                                            # error again, check
                                            # https://console.groq.com/docs/deprecations
                                            # before assuming it's a normal
                                            # rate limit.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"  # gemini-2.0-flash was deprecated/shut
                                            # down June 1 2026 -- if you see
                                            # "limit: 0" errors again in future,
                                            # check https://ai.google.dev/gemini-api/docs/pricing
                                            # for which models still have a free tier


class LLMError(Exception):
    pass


class DailyQuotaExceeded(LLMError):
    """Raised when a provider's daily token/request cap is hit -- retrying with
    backoff cannot fix this (the reset is usually hours away), so callers
    should stop hammering this model/provider rather than burn retries."""
    pass


def _is_daily_quota_error(resp_text):
    t = resp_text.lower()
    return ("tokens per day" in t or "requests per day" in t
            or " tpd" in t or " rpd" in t or "daily" in t)


def _call_groq(prompt, model=DEFAULT_GROQ_MODEL, temperature=0.2, max_tokens=1536, retries=3):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMError("GROQ_API_KEY is not set. Get a free key at https://console.groq.com")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err = None
    for attempt in range(retries):
        resp = requests.post(GROQ_URL, headers=headers, json=body, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        if resp.status_code == 429:
            if _is_daily_quota_error(resp.text):
                # This is a per-day cap (e.g. "100000 tokens per day" for this
                # specific model). Waiting 5-15s will never fix it -- fail
                # immediately instead of burning 3 useless retries.
                raise DailyQuotaExceeded(
                    f"Groq daily quota exhausted for model '{model}': {resp.text}"
                )
            # Genuine short-lived per-minute rate limit -- backoff is worth it.
            wait = 5 * (attempt + 1)
            print(f"  [groq] rate limited, waiting {wait}s...")
            time.sleep(wait)
            last_err = resp.text
            continue
        raise LLMError(f"Groq API error {resp.status_code}: {resp.text}")

    raise LLMError(f"Groq API failed after {retries} retries: {last_err}")




def _call_gemini(prompt, model=DEFAULT_GEMINI_MODEL, temperature=0.2, max_tokens=1536, retries=3):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise LLMError("GOOGLE_API_KEY is not set. Get a free key at https://aistudio.google.com")

    url = GEMINI_URL_TMPL.format(model=model)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }

    last_err = None
    for attempt in range(retries):
        resp = requests.post(url, params={"key": api_key}, json=body, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        if resp.status_code == 429:
            if _is_daily_quota_error(resp.text):
                raise DailyQuotaExceeded(
                    f"Gemini daily quota exhausted for model '{model}': {resp.text}"
                )
            wait = 5 * (attempt + 1)
            print(f"  [gemini] rate limited, waiting {wait}s...")
            time.sleep(wait)
            last_err = resp.text
            continue
        raise LLMError(f"Gemini API error {resp.status_code}: {resp.text}")

    raise LLMError(f"Gemini API failed after {retries} retries: {last_err}")


def call_llm(prompt, provider="groq", model=None, temperature=0.2, max_tokens=1536):
    """Unified entry point. provider in {'groq', 'gemini'}."""
    if provider == "groq":
        return _call_groq(prompt, model=model or DEFAULT_GROQ_MODEL,
                           temperature=temperature, max_tokens=max_tokens)
    elif provider == "gemini":
        return _call_gemini(prompt, model=model or DEFAULT_GEMINI_MODEL,
                             temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def extract_json_object(text):
    """
    Same robust extraction logic as dt_code/llm_response_processing/response_preprocess.py
    -- handles raw JSON, ```json fenced blocks, or JSON embedded in extra text.
    """
    if not text:
        return None
    if isinstance(text, dict):
        return text

    s = text.strip()
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

    import re
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None


if __name__ == "__main__":
    # Quick manual smoke test -- only runs real network calls if GROQ_API_KEY is set.
    if os.environ.get("GROQ_API_KEY"):
        out = call_llm("Reply with exactly this JSON: {\"ok\": true}")
        print("Groq response:", out)
        print("Parsed:", extract_json_object(out))
    else:
        print("GROQ_API_KEY not set -- skipping live call. "
              "Get a free key at https://console.groq.com and set it to test.")