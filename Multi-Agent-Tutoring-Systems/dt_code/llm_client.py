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
import re
import time
import json
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
    """Raised when a provider's daily token/request cap is hit -- retrying with
    backoff cannot fix this (the reset is usually hours away), so callers
    should stop hammering this model/provider rather than burn retries."""
    pass


def _is_daily_quota_error(resp_text):
    t = resp_text.lower()
    return ("tokens per day" in t or "requests per day" in t
            or " tpd" in t or " rpd" in t or "daily" in t)


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
    BUGFIX: openai/gpt-oss-20b and gpt-oss-120b are reasoning models. On Groq
    they spend part of the token budget on hidden chain-of-thought (returned
    separately in a `reasoning` field), and only write the final answer into
    `content` once reasoning finishes. With a low max_tokens and no cap on
    reasoning effort, the model can burn the ENTIRE budget on reasoning and
    never get to write `content` at all -- which returns as "", not
    malformed JSON. That's why extract_json_object was getting a bare empty
    string for these models specifically.

    Fix: explicitly cap reasoning_effort="low" for gpt-oss models (Groq
    supports low/medium/high, see console.groq.com/docs/reasoning) so less
    of the budget goes to hidden thinking, and raise the default max_tokens
    as a safety margin. qwen models use reasoning_effort differently
    (none/default) so we don't set it for those.
    """
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
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"

    last_err = None
    for attempt in range(retries):
        resp = requests.post(GROQ_URL, headers=headers, json=body, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if not content:
                # Content came back genuinely empty -- almost always means
                # reasoning consumed the whole max_tokens budget for a
                # gpt-oss model. Surface this distinctly from a normal parse
                # failure so it's easy to diagnose from the failures log.
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
    Same robust extraction logic as dt_code/llm_response_processing/response_preprocess.py
    -- handles raw JSON, ```json fenced blocks, or JSON embedded in extra text.

    BUGFIX: reasoning models (e.g. qwen/qwen3.6-27b) wrap their answer in
    <think>...</think> before the actual JSON. Their thinking text very often
    contains stray '{'/'}' characters -- e.g. when the model quotes the
    Response_Format template back to itself while reasoning about it, or
    writes example JSON mid-thought. The old greedy regex (searching from the
    first '{' to the last '}') would match from the FIRST '{' anywhere in
    the text (often inside
    <think>) to the LAST '}' (the real answer), swallowing everything in
    between into one invalid blob. Stripping <think> blocks first removes
    that noise before the regex ever runs.
    """
    if not text:
        return None
    if isinstance(text, dict):
        return text

    s = text.strip()

    # Strip <think>...</think> reasoning blocks (case-insensitive, may span
    # multiple lines) before attempting any parse below.
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
    # Quick manual smoke test -- only runs real network calls if GROQ_API_KEY is set.
    if os.environ.get("GROQ_API_KEY"):
        out = call_llm("Reply with exactly this JSON: {\"ok\": true}")
        print("Groq response:", out)
        print("Parsed:", extract_json_object(out))
    else:
        print("GROQ_API_KEY not set -- skipping live call. "
              "Get a free key at https://console.groq.com and set it to test.")
