"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of November 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-3.8-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-3.8-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


def _require_api_key(name: str) -> str:
    load_dotenv()
    api_key = os.getenv(name)
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set the {name} environment variable before running."
        )
    return api_key


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING_1M_TOKENS[model]
    return (
        input_tokens * pricing["input"] + output_tokens * pricing["output"]
    ) / 1_000_000


def _is_gemini_unavailable_error(exc: Exception) -> bool:
    return (
        getattr(exc, "code", None) == 503
        or getattr(exc, "status_code", None) == 503
        or "503" in str(exc)
        or "UNAVAILABLE" in str(exc).upper()
    )


def _retry_gemini_request(request: Callable[[], Any]) -> Any:
    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            return request()
        except Exception as exc:
            if not _is_gemini_unavailable_error(exc) or attempt == max_retries:
                raise
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI

    client = OpenAI(api_key=_require_api_key("OPENAI_API_KEY"))
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start_time

    response_text = response.choices[0].message.content or ""
    usage_data = response.usage
    usage = {
        "input_tokens": usage_data.prompt_tokens if usage_data else 0,
        "output_tokens": usage_data.completion_tokens if usage_data else 0,
    }

    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 3.8 Flash (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_require_api_key("GEMINI_API_KEY"))

    start_time = time.perf_counter()
    response = _retry_gemini_request(
        lambda: client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_tokens,
            ),
        )
    )
    latency = time.perf_counter() - start_time

    metadata = response.usage_metadata
    usage = {
        "input_tokens": (
            getattr(metadata, "prompt_token_count", 0) or 0
            if metadata else 0
        ),
        "output_tokens": (
            getattr(metadata, "candidates_token_count", 0) or 0
            if metadata else 0
        ),
    }

    return response.text or "", latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from anthropic import Anthropic

    client = Anthropic(api_key=_require_api_key("ANTHROPIC_API_KEY"))

    # Anthropic accepts temperature and top_p, but they should not both be
    # supplied in the same request. Preserve the original arguments by using
    # temperature as the primary sampling control.
    del top_p

    start_time = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - start_time

    response_text = "\n".join(
        block.text for block in response.content
        if getattr(block, "type", None) == "text"
    )
    usage = {
        "input_tokens": response.usage.input_tokens or 0,
        "output_tokens": response.usage.output_tokens or 0,
    }

    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 3.8 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    model_calls = (
        ("gpt4o", OPENAI_MODEL, call_openai),
        ("gpt4o_mini", OPENAI_MINI_MODEL, call_openai),
        ("gemini_flash", GEMINI_MODEL, call_gemini),
    )

    results = {}
    for result_key, model_name, call_function in model_calls:
        response_text, latency, usage = call_function(prompt, model=model_name)
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]

        results[result_key] = {
            "response": response_text,
            "latency": latency,
            "cost": calculate_cost(model_name, input_tokens, output_tokens),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    return results
    # TODO: Call call_openai with default gpt-4o model
    # TODO: Call call_openai with gpt-4o-mini model
    # TODO: Call call_gemini with default gemini-3.8-flash model
    # TODO: Calculate costs exactly based on input and output token counts using PRICING_1M_TOKENS
    #       Formula: Cost = (input_tokens * input_rate_per_1M + output_tokens * output_rate_per_1M) / 1,000,000
    # TODO: Assemble and return the comparison dictionary.
    raise NotImplementedError("Implement compare_models")


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 3.8 Flash (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_require_api_key("GEMINI_API_KEY"))

    history = []

    config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=1024
    )

    print("\n=== Starting Gemini 3.8 Chatbot (type 'quit' to exit) ===\n")
    print("======================================================")
    print("🤖 Vin Smart Future – Intelligent Chat Assistant")
    print("Powered by Google Gemini 3.8 Flash")
    print("Type 'quit' or 'exit' to end the session.")
    print("======================================================\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        conversation = history + [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)]
            )
        ]

        print("Gemini: ", end="", flush=True)

        response_parts = []

        try:
            max_retries = 3
            answer = ""

            for attempt in range(max_retries + 1):
                try:
                    response_parts = []

                    stream = client.models.generate_content_stream(
                        model=GEMINI_MODEL,
                        contents=conversation,
                        config=config
                    )

                    for chunk in stream:
                        if chunk.text:
                            print(chunk.text, end="", flush=True)
                            response_parts.append(chunk.text)

                    answer = "".join(response_parts)
                    print()
                    break

                except Exception as e:
                    error_message = str(e)

                    # Retry only for temporary server errors
                    if "503" in error_message or "UNAVAILABLE" in error_message:

                        if attempt < max_retries:
                            wait_time = 2 ** attempt

                            print(
                                f"\nGemini server is busy. "
                                f"Retrying in {wait_time} seconds... "
                                f"({attempt + 1}/{max_retries})"
                            )

                            time.sleep(wait_time)

                        else:
                            raise e
                    else:
                        raise e

            if answer:
                history.extend([
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=user_input)]
                    ),
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=answer)]
                    )
                ])

                history = history[-6:]

        except Exception as e:
            print(f"\nError: {e}")

# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """

    if max_retries < 0:
        raise ValueError("max_retries must be >= 0.")
    if base_delay < 0:
        raise ValueError("base_delay must be >= 0.")

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))
    # TODO: implement retry loop with exponential backoff
    raise NotImplementedError("Implement retry_with_backoff")


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results = []

    for prompt in prompts:
        comparison = compare_models(prompt)
        results.append({"prompt": prompt, **comparison})

    return results
    # TODO: iterate over prompts, call compare_models, and inject the original "prompt".
    raise NotImplementedError("Implement batch_compare")


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    headers = [
        "Prompt",
        "Model",
        "Response (truncated)",
        "Latency",
        "Tokens (In/Out)",
        "Cost (USD)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    model_labels = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o Mini",
        "gemini_flash": "Gemini 3.8 Flash",
    }

    for item in results:
        prompt_text = str(item.get("prompt", ""))
        prompt_text = prompt_text.replace("\n", " ").replace("|", "\\|")

        for model_key, model_label in model_labels.items():
            if model_key not in item:
                continue

            stats = item[model_key]
            response_text = str(stats.get("response", ""))
            response_text = response_text.replace("\n", " ").replace("|", "\\|")
            if len(response_text) > 50:
                response_text = response_text[:47] + "..."

            latency = float(stats.get("latency", 0.0))
            input_tokens = int(stats.get("input_tokens", 0))
            output_tokens = int(stats.get("output_tokens", 0))
            cost = float(stats.get("cost", 0.0))

            row = [
                prompt_text,
                model_label,
                response_text,
                f"{latency:.2f}s",
                f"{input_tokens}/{output_tokens}",
                f"${cost:.6f}",
            ]
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
    # TODO: Build and return the formatted table string. Truncate response to 50 chars for clean display.
    raise NotImplementedError("Implement format_comparison_table")


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 3.8 Flash Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
