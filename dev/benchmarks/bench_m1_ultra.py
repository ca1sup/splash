#!/usr/bin/env python3
"""Reproducible HTTP benchmark runner for the local Apple7 Splash build.

The runner records native phase timings reported by Splash and independently
measures request/first-content timing with a monotonic client clock. It never
stores prompt text: prompts are deterministic synthetic material and only
their hashes are written to JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import time
import uuid
from pathlib import Path


def sha256(value: object) -> str:
    if isinstance(value, bytes):
        data = value
    else:
        data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def parse_contexts(value: str) -> list[int]:
    try:
        result = [int(part) for part in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("contexts must be comma-separated integers") from None
    if not result or any(context <= 0 for context in result):
        raise argparse.ArgumentTypeError("contexts must be positive")
    return result


class BenchError(RuntimeError):
    pass


def request(base: str, method: str, path: str, body: dict | None = None, timeout: float = 1800.0):
    host = base.removeprefix("http://").removeprefix("https://").rstrip("/")
    if "/" in host:
        host = host.split("/", 1)[0]
    if ":" in host and not host.startswith("["):
        hostname, port = host.rsplit(":", 1)
        connection = http.client.HTTPConnection(hostname, int(port), timeout=timeout)
    else:
        connection = http.client.HTTPConnection(host, 80, timeout=timeout)
    payload = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    headers = {} if payload is None else {"Content-Type": "application/json"}
    try:
        started = time.monotonic()
        connection.request(method, path, payload, headers)
        response = connection.getresponse()
        raw = response.read()
        elapsed = time.monotonic() - started
    finally:
        connection.close()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BenchError(f"{method} {path} returned invalid JSON: {raw[:200]!r}") from error
    if response.status != 200:
        raise BenchError(f"{method} {path} returned HTTP {response.status}: {document!r}")
    return document, elapsed


def stream_request(base: str, body: dict, timeout: float = 1800.0) -> tuple[dict, float, float | None, int]:
    host = base.removeprefix("http://").removeprefix("https://").rstrip("/")
    if "/" in host:
        host = host.split("/", 1)[0]
    hostname, port = host.rsplit(":", 1) if ":" in host else (host, "80")
    connection = http.client.HTTPConnection(hostname, int(port), timeout=timeout)
    payload = json.dumps(body, separators=(",", ":")).encode()
    started = time.monotonic()
    first_content: float | None = None
    chunks: list[dict] = []
    committed_events = 0
    try:
        connection.request(
            "POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        if response.status != 200:
            raise BenchError(f"stream returned HTTP {response.status}: {response.read()[:400]!r}")
        for raw_line in response.read().splitlines():
            if not raw_line.startswith(b"data: "):
                continue
            event = raw_line[6:]
            if event == b"[DONE]":
                continue
            document = json.loads(event)
            chunks.append(document)
            if document.get("usage", {}).get("completion_tokens", 0):
                committed_events = document["usage"]["completion_tokens"]
            for choice in document.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    if first_content is None:
                        first_content = time.monotonic()
    finally:
        connection.close()
    if not chunks:
        raise BenchError("stream returned no JSON chunks")
    final = chunks[-1]
    for document in reversed(chunks):
        if document.get("metrics") is not None:
            final = document
            break
    return final, time.monotonic() - started, (
        None if first_content is None else first_content - started
    ), committed_events


def prompt_for(target_tokens: int, nonce: str, suffix: str = "") -> str:
    # This is intentionally synthetic and public. The tokenizer count after
    # chat templating is authoritative and is recorded in each result.
    block = (
        "The archive contains ordinary project notes about interfaces, tests, "
        "memory budgets, and implementation details. "
    )
    body = (block * max(1, target_tokens * 4 // len(block)))[: target_tokens * 5]
    return (
        f"Benchmark corpus {nonce}. Analyze the following neutral engineering notes, "
        "then answer with a long numbered list from 1 upward.\n\n"
        f"{body}\n{suffix}"
    )


def body(model: str, prompt: str, output_tokens: int, reasoning: str) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": output_tokens,
        "temperature": 0,
        "reasoning_effort": reasoning,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def run_one(base: str, model: str, prompt: str, reasoning: str, output_tokens: int):
    before, _ = request(base, "GET", "/status")
    result, wall, first_content, streamed_tokens = stream_request(
        base, body(model, prompt, output_tokens, reasoning)
    )
    after, _ = request(base, "GET", "/status")
    usage = result.get("usage", {})
    metrics = result.get("metrics", {})
    before_metrics = before.get("metrics", {})
    after_metrics = after.get("metrics", {})
    native_delta = {
        key: after_metrics.get(key, 0) - before_metrics.get(key, 0)
        for key in (
            "prefill_wall_ms",
            "decode_wall_ms",
            "prefill_input_tokens",
            "decode_output_tokens",
            "drafted_tokens",
            "accepted_draft_tokens",
        )
    }
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_prompt_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens"),
        "output_tokens_committed": usage.get("completion_tokens", streamed_tokens),
        "prefill_seconds": native_delta["prefill_wall_ms"] / 1000.0,
        "decode_seconds": native_delta["decode_wall_ms"] / 1000.0,
        "http_ttft_seconds": None if first_content is None else first_content,
        "request_seconds": wall,
        "cache": metrics.get("cache", {}),
        "native_metrics": metrics,
        "native_delta": native_delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("SPLASH_BENCH_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=parse_contexts, default=[512, 4096, 16384, 32768])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--scenario", choices=("uncached", "exact", "append", "all"), default="all")
    args = parser.parse_args()
    if args.samples <= 0 or args.warmup < 0 or args.output_tokens <= 0:
        parser.error("samples/output-tokens must be positive and warmup must be nonnegative")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scenarios = ("uncached", "exact", "append") if args.scenario == "all" else (args.scenario,)
    run_manifest = {
        "runner": "bench_m1_ultra.py",
        "run_id": uuid.uuid4().hex,
        "url": args.url,
        "model": args.model,
        "contexts_requested": args.contexts,
        "samples": args.samples,
        "warmup": args.warmup,
        "output_tokens_requested": args.output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "scenarios": scenarios,
        "clock": "time.monotonic",
    }
    with args.output.open("a", encoding="utf-8") as output:
        output.write(json.dumps({"record_type": "manifest", **run_manifest}) + "\n")
        for warmup in range(args.warmup):
            prompt = prompt_for(128, f"warmup-{run_manifest['run_id']}-{warmup}")
            run_one(args.url, args.model, prompt, args.reasoning_effort, min(32, args.output_tokens))
        for scenario in scenarios:
            for sample in range(args.samples):
                for requested in args.contexts:
                    nonce = f"{scenario}-{run_manifest['run_id']}-{sample}-{requested}"
                    prompt = prompt_for(requested, nonce)
                    if scenario == "append":
                        prompt += "\nAppend-only suffix: compare the final two implementation choices."
                    if scenario == "exact":
                        run_one(args.url, args.model, prompt, args.reasoning_effort, args.output_tokens)
                        measured_prompt = prompt
                    elif scenario == "append":
                        measured_prompt = prompt + "\nNew suffix for the next turn: give one concise conclusion."
                    else:
                        measured_prompt = prompt
                    measured = run_one(
                        args.url,
                        args.model,
                        measured_prompt,
                        args.reasoning_effort,
                        args.output_tokens,
                    )
                    record = {
                        "record_type": "measurement",
                        "run_id": run_manifest["run_id"],
                        "engine": "splash",
                        "scenario": scenario,
                        "sample": sample,
                        "requested_prompt_tokens": requested,
                        "prompt_sha256": sha256(measured_prompt),
                        "rendered_prompt_tokens": measured["prompt_tokens"],
                        "cached_prompt_tokens": measured["cached_prompt_tokens"],
                        "output_tokens_committed": measured["output_tokens_committed"],
                        "prefill_seconds": measured["prefill_seconds"],
                        "decode_seconds": measured["decode_seconds"],
                        "http_ttft_seconds": measured["http_ttft_seconds"],
                        "request_seconds": measured["request_seconds"],
                        "cache": measured["cache"],
                        "native_metrics": measured["native_metrics"],
                        "native_delta": measured["native_delta"],
                        "peak_process_bytes": None,
                        "peak_metal_bytes": None,
                        "swap_delta_bytes": None,
                        "correctness": "not-run",
                        "exit_status": 0,
                    }
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()
                    print(
                        f"{scenario} sample={sample} requested={requested} actual="
                        f"{record['rendered_prompt_tokens']} cache={record['cached_prompt_tokens']} "
                        f"decode_tokens={record['output_tokens_committed']}",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
