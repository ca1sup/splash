#!/usr/bin/env python3
"""Validate and summarize bench_m1_ultra.py JSONL measurements."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REQUIRED = {
    "run_id",
    "scenario",
    "sample",
    "requested_prompt_tokens",
    "rendered_prompt_tokens",
    "cached_prompt_tokens",
    "output_tokens_committed",
    "prefill_seconds",
    "decode_seconds",
    "http_ttft_seconds",
    "request_seconds",
}


def load(path: Path) -> tuple[dict, list[dict]]:
    manifest = None
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("record_type") == "manifest":
            if manifest is not None:
                raise ValueError(f"{path}:{line_number}: duplicate manifest")
            manifest = row
        elif row.get("record_type") == "measurement":
            missing = REQUIRED - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
            if row["exit_status"] != 0:
                raise ValueError(f"{path}:{line_number}: failed measurement")
            records.append(row)
        else:
            raise ValueError(f"{path}:{line_number}: unknown record type")
    if manifest is None or not records:
        raise ValueError(f"{path}: manifest and at least one measurement are required")
    return manifest, records


def median_row(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "min": min(ordered),
        "max": max(ordered),
        "stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "samples": len(ordered),
    }


def summarize(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in records:
        key = row["scenario"], row["requested_prompt_tokens"]
        groups[key].append(row)
    result = []
    for (scenario, requested), rows in sorted(groups.items()):
        if len({row["run_id"] for row in rows}) != 1:
            raise ValueError("records from multiple runner invocations were mixed")
        prompt_tokens = [row["rendered_prompt_tokens"] for row in rows]
        prefill = [
            row["native_delta"]["prefill_input_tokens"] / row["prefill_seconds"]
            for row in rows
            if row["prefill_seconds"] and row["native_delta"]["prefill_input_tokens"]
        ]
        decode = [
            row["native_delta"]["decode_output_tokens"] / row["decode_seconds"]
            for row in rows
            if row["decode_seconds"] and row["native_delta"]["decode_output_tokens"]
        ]
        effective = [
            row["rendered_prompt_tokens"] / row["http_ttft_seconds"]
            for row in rows
            if row["http_ttft_seconds"]
        ]
        result.append(
            {
                "scenario": scenario,
                "requested_prompt_tokens": requested,
                "actual_prompt_tokens": median_row([float(x) for x in prompt_tokens]),
                "cached_prompt_tokens": median_row(
                    [float(row["cached_prompt_tokens"] or 0) for row in rows]
                ),
                "native_prefill_tok_s": median_row(prefill) if prefill else None,
                "native_decode_tok_s": median_row(decode) if decode else None,
                "effective_http_prefill_tok_s": median_row(effective)
                if effective
                else None,
                "http_ttft_ms": median_row(
                    [
                        row["http_ttft_seconds"] * 1000
                        for row in rows
                        if row["http_ttft_seconds"]
                    ]
                ),
                "output_tokens": median_row(
                    [float(row["output_tokens_committed"]) for row in rows]
                ),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable summary"
    )
    args = parser.parse_args()
    manifest, records = load(args.input)
    output = {"manifest": manifest, "summary": summarize(records)}
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(
            "scenario requested actual cached prefill_tok/s decode_tok/s TTFT_ms output"
        )
        for row in output["summary"]:
            prefill = row["native_prefill_tok_s"]
            decode = row["native_decode_tok_s"]
            ttft = row["http_ttft_ms"]
            print(
                f"{row['scenario']:7} {row['requested_prompt_tokens']:8} "
                f"{row['actual_prompt_tokens']['median']:7.0f} "
                f"{row['cached_prompt_tokens']['median']:6.0f} "
                f"{prefill['median'] if prefill else 'n/a':>12} "
                f"{decode['median'] if decode else 'n/a':>12} "
                f"{ttft['median'] if ttft else 'n/a':>8} "
                f"{row['output_tokens']['median']:6.0f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
