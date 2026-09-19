import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dev.benchmarks import bench_m1_ultra as bench
from dev.benchmarks import summarize_m1_ultra as summary


def measurement(
    prompt_hash: str, decode_seconds: float, prefill_seconds: float
) -> dict:
    return {
        "scenario": "uncached",
        "sample": 0,
        "requested_prompt_tokens": 512,
        "prompt_sha256": prompt_hash,
        "native_delta": {
            "decode_output_tokens": 4,
            "prefill_input_tokens": 512,
        },
        "decode_seconds": decode_seconds,
        "prefill_seconds": prefill_seconds,
    }


class M1UltraBenchmarkTests(unittest.TestCase):
    def test_corpus_seed_makes_prompts_reproducible(self):
        self.assertEqual(
            bench.prompt_for(512, "uncached-seed-0-512"),
            bench.prompt_for(512, "uncached-seed-0-512"),
        )
        self.assertNotEqual(
            bench.prompt_for(512, "uncached-seed-0-512"),
            bench.prompt_for(512, "uncached-other-0-512"),
        )

    def test_paired_ratio_is_candidate_over_reference(self):
        manifest = {
            "model": "model",
            "contexts_requested": [512],
            "output_tokens_requested": 4,
            "reasoning_effort": "none",
            "corpus_seed": "seed",
            "sampling_config_hash": "hash",
        }
        result = summary.paired_comparison(
            manifest,
            [measurement("same", 2.0, 1.0)],
            manifest,
            [measurement("same", 1.0, 0.5)],
        )
        self.assertEqual(
            result[0]["paired_decode_ratio_candidate_over_reference"]["median"],
            2.0,
        )
        self.assertEqual(
            result[0]["paired_prefill_ratio_candidate_over_reference"]["median"],
            2.0,
        )

    def test_mismatched_corpus_is_rejected(self):
        reference = {"corpus_seed": "one"}
        candidate = {"corpus_seed": "two"}
        with self.assertRaisesRegex(ValueError, "corpus_seed"):
            summary.validate_comparable(reference, candidate)

    def test_stream_measurement_records_first_content_before_completion(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers["Content-Length"])
                json.loads(self.rfile.read(length))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(
                    b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
                )
                self.wfile.flush()
                time.sleep(0.02)
                self.wfile.write(
                    b'data: {"choices":[{"delta":{"content":"second"}}],'
                    b'"usage":{"completion_tokens":2},"metrics":{}}\n\n'
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            def log_message(self, *_args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result, wall, ttft, tokens = bench.stream_request(
                f"http://127.0.0.1:{server.server_port}", {}
            )
        finally:
            server.shutdown()
            thread.join(timeout=1)
            server.server_close()
        self.assertEqual(result["usage"]["completion_tokens"], 2)
        self.assertEqual(tokens, 2)
        self.assertIsNotNone(ttft)
        self.assertLess(ttft, wall - 0.005)


if __name__ == "__main__":
    unittest.main()
