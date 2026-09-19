import unittest

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


if __name__ == "__main__":
    unittest.main()
