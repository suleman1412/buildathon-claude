"""Local wiring tests. Use MockLLMClient (no model, no network) and a
trivial synthetic 'repo' (a two-line Python script) in place of a real
cloned repo -- these prove the pipeline is wired correctly without any
GPU, network access, or real research code. The actual model + real repo
execution only ever happens inside the Kaggle/Colab notebook.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.execution import extract_metric
from src.extraction import ExtractedClaim, extract_claim
from src.llm_client import AnthropicLLMClient, MockLLMClient, SpendCapExceeded, _parse_json_loose
from src.pipeline import run_pipeline
from src.verification import verify


class TestJSONRepair(unittest.TestCase):
    def test_repairs_stray_backslash_escape(self):
        # Real failure from a local model: hyphenation artifact "LAN-\GUAGE"
        # is not a valid JSON escape (\G).
        raw = r'{"title": "LARGE LAN-\GUAGE MODELS", "claimed_metric_value": 91.5}'
        result = _parse_json_loose(raw)
        self.assertEqual(result["title"], "LARGE LAN-\\GUAGE MODELS")
        self.assertEqual(result["claimed_metric_value"], 91.5)

    def test_still_parses_valid_json(self):
        raw = '{"title": "Fine", "claimed_metric_value": 1.0}'
        result = _parse_json_loose(raw)
        self.assertEqual(result["title"], "Fine")


def _fake_anthropic_response(text: str, input_tokens: int, output_tokens: int):
    response = unittest.mock.MagicMock()
    response.content = [unittest.mock.MagicMock(text=text)]
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    return response


class TestAnthropicSpendCap(unittest.TestCase):
    def _make_client(self, max_spend_usd, input_tokens, output_tokens):
        client = AnthropicLLMClient(model="claude-haiku-4-5-20251001", api_key="fake", max_spend_usd=max_spend_usd)
        client._client.messages.create = unittest.mock.MagicMock(
            return_value=_fake_anthropic_response("ok", input_tokens, output_tokens)
        )
        return client

    def test_tracks_spend_and_allows_calls_under_cap(self):
        client = self._make_client(max_spend_usd=2.0, input_tokens=1000, output_tokens=1000)
        client.complete("sys", "user")
        # haiku rates: (1.00, 5.00) per Mtok -> 1000*1.00/1e6 + 1000*5.00/1e6 = 0.000006
        self.assertGreater(client.spend_usd, 0)
        self.assertLess(client.spend_usd, 2.0)

    def test_raises_before_next_call_once_cap_hit(self):
        # Huge output token count on a low cap -> first call pushes spend over the cap.
        client = self._make_client(max_spend_usd=0.001, input_tokens=100_000, output_tokens=100_000)
        client.complete("sys", "user")  # this call is allowed to complete
        self.assertGreater(client.spend_usd, 0.001)
        with self.assertRaises(SpendCapExceeded):
            client.complete("sys", "user")  # blocked before making another API call


class TestExtraction(unittest.TestCase):
    def test_parses_response(self):
        response = json.dumps({
            "title": "Test Paper",
            "github_repo": "https://github.com/example/repo",
            "dataset": "sst2",
            "claimed_metric_name": "accuracy",
            "claimed_metric_value": 95.1,
            "claimed_metric_unit": "%",
            "eval_notes": "RoBERTa-large + LoRA on SST-2 validation split.",
        })
        llm = MockLLMClient([response])
        claim = extract_claim(llm, "some paper text")
        self.assertEqual(claim.title, "Test Paper")
        self.assertEqual(claim.claimed_metric_value, 95.1)


class TestExtractMetric(unittest.TestCase):
    def test_regex_match(self):
        stdout = "Running eval...\nFinal Accuracy: 94.87\nDone."
        value = extract_metric(stdout, r"Final Accuracy: ([\d.]+)")
        self.assertEqual(value, 94.87)

    def test_no_match(self):
        self.assertIsNone(extract_metric("no numbers here", r"Accuracy: ([\d.]+)"))


class TestVerify(unittest.TestCase):
    def _claim(self):
        return ExtractedClaim(
            title="T", github_repo="url", dataset="d",
            claimed_metric_name="accuracy", claimed_metric_value=95.1,
            claimed_metric_unit="%", eval_notes="",
        )

    def test_pass_within_tolerance(self):
        result = verify(self._claim(), 94.9, tolerance_relative=0.05)
        self.assertTrue(result.passed)

    def test_fail_outside_tolerance(self):
        result = verify(self._claim(), 40.0, tolerance_relative=0.05)
        self.assertFalse(result.passed)

    def test_fail_when_reproduced_missing(self):
        result = verify(self._claim(), None, tolerance_relative=0.05)
        self.assertFalse(result.passed)


class TestFullPipelineSynthetic(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.synthetic_repo = self.tmpdir / "synthetic_repo"
        self.synthetic_repo.mkdir()
        (self.synthetic_repo / "eval.py").write_text("print('Final Accuracy: 95.05')\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pipeline_pass(self):
        extraction_response = json.dumps({
            "title": "Synthetic Paper",
            "github_repo": "https://example.invalid/repo.git",
            "dataset": "toy",
            "claimed_metric_name": "accuracy",
            "claimed_metric_value": 95.1,
            "claimed_metric_unit": "%",
            "eval_notes": "Synthetic test.",
        })
        run_plan_response = json.dumps({
            "command": "python3 eval.py",
            "metric_regex": r"Final Accuracy: ([\d.]+)",
        })
        llm = MockLLMClient([extraction_response, run_plan_response])
        config = Config(workdir=str(self.tmpdir / "runs"))

        def fake_clone_repo(repo_url, dest_dir, timeout_sec=300):
            shutil.copytree(self.synthetic_repo, dest_dir)
            return dest_dir

        with patch("src.pipeline.clone_repo", side_effect=fake_clone_repo), \
             patch("src.pipeline.install_requirements", return_value=""), \
             patch("src.pipeline.extract_text", return_value="synthetic paper text"):
            result = run_pipeline("fake.pdf", llm, config)

        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.reproduced_value, 95.05)


if __name__ == "__main__":
    unittest.main()
