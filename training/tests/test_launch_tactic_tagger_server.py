#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_DIR))

import launch_tactic_tagger_server


class LaunchTacticTaggerServerTest(unittest.TestCase):
    def test_supports_hf_style_adapter_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for tactic in ["advice", "validation"]:
                os.makedirs(os.path.join(tmpdir, tactic), exist_ok=True)

            self.assertEqual(
                launch_tactic_tagger_server.resolve_adapter_path(tmpdir, "advice"),
                os.path.join(tmpdir, "advice"),
            )
            self.assertEqual(
                launch_tactic_tagger_server.resolve_adapter_path(tmpdir, "validation"),
                os.path.join(tmpdir, "validation"),
            )

    def test_prefers_legacy_layout_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_path = os.path.join(tmpdir, "Llama-3.1-8B-Instruct-tagger-advice")
            hf_path = os.path.join(tmpdir, "advice")
            os.makedirs(legacy_path, exist_ok=True)
            os.makedirs(hf_path, exist_ok=True)

            self.assertEqual(
                launch_tactic_tagger_server.resolve_adapter_path(tmpdir, "advice"),
                legacy_path,
            )


if __name__ == "__main__":
    unittest.main()