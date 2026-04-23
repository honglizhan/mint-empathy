import importlib.util
import unittest
from pathlib import Path

from absl import flags

_MODULE = None


def _load_with_safe_absl_flags(spec):
    original_define_string = flags.DEFINE_string

    def safe_define_string(name, default, help_string, *args, **kwargs):
        if name in flags.FLAGS:
            return flags.FLAGS[name]
        return original_define_string(name, default, help_string, *args, **kwargs)

    flags.DEFINE_string = safe_define_string
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        flags.DEFINE_string = original_define_string


def load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    repo_root = Path(__file__).resolve().parents[2]
    mod_path = repo_root / "evaluation" / "step4_analyze.py"
    spec = importlib.util.spec_from_file_location("step4_analyze_mod", mod_path)
    mod = _load_with_safe_absl_flags(spec)
    _MODULE = mod
    return _MODULE


class BootstrapDefaultsTest(unittest.TestCase):
    def test_bootstrap_defaults_use_10000_resamples(self):
        mod = load_module()
        self.assertEqual(mod.bootstrap_means.__defaults__[0], 10000)
        self.assertEqual(mod.bootstrap_stickiness.__defaults__[0], 10000)

    def test_tac_per_turn_never_gets_significance_marker(self):
        mod = load_module()
        self.assertEqual(mod.sig_marker(0.0001, 1.0, "tac_per_turn"), "")
        self.assertEqual(mod.sig_marker(0.0001, -1.0, "tac_per_turn"), "")


if __name__ == "__main__":
    unittest.main()
