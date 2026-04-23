import importlib.util
import unittest
from pathlib import Path

from absl import flags

_MODULE = None


def _load_with_safe_absl_flags(spec):
    original_define_string = flags.DEFINE_string
    original_define_integer = flags.DEFINE_integer

    def safe_define_string(name, default, help_string, *args, **kwargs):
        if name in flags.FLAGS:
            return flags.FLAGS[name]
        return original_define_string(name, default, help_string, *args, **kwargs)

    def safe_define_integer(name, default, help_string, *args, **kwargs):
        if name in flags.FLAGS:
            return flags.FLAGS[name]
        return original_define_integer(name, default, help_string, *args, **kwargs)

    flags.DEFINE_string = safe_define_string
    flags.DEFINE_integer = safe_define_integer
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        flags.DEFINE_string = original_define_string
        flags.DEFINE_integer = original_define_integer


def load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    repo_root = Path(__file__).resolve().parents[2]
    mod_path = repo_root / 'evaluation' / 'step1_sample.py'
    spec = importlib.util.spec_from_file_location('step1_sample_mod', mod_path)
    mod = _load_with_safe_absl_flags(spec)
    _MODULE = mod
    return _MODULE


class VerbalizedSamplingParsingTest(unittest.TestCase):
    def test_parse_xml_attribute_responses(self):
        mod = load_module()
        raw_text = (
            '<response text="First reply." probability="0.3"/>\n'
            '<response text="Second reply." probability="0.2"/>\n'
            '<response text="Third reply." probability="0.5"/>'
        )
        candidates, reasoning = mod.parse_vs_json(raw_text)
        self.assertEqual(reasoning, '')
        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0]['text'], 'First reply.')
        self.assertEqual(candidates[1]['text'], 'Second reply.')
        self.assertEqual(candidates[2]['probability'], '0.5')

    def test_parse_nested_xml_responses(self):
        mod = load_module()
        raw_text = (
            '<response><text>Alpha</text><probability>0.2</probability></response>\n'
            '<response><text>Beta</text><probability>0.3</probability></response>\n'
            '<response><text>Gamma</text><probability>0.5</probability></response>'
        )
        candidates, _ = mod.parse_vs_json(raw_text)
        self.assertEqual(len(candidates), 3)
        self.assertEqual([c['text'] for c in candidates], ['Alpha', 'Beta', 'Gamma'])

    def test_parse_probability_lines(self):
        mod = load_module()
        raw_text = (
            'First candidate.\n(Probability: 0.8)\n'
            'Second candidate.\n(Probability: 0.1)\n'
            'Third candidate.\n(Probability: 0.1)'
        )
        candidates, _ = mod.parse_vs_json(raw_text)
        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0]['text'], 'First candidate.')
        self.assertEqual(candidates[1]['probability'], '0.1')
        self.assertEqual(candidates[2]['text'], 'Third candidate.')


if __name__ == '__main__':
    unittest.main()
