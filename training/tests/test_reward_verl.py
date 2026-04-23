"""
Tests for the batch VERL reward function (reward_verl.py).

Tests mock the tagger server via AsyncMock to avoid needing a live server.
Run: python -m pytest training/tests/test_reward_verl.py
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

from termcolor import cprint

# Ensure sibling modules are importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from reward_func_tactics_kl_bigram_entropy import (
    TACTIC_NAMES,
    clean_history_structs,
    compute_bigram_surprisal,
    compute_kl_divergence,
    compute_trigram_surprisal,
    compute_within_turn_entropy,
)
from reward_verl import _build_tag_requests, _tag_one, compute_score

# Default reward kwargs matching paper's diversity_only config
REWARD_KWARGS = {
    "reward_mode": "diversity_only",
    "tactic_tagger_server_url": "http://localhost:8100/v1",
    "tactic_tagger_prompts_dir": "/tmp/fake_prompts/",
    "kl_gamma": 1.0,
    "bigram_gamma": 0.0,
    "trigram_gamma": 0.0,
    "entropy_gamma": 0.0,
    "smoothing_alpha": 0.1,
    "tactic_tagger_temperature": 0.1,
    "tactic_tagger_max_tokens": 64,
}


def make_mock_response(content):
    """Create a mock OpenAI chat completion response."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    return mock


def run_with_mock_tagger(solution_strs, extra_infos, tagger_responses):
    """
    Run compute_score with a mocked async tagger (batch interface).

    solution_strs: list of response strings
    extra_infos: list of extra_info dicts
    tagger_responses: dict mapping (sentence_substring, tactic_name) -> score (0 or 1)
    """
    if isinstance(solution_strs, str):
        solution_strs = [solution_strs]
    if isinstance(extra_infos, dict) or extra_infos is None:
        extra_infos = [extra_infos]

    async def mock_create(**kwargs):
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        user_msg = messages[-1]["content"] if messages else ""

        sentence_part = user_msg
        if "Sentence:" in user_msg:
            sentence_part = user_msg.split("Sentence:")[-1]

        for (substr, tactic), score in tagger_responses.items():
            if tactic == model and substr in sentence_part:
                return make_mock_response(f"<score>{score}</score>")
        return make_mock_response("<score>0</score>")

    fake_tactic_info = {}
    for tactic in TACTIC_NAMES:
        fake_tactic_info[tactic] = {
            "system_prompt": "You are a tagger.",
            "user_template": "Response: {Full_Response}\nSentence: {Sentence}\nScore?",
        }

    import reward_verl
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)

    old_client = reward_verl._client
    old_tactic_info = reward_verl._tactic_info
    reward_verl._client = mock_client
    reward_verl._client_url = REWARD_KWARGS["tactic_tagger_server_url"]
    reward_verl._tactic_info = fake_tactic_info
    reward_verl._tactic_info_dir = REWARD_KWARGS["tactic_tagger_prompts_dir"]

    try:
        n = len(solution_strs)
        results = compute_score(
            data_sources=["tactic_diversity"] * n,
            solution_strs=solution_strs,
            ground_truths=[""] * n,
            extra_infos=extra_infos,
            **REWARD_KWARGS,
        )
    finally:
        reward_verl._client = old_client
        reward_verl._tactic_info = old_tactic_info

    return results


def test_return_format():
    """Test that compute_score returns a list of floats (batch interface)."""
    cprint("Test: return format (batch)", "cyan", force_color=True)

    results = run_with_mock_tagger(
        solution_strs=["I understand your feelings. Let me help you."],
        extra_infos=[{"prompt_text": "p1", "tactic_history_counts": {}, "tactic_history_bigrams": {}, "tactic_history_trigrams": {}, "num_previous_assistant_turns": 0}],
        tagger_responses={("understand", "validation"): 1, ("help", "assistance"): 1},
    )

    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert isinstance(results[0], (int, float)), f"Score should be numeric, got {type(results[0])}"

    cprint("  PASS", "green", force_color=True)


def test_batch_multiple():
    """Test that batch with multiple samples returns correct count."""
    cprint("Test: batch with 2 samples", "cyan", force_color=True)

    results = run_with_mock_tagger(
        solution_strs=["I understand.", "That sounds hard."],
        extra_infos=[
            {"prompt_text": "p1", "tactic_history_counts": {}, "tactic_history_bigrams": {}},
            {"prompt_text": "p2", "tactic_history_counts": {}, "tactic_history_bigrams": {}},
        ],
        tagger_responses={("understand", "validation"): 1, ("hard", "validation"): 1},
    )

    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"

    cprint("  PASS", "green", force_color=True)


def test_empty_response():
    """Test that empty/blank responses return zero score."""
    cprint("Test: empty response", "cyan", force_color=True)

    for sol in ["", "   "]:
        results = run_with_mock_tagger(
            solution_strs=[sol],
            extra_infos=[{"prompt_text": "p1", "tactic_history_counts": {}, "tactic_history_bigrams": {}}],
            tagger_responses={},
        )
        assert results[0] == 0.0, f"Expected 0.0 for empty response '{sol}', got {results[0]}"

    cprint("  PASS", "green", force_color=True)


def test_thinking_model_stripping():
    """Test that <think>...</think> blocks are stripped before scoring."""
    cprint("Test: thinking model stripping", "cyan", force_color=True)

    results = run_with_mock_tagger(
        solution_strs=["<think>Let me think about this carefully.</think>I hear you."],
        extra_infos=[{"prompt_text": "p1", "tactic_history_counts": {}, "tactic_history_bigrams": {}}],
        tagger_responses={("hear you", "validation"): 1},
    )

    assert results[0] > 0.0

    cprint("  PASS", "green", force_color=True)


def test_clean_history_structs_filters_none_values():
    """Test shared helper that cleans None values in nested history dicts."""
    cprint("Test: clean history helper", "cyan", force_color=True)

    hist_counts, hist_bigrams, hist_trigrams = clean_history_structs(
        {"validation": 2, "advice": None},
        {"validation": {"advice": 3, "questioning": None}, "advice": None},
        {"validation": {"advice": {"questioning": 1, "empowerment": None}, "questioning": None}, "advice": None},
    )

    assert hist_counts == {"validation": 2}
    assert hist_bigrams == {"validation": {"advice": 3}}
    assert hist_trigrams == {"validation": {"advice": {"questioning": 1}}}

    cprint("  PASS", "green", force_color=True)


def test_tag_one_returns_0_on_exception():
    """Test that _tag_one gracefully returns 0 when tagger server fails."""
    cprint("Test: _tag_one exception handling", "cyan", force_color=True)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=ConnectionError("server down"))
    sem = asyncio.Semaphore(4)
    result = asyncio.run(_tag_one(mock_client, "validation", [], 0.1, 64, sem))
    assert result == 0

    cprint("  PASS", "green", force_color=True)


def run_all_tests():
    tests = [
        test_return_format,
        test_batch_multiple,
        test_empty_response,
        test_thinking_model_stripping,
        test_clean_history_structs_filters_none_values,
        test_tag_one_returns_0_on_exception,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            cprint(f"  FAIL: {e}", "red", force_color=True)
            failed += 1

    cprint(f"\nResults: {passed} passed, {failed} failed out of {len(tests)} tests", "cyan", force_color=True)
    if failed == 0:
        cprint("All tests passed!", "green", force_color=True)
    else:
        cprint("Some tests FAILED.", "red", force_color=True)
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
