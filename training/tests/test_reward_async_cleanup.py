import asyncio
import os
import sys
import types
import unittest
from unittest.mock import patch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import reward_verl


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if not text:
            return []
        return [1] * max(1, len(text.split()))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "\n".join(message["content"] for message in messages)


class ForeignLoopSession:
    def __init__(self):
        self.closed = False
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1
        raise AssertionError("cross-loop close attempted")


class FakeResponseContext:
    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return self.payload


class FakeClientSession:
    instances = []

    def __init__(self, *args, **kwargs):
        self.closed = False
        self.close_calls = 0
        self.created_loop_id = id(asyncio.get_running_loop())
        FakeClientSession.instances.append(self)

    def post(self, url, json):
        return FakeResponseContext({"rewards": [0.0]})

    async def close(self):
        self.close_calls += 1
        current_loop_id = id(asyncio.get_running_loop())
        if current_loop_id != self.created_loop_id:
            raise AssertionError("session closed from the wrong loop")
        self.closed = True


class FakeAsyncOpenAI:
    instances = []

    def __init__(self, *args, **kwargs):
        self.close_calls = 0
        self.closed = False
        self.created_loop_id = id(asyncio.get_running_loop())
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )
        FakeAsyncOpenAI.instances.append(self)

    async def _create(self, **kwargs):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="<score>0</score>")
                )
            ]
        )

    async def close(self):
        self.close_calls += 1
        current_loop_id = id(asyncio.get_running_loop())
        if current_loop_id != self.created_loop_id:
            raise AssertionError("client closed from the wrong loop")
        self.closed = True


class RewardAsyncCleanupTest(unittest.TestCase):
    def setUp(self):
        FakeClientSession.instances = []
        FakeAsyncOpenAI.instances = []
        self._clear_thread_local()
        reward_verl._jsonl_buffer = []
        reward_verl._jsonl_call_counter = 0

    def tearDown(self):
        self._clear_thread_local()

    def _clear_thread_local(self):
        for name in [
            "clients",
            "client_cycle",
            "server_urls",
            "client_timeout_s",
            "client_loop_id",
            "rm_session",
            "rm_session_timeout_s",
            "rm_session_loop_id",
        ]:
            if hasattr(reward_verl._tl, name):
                delattr(reward_verl._tl, name)

    def test_get_session_does_not_close_foreign_loop_session(self):
        reward_verl._tl.rm_session = ForeignLoopSession()
        reward_verl._tl.rm_session_timeout_s = 20.0
        reward_verl._tl.rm_session_loop_id = -1

        async def run_test():
            with patch.object(reward_verl.aiohttp, "ClientSession", FakeClientSession):
                session = await reward_verl._get_session(20.0)
            return session

        session = asyncio.run(run_test())

        self.assertIsInstance(session, FakeClientSession)
        self.assertEqual(reward_verl._tl.rm_session.close_calls, 0)
        self.assertIs(reward_verl._tl.rm_session, session)

    def test_compute_score_closes_loop_scoped_async_resources(self):
        fake_tactic_info = {
            "validation": {
                "system_prompt": "You are a tagger.",
                "user_template": "Response: {Full_Response}\nSentence: {Sentence}\nScore?",
            }
        }

        kwargs = {
            "reward_mode": "quality_plus_diversity",
            "rm_server_url": "http://localhost:5100",
            "tactic_tagger_server_url": "http://localhost:8100/v1,http://localhost:8101/v1,http://localhost:8102/v1",
            "tactic_tagger_prompts_dir": "/tmp/fake_prompts",
            "kl_gamma": 1.0,
            "bigram_gamma": 0.0,
            "trigram_gamma": 0.0,
            "entropy_gamma": 0.0,
            "diversity_weight": 1.0,
            "smoothing_alpha": 0.1,
            "tactic_tagger_temperature": 0.1,
            "tactic_tagger_max_tokens": 64,
            "tactic_tagger_max_concurrent": 1,
        }

        with patch.object(reward_verl, "_get_tokenizer", return_value=FakeTokenizer()), \
             patch.object(reward_verl, "_get_tactic_info", return_value=fake_tactic_info), \
             patch.object(reward_verl, "sent_tokenize", lambda text: [text]), \
             patch.object(reward_verl.aiohttp, "ClientSession", FakeClientSession), \
             patch.object(reward_verl, "AsyncOpenAI", FakeAsyncOpenAI):
            results = reward_verl.compute_score(
                data_sources=["source"],
                solution_strs=["I hear you and I understand."],
                ground_truths=[""],
                extra_infos=[{"prompt_text": "prompt"}],
                **kwargs,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(FakeClientSession.instances), 1)
        self.assertEqual(len(FakeAsyncOpenAI.instances), 3)
        self.assertTrue(all(client.close_calls == 1 for client in FakeAsyncOpenAI.instances))
        self.assertTrue(all(client.closed for client in FakeAsyncOpenAI.instances))
        self.assertEqual(FakeClientSession.instances[0].close_calls, 1)
        self.assertTrue(FakeClientSession.instances[0].closed)
        self.assertFalse(hasattr(reward_verl._tl, "clients"))
        self.assertFalse(hasattr(reward_verl._tl, "rm_session"))


if __name__ == "__main__":
    unittest.main()
