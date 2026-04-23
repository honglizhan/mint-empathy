import asyncio
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from reward_manager_compat import UidBatchRewardManager
from verl import DataProto


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return f"resp-{int(token_ids[0].item())}"


def _make_dataproto(uid, response_token, validate=False):
    batch = TensorDict(
        {
            "prompts": torch.tensor([[1, 2]], dtype=torch.long),
            "responses": torch.tensor([[response_token, response_token + 1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        },
        batch_size=[1],
    )
    non_tensor_batch = {
        "uid": np.array([uid], dtype=object),
        "data_source": np.array(["tactic_diversity"], dtype=object),
        "reward_model": np.array([{"ground_truth": "", "style": "rule"}], dtype=object),
        "extra_info": np.array([{"prompt_text": "prompt"}], dtype=object),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"validate": validate})


class RewardManagerCompatTest(unittest.IsolatedAsyncioTestCase):
    async def test_uid_batch_groups_rollouts(self):
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {"rollout": {"n": 2, "val_kwargs": {"n": 1}}},
                "reward": {"num_workers": 1, "uid_batch_timeout_s": 5.0},
            }
        )

        calls = []

        def compute_score(data_sources, solution_strs, ground_truths, extra_infos=None, **kwargs):
            calls.append((list(data_sources), list(solution_strs)))
            return [
                {"score": 0.25, "index": 0, "response": solution_strs[0]},
                {"score": 0.75, "index": 1, "response": solution_strs[1]},
            ]

        manager = UidBatchRewardManager(config, _FakeTokenizer(), compute_score)
        data_a = _make_dataproto("uid-1", 10)
        data_b = _make_dataproto("uid-1", 20)

        result_a, result_b = await asyncio.gather(manager.run_single(data_a), manager.run_single(data_b))

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], ["tactic_diversity", "tactic_diversity"])
        self.assertEqual(result_a["reward_score"], 0.25)
        self.assertEqual(result_b["reward_score"], 0.75)
        self.assertEqual(result_a["reward_extra_info"]["response"], "resp-10")
        self.assertEqual(result_b["reward_extra_info"]["response"], "resp-20")

    async def test_validation_uses_val_rollout_n(self):
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {"rollout": {"n": 8, "val_kwargs": {"n": 1}}},
                "reward": {"num_workers": 1, "uid_batch_timeout_s": 5.0},
            }
        )

        def compute_score(data_sources, solution_strs, ground_truths, extra_infos=None, **kwargs):
            return [{"score": 1.0}]

        manager = UidBatchRewardManager(config, _FakeTokenizer(), compute_score)
        result = await manager.run_single(_make_dataproto("uid-val", 30, validate=True))
        self.assertEqual(result["reward_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
