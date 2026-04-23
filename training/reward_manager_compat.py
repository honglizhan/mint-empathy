"""
Compatibility reward manager for current VERL reward-loop builds.

This manager preserves the legacy "batch reward" behavior used by
`reward_verl.py` by grouping repeated rollouts with the same VERL `uid` and
calling the custom reward function once per uid group.

It is intended for configurations where:
  - `reward.reward_manager.source=importlib`
  - `reward.reward_manager.name=UidBatchRewardManager`
  - `reward.num_workers=1`
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


def _as_dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    raise TypeError(f"Expected {field_name} to be dict-like, got {type(value).__name__}")


class UidBatchRewardManager(RewardManagerBase):
    _pending_groups: dict[str, dict[str, Any]] = {}
    _pending_lock = threading.Lock()

    @classmethod
    def init_class(cls, config, tokenizer):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        cls._pending_groups = {}
        cls._pending_lock = threading.Lock()

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        score_fn = getattr(self.compute_score, "func", self.compute_score)
        self.is_async_reward_score = inspect.iscoroutinefunction(score_fn)
        self.group_timeout_s = float(config.reward.get("uid_batch_timeout_s", 120.0))

        num_workers = int(config.reward.num_workers)
        if num_workers != 1:
            raise ValueError(
                "UidBatchRewardManager requires reward.num_workers=1 so all repeated rollouts "
                f"for a uid land on the same reward worker. Got reward.num_workers={num_workers}."
            )

    def _expected_group_size(self, data: DataProto) -> int:
        validate = bool(data.meta_info.get("validate", False))
        if validate:
            return int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
        return int(self.config.actor_rollout_ref.rollout.n)

    async def _decode_response(self, data: DataProto) -> str:
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        return await self.loop.run_in_executor(None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True))

    async def _score_group(self, uid: str, samples: list[dict[str, Any]]) -> None:
        try:
            kwargs = {
                "data_sources": [sample["data_source"] for sample in samples],
                "solution_strs": [sample["solution_str"] for sample in samples],
                "ground_truths": [sample["ground_truth"] for sample in samples],
                "extra_infos": [sample["extra_info"] for sample in samples],
            }
            if self.is_async_reward_score:
                outputs = await self.compute_score(**kwargs)
            else:
                outputs = await self.loop.run_in_executor(None, lambda: self.compute_score(**kwargs))

            if not isinstance(outputs, list):
                raise TypeError(f"Expected batched reward outputs to be a list, got {type(outputs).__name__}")
            if len(outputs) != len(samples):
                raise ValueError(
                    f"Reward output length mismatch for uid={uid}: "
                    f"expected {len(samples)} results, got {len(outputs)}."
                )

            for sample, output in zip(samples, outputs, strict=True):
                score = output["score"] if isinstance(output, dict) else float(output)
                reward_extra_info = dict(output) if isinstance(output, dict) else {"acc": score}
                if not sample["future"].done():
                    sample["future"].set_result({"reward_score": score, "reward_extra_info": reward_extra_info})
        except Exception as exc:
            for sample in samples:
                if not sample["future"].done():
                    sample["future"].set_exception(exc)

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        assert len(data) == 1, "UidBatchRewardManager only supports single data items"
        data_item = data[0]

        uid_value = data_item.non_tensor_batch.get("uid", None)
        if uid_value is None:
            raise ValueError("UidBatchRewardManager requires `uid` in non_tensor_batch.")

        uid = str(uid_value)
        expected_size = self._expected_group_size(data)
        if expected_size <= 0:
            raise ValueError(f"Invalid expected group size for uid={uid}: {expected_size}")

        reward_model = _as_dict(data_item.non_tensor_batch.get("reward_model", {}), "reward_model")
        extra_info = _as_dict(data_item.non_tensor_batch.get("extra_info", {}), "extra_info")
        tool_extra_fields = _as_dict(data_item.non_tensor_batch.get("tool_extra_fields", None), "tool_extra_fields")
        if tool_extra_fields:
            extra_info.update(tool_extra_fields)

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = _as_dict(data_item.non_tensor_batch.get("reward_scores", {}), "reward_scores")
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        sample = {
            "data_source": data_item.non_tensor_batch["data_source"],
            "solution_str": await self._decode_response(data),
            "ground_truth": reward_model.get("ground_truth", None),
            "extra_info": extra_info,
            "future": asyncio.get_running_loop().create_future(),
        }

        launch_group = None
        with self.__class__._pending_lock:
            group = self.__class__._pending_groups.setdefault(
                uid,
                {"samples": [], "expected_size": expected_size, "created_at": time.monotonic()},
            )
            if group["expected_size"] != expected_size:
                raise ValueError(
                    f"Mismatched expected group size for uid={uid}: "
                    f"{group['expected_size']} vs {expected_size}."
                )

            group["samples"].append(sample)
            if len(group["samples"]) == expected_size:
                launch_group = group["samples"]
                del self.__class__._pending_groups[uid]
            elif len(group["samples"]) > expected_size:
                raise ValueError(
                    f"Received too many samples for uid={uid}: "
                    f"expected {expected_size}, got {len(group['samples'])}."
                )

        if launch_group is not None:
            asyncio.create_task(self._score_group(uid, launch_group))

        try:
            return await asyncio.wait_for(sample["future"], timeout=self.group_timeout_s)
        except asyncio.TimeoutError as exc:
            with self.__class__._pending_lock:
                group = self.__class__._pending_groups.get(uid)
                if group is not None:
                    group["samples"] = [item for item in group["samples"] if item["future"] is not sample["future"]]
                    if not group["samples"]:
                        del self.__class__._pending_groups[uid]
                    seen = len(group["samples"])
                else:
                    seen = expected_size
            raise TimeoutError(
                f"Timed out waiting for rollout group uid={uid}. "
                f"Expected {expected_size} samples, observed {seen} pending after cleanup. "
                "Check rollout.n / val_kwargs.n and keep reward.num_workers=1."
            ) from exc
