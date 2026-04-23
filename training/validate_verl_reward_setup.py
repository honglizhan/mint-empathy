#!/usr/bin/env python3
"""
Cheap preflight for VERL Hydra overrides and reward-manager loading.

This catches config drift before a job spends time starting Ray or remote
services. It intentionally validates only schema and import/load behavior.
"""

from __future__ import annotations

import os
import sys

from hydra import compose, initialize_config_dir

import verl.trainer.main_ppo as main_ppo
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.ppo.reward import load_reward_manager


class _DummyTokenizer:
    def decode(self, *args, **kwargs):
        return ""


def _require_file(path: str, label: str) -> None:
    if not path:
        raise ValueError(f"{label} is empty")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _require_dir(path: str, label: str) -> None:
    if not path:
        raise ValueError(f"{label} is empty")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _tagger_request_budget_s(reward_kwargs: dict) -> float:
    timeout_s = float(reward_kwargs.get("tactic_tagger_timeout_s", 20.0))
    max_retries = max(1, int(reward_kwargs.get("tactic_tagger_max_retries", 6)))
    base_delay_s = float(reward_kwargs.get("tactic_tagger_retry_base_delay_s", 1.0))
    backoff_s = sum(min(base_delay_s * (2**attempt), 30.0) for attempt in range(max_retries - 1))
    return timeout_s * max_retries + backoff_s


def _rm_request_budget_s(reward_kwargs: dict) -> float:
    timeout_s = float(reward_kwargs.get("rm_timeout_s", 20.0))
    max_retries = max(1, int(reward_kwargs.get("rm_max_retries", 6)))
    base_delay_s = float(reward_kwargs.get("rm_retry_base_delay_s", 1.0))
    backoff_s = sum(min(base_delay_s * (2**attempt), 30.0) for attempt in range(max_retries - 1))
    return timeout_s * max_retries + backoff_s


def _validate_rm_runtime_import() -> None:
    cuda_home = os.environ.get("CUDA_HOME", "").strip()
    if not cuda_home:
        raise ValueError("CUDA_HOME must be set when the reward mode requires the RM server.")
    if not os.path.isdir(cuda_home):
        raise FileNotFoundError(f"CUDA_HOME does not exist: {cuda_home}")

    import openrlhf.cli.serve_rm  # noqa: F401


def main(argv: list[str]) -> int:
    config_dir = os.path.join(os.path.dirname(main_ppo.__file__), "config")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(config_name="ppo_trainer", overrides=argv)

    config = migrate_legacy_reward_impl(config)

    reward_cfg = config.reward
    custom_reward_cfg = reward_cfg.custom_reward_function
    reward_manager_cfg = reward_cfg.reward_manager
    reward_kwargs = dict(custom_reward_cfg.get("reward_kwargs", {}))

    _require_file(config.data.train_files, "data.train_files")
    _require_file(config.data.val_files, "data.val_files")
    _require_file(custom_reward_cfg.path, "reward.custom_reward_function.path")

    prompts_dir = str(reward_kwargs.get("tactic_tagger_prompts_dir", ""))
    if prompts_dir:
        _require_dir(prompts_dir, "reward.custom_reward_function.reward_kwargs.tactic_tagger_prompts_dir")

    if reward_manager_cfg.source == "importlib":
        _require_file(reward_manager_cfg.module.path, "reward.reward_manager.module.path")

    manager = load_reward_manager(config, tokenizer=_DummyTokenizer())

    if manager.__class__.__name__ == "UidBatchRewardManager" and int(reward_cfg.num_workers) != 1:
        raise ValueError(
            "UidBatchRewardManager requires reward.num_workers=1. "
            f"Got reward.num_workers={reward_cfg.num_workers}."
        )

    reward_mode = str(reward_kwargs.get("reward_mode", "diversity_only"))
    needs_tagger = reward_mode in {"diversity_only", "quality_plus_diversity", "quality_times_diversity"}
    needs_rm = reward_mode != "diversity_only"

    if needs_rm:
        rm_server_url = str(reward_kwargs.get("rm_server_url", ""))
        if not rm_server_url:
            raise ValueError("reward.custom_reward_function.reward_kwargs.rm_server_url is required.")
        _validate_rm_runtime_import()

    if manager.__class__.__name__ == "UidBatchRewardManager" and (needs_tagger or needs_rm):
        uid_batch_timeout_s = float(reward_cfg.get("uid_batch_timeout_s", 120.0))
        request_budgets = []
        if needs_tagger:
            request_budgets.append(("tagger", _tagger_request_budget_s(reward_kwargs)))
        if needs_rm:
            request_budgets.append(("rm", _rm_request_budget_s(reward_kwargs)))
        worst_label, worst_budget_s = max(request_budgets, key=lambda item: item[1])
        if uid_batch_timeout_s <= worst_budget_s:
            raise ValueError(
                "reward.uid_batch_timeout_s is too low for the configured reward request retry budget. "
                f"uid_batch_timeout_s={uid_batch_timeout_s}, "
                f"worst_case_single_{worst_label}_request_s={worst_budget_s:.1f}."
            )

    print("VERL preflight OK")
    print(f"Reward manager: {manager.__class__.__module__}.{manager.__class__.__name__}")
    print(f"Reward function: {custom_reward_cfg.path}:{custom_reward_cfg.name}")
    print(f"Reward workers: {reward_cfg.num_workers}")
    print(f"Train rollout n: {config.actor_rollout_ref.rollout.n}")
    print(f"Validation rollout n: {config.actor_rollout_ref.rollout.val_kwargs.n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))