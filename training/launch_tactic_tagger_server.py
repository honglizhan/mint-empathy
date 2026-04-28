#!/usr/bin/env python3
"""
Launch a vLLM OpenAI-compatible server for TacticTagger on a dedicated GPU.

This server runs independently from the GRPO training process and provides
fast batch inference for tactic tagging via HTTP API.

Usage:
    # Launch on GPU 0 with adapters from Hugging Face (default)
    CUDA_VISIBLE_DEVICES=0 python launch_tactic_tagger_server.py

    # Or specify a local adapter directory
    python launch_tactic_tagger_server.py --adapter_base_dir ./lora_adapters --gpu 0

The server exposes an OpenAI-compatible API at http://localhost:8100/v1
"""

import os
import sys
import subprocess
from absl import app, flags
from huggingface_hub import snapshot_download
from termcolor import cprint

from reward_func_tactics_kl_bigram_entropy import TACTIC_NAMES

FLAGS = flags.FLAGS

flags.DEFINE_string('model_id', 'meta-llama/Llama-3.1-8B-Instruct', 'Base model ID')
flags.DEFINE_string('adapter_base_dir',
    'hongli-zhan/empathy-tactic-taggers-llama3.1-8b',
    'Local adapter directory or Hugging Face repo ID containing tactic LoRA adapters')
flags.DEFINE_string('hf_revision', 'main', 'Hugging Face revision for adapter_base_dir when it is a repo ID')
flags.DEFINE_integer('gpu', 0, 'GPU to use for the server')
flags.DEFINE_integer('port', 8100, 'Port for the server')
flags.DEFINE_string('host', '0.0.0.0', 'Host to bind to')
flags.DEFINE_float('gpu_memory_utilization', 0.95, 'GPU memory utilization')
flags.DEFINE_integer('max_model_len', 8192, 'Maximum model length')
flags.DEFINE_integer('max_lora_rank', 64, 'Maximum LoRA rank')
flags.DEFINE_integer('tensor_parallel_size', 1, 'Number of GPUs for tensor parallelism')


def resolve_adapter_base_dir(adapter_base_dir: str, hf_revision: str) -> str:
    """Return a local adapter directory, downloading from Hugging Face when needed."""
    if os.path.isdir(adapter_base_dir):
        return adapter_base_dir

    if "/" not in adapter_base_dir:
        return adapter_base_dir

    cprint(
        f"[INFO] Downloading tactic tagger adapters from {adapter_base_dir}@{hf_revision}",
        "cyan",
        force_color=True,
    )
    allow_patterns = [f"{tactic}/*" for tactic in TACTIC_NAMES]
    return snapshot_download(
        repo_id=adapter_base_dir,
        revision=hf_revision,
        allow_patterns=allow_patterns,
    )


def resolve_adapter_path(adapter_base_dir: str, tactic: str) -> str | None:
    """Resolve either the legacy local layout or the HF-style layout for one tactic."""
    candidates = [
        os.path.join(adapter_base_dir, f"Llama-3.1-8B-Instruct-tagger-{tactic}"),
        os.path.join(adapter_base_dir, tactic),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def build_lora_modules_arg(adapter_base_dir: str) -> str:
    """
    Build the --lora-modules argument for vLLM server.

    Format: name=path,name=path,...
    """
    lora_modules = []
    for tactic in TACTIC_NAMES:
        adapter_path = resolve_adapter_path(adapter_base_dir, tactic)
        if adapter_path:
            # vLLM uses name=path format
            lora_modules.append(f"{tactic}={adapter_path}")
        else:
            legacy_path = os.path.join(adapter_base_dir, f"Llama-3.1-8B-Instruct-tagger-{tactic}")
            hf_path = os.path.join(adapter_base_dir, tactic)
            cprint(
                f"[WARNING] Adapter not found for {tactic}: checked {legacy_path} and {hf_path}",
                "red",
            )

    return " ".join(lora_modules)


def check_gpu_memory(gpu_id: int) -> None:
    """Check and display GPU memory status before launching (nvidia-smi only, no torch init)."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits', f'--id={gpu_id}'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            used, total = result.stdout.strip().split(',')
            cprint(f"[GPU CHECK] GPU {gpu_id}: {used.strip()} MB used / {total.strip()} MB total", "cyan")
        else:
            cprint(f"[GPU CHECK] nvidia-smi failed for GPU {gpu_id}", "red")
    except FileNotFoundError:
        cprint("[GPU CHECK] nvidia-smi not found", "red")


def launch_server(
    model_id: str,
    adapter_base_dir: str,
    hf_revision: str,
    gpu: int,
    port: int,
    host: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_lora_rank: int,
    tensor_parallel_size: int,
):
    """Launch vLLM server with LoRA adapters for tactic tagging."""

    adapter_base_dir = resolve_adapter_base_dir(adapter_base_dir, hf_revision)

    # Set GPU - only if not already set by the calling shell
    # (avoids conflicts with vLLM's multiprocess architecture)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        cprint(f"[INFO] CUDA_VISIBLE_DEVICES already set to: {os.environ['CUDA_VISIBLE_DEVICES']}", "cyan")

    # Check GPU memory status before launching
    # Note: When CUDA_VISIBLE_DEVICES is set, GPU 0 in the subprocess = physical GPU in env var
    # So we always check GPU 0 from the subprocess perspective
    check_gpu_memory(0)

    # Build LoRA modules argument
    lora_modules_str = build_lora_modules_arg(adapter_base_dir)

    if not lora_modules_str:
        cprint("[ERROR] No LoRA adapters found!", "red")
        sys.exit(1)

    cprint(f"[INFO] Launching vLLM server on GPU {gpu}, port {port}", "cyan")
    cprint(f"[INFO] Model: {model_id}", "cyan")
    cprint(f"[INFO] Loaded {len(lora_modules_str.split())} LoRA adapters", "cyan")

    # Build command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--host", host,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--enable-lora",
        "--max-lora-rank", str(max_lora_rank),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--lora-modules", *lora_modules_str.split(),
    ]

    cprint(f"[INFO] Command: {' '.join(cmd)}", "cyan")
    cprint("=" * 60, "cyan")

    # Execute
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        cprint("\n[INFO] Server stopped by user", "cyan")
    except subprocess.CalledProcessError as e:
        cprint(f"[ERROR] Server failed: {e}", "red")
        sys.exit(1)


def main(_):
    launch_server(
        model_id=FLAGS.model_id,
        adapter_base_dir=FLAGS.adapter_base_dir,
        hf_revision=FLAGS.hf_revision,
        gpu=FLAGS.gpu,
        port=FLAGS.port,
        host=FLAGS.host,
        gpu_memory_utilization=FLAGS.gpu_memory_utilization,
        max_model_len=FLAGS.max_model_len,
        max_lora_rank=FLAGS.max_lora_rank,
        tensor_parallel_size=FLAGS.tensor_parallel_size,
    )


if __name__ == "__main__":
    app.run(main)