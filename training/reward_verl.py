"""
Unified VERL-compatible batch reward function with configurable reward modes.

Supports 5 reward modes via the `reward_mode` kwarg:
  - diversity_only:           tactic KL + entropy (no RM)
  - quality_plus_diversity:   quality_norm + diversity_weight * diversity_norm
  - quality_times_diversity:  quality_norm * diversity_norm
  - quality_only:             RM quality score only (per-group normalized)
  - quality_x_r1_zero_div:   RM quality score only (per-group normalized, VERL native entropy handles diversity)

All modes use per-group min-max normalization, multiplicative length scaling,
and JSONL sample logging. Modes that include diversity use sentence-level
tactic tagging via HTTP server.

Batch interface: sync `compute_score(data_sources, solution_strs, ground_truths, extra_infos, **kwargs)`
that calls `asyncio.run()` internally. In the current VERL build used here, the
launcher wires this through VERL's batch reward manager with:
    custom_reward_function.path=/abs/path/reward_verl.py
    custom_reward_function.name=compute_score
    reward_manager.name=batch

The **kwargs come from Hydra config:
    custom_reward_function.reward_kwargs.reward_mode=quality_plus_diversity
    custom_reward_function.reward_kwargs.rm_server_url=http://localhost:5100
    custom_reward_function.reward_kwargs.tactic_tagger_server_url=http://localhost:8100/v1
    etc.
"""

import asyncio
import itertools
import json
import os
import re
import sys
import threading
from collections import Counter

# Ensure sibling modules are importable when loaded by VERL/Ray
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import aiohttp
from nltk.tokenize import sent_tokenize
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from reward_func_tactics_kl_bigram_entropy import (
    TACTIC_NAMES,
    clean_history_structs,
    compute_bigram_surprisal,
    compute_kl_divergence,
    compute_trigram_surprisal,
    compute_within_turn_entropy,
    extract_response_from_thinking_model,
    load_tactic_info,
    parse_tactic_score,
)

VALID_REWARD_MODES = {
    "diversity_only",
    "quality_plus_diversity",
    "quality_times_diversity",
    "quality_only",
    "quality_x_r1_zero_div",
}

# ---------------------------------------------------------------------------
# Global lazy-init state (one per worker process)
# ---------------------------------------------------------------------------

# RM (Skywork-Reward-V2)
_tokenizer = None
_TOKENIZER_NAME = os.environ.get("REWARD_TOKENIZER", "meta-llama/Llama-3.1-8B-Instruct")

# Thread-local storage for async clients (each asyncio.run() gets its own event loop,
# so aiohttp sessions and AsyncOpenAI clients must not be shared across loops/threads)
_tl = threading.local()

# Tactic tagger (shared, non-async state)
_tactic_info = None
_tactic_info_dir = None

_init_lock = threading.Lock()

# Tagger failure tracking
_tagger_fail_count = 0
_tagger_fail_lock = threading.Lock()

# JSONL sample logging
_jsonl_buffer = []
_jsonl_call_counter = 0
_jsonl_lock = threading.Lock()

# Per-worker tagger concurrency limiter (semaphore created per asyncio.run call)

# Tactic label detection (format violation penalty)
_TACTIC_NAMES_RE = (
    r"validation|validate|validating"
    r"|advice|advising"
    r"|paraphrasing|paraphrase"
    r"|empowerment|empower|empowering"
    r"|reappraisal|reappraise|reframing"
    r"|emotional[_ ]expression|express(?:ing)? emotion"
    r"|self[_ -]disclosure|self[_ -]disclose"
    r"|questioning"
    r"|information|informing"
    r"|assistance|assisting"
)
_TACTIC_LABEL_RE = re.compile(
    r"\*\*(?:" + _TACTIC_NAMES_RE + r")\*\*"
    r"|\*(?:" + _TACTIC_NAMES_RE + r")\*"
    r"|\d+\.\s*\*{1,2}(?:" + _TACTIC_NAMES_RE + r")"
    r"|##?#?\s*(?:" + _TACTIC_NAMES_RE + r")"
    r"|\((?:" + _TACTIC_NAMES_RE + r")\)"
    r"|(?:empathy\s+)?tactic(?:s)?\s*(?:used|count|so far|in this)"
    r"|tactics\s*:",
    re.IGNORECASE,
)
_FORMAT_VIOLATION_PENALTY = -2.0

# Multiplicative length scaling
_LENGTH_TARGET = 200


def _count_tokens(response_text, tokenizer):
    return len(tokenizer.encode(response_text, add_special_tokens=False))


def _compute_length_scale(num_tokens):
    return min(1.0, _LENGTH_TARGET / max(num_tokens, 1))


# ---------------------------------------------------------------------------
# Lazy init helpers
# ---------------------------------------------------------------------------

def _get_tokenizer():
    global _tokenizer
    with _init_lock:
        if _tokenizer is None:
            local_only = os.environ.get("TOKENIZER_LOCAL_ONLY", "false").lower() == "true"
            _tokenizer = AutoTokenizer.from_pretrained(
                _TOKENIZER_NAME, local_files_only=local_only
            )
        return _tokenizer


def _strip_think_blocks(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def _get_session(timeout_s):
    current_loop_id = id(asyncio.get_running_loop())
    session = getattr(_tl, 'rm_session', None)
    tl_timeout = getattr(_tl, 'rm_session_timeout_s', None)
    tl_loop_id = getattr(_tl, 'rm_session_loop_id', None)
    if (session is None or session.closed or tl_timeout != timeout_s
            or tl_loop_id != current_loop_id):
        if session is not None and not session.closed:
            try:
                await session.close()
            except Exception:
                pass
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_s)
        )
        _tl.rm_session = session
        _tl.rm_session_timeout_s = timeout_s
        _tl.rm_session_loop_id = current_loop_id
    return session


async def _call_rm(server_url, text, timeout_s=20.0, max_retries=6, base_delay=1.0):
    session = await _get_session(timeout_s)
    url = f"{server_url}/get_reward"
    payload = {"query": [text]}
    for attempt in range(max_retries):
        try:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return float(data["rewards"][0])
        except Exception as e:
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), 30.0)
                await asyncio.sleep(delay)
            else:
                print(f"[reward_verl] WARNING: RM server failed after {max_retries} retries: {e}")
                return 0.0


def _get_clients(server_url_csv, timeout_s):
    urls = [u.strip() for u in server_url_csv.split(",") if u.strip()]
    current_loop_id = id(asyncio.get_running_loop())
    tl_urls = getattr(_tl, 'server_urls', None)
    tl_timeout = getattr(_tl, 'client_timeout_s', None)
    tl_loop_id = getattr(_tl, 'client_loop_id', None)
    if tl_urls != urls or tl_timeout != timeout_s or tl_loop_id != current_loop_id:
        _tl.clients = [
            AsyncOpenAI(base_url=u, api_key="EMPTY", timeout=timeout_s)
            for u in urls
        ]
        _tl.server_urls = urls
        _tl.client_cycle = itertools.cycle(range(len(_tl.clients)))
        _tl.client_timeout_s = timeout_s
        _tl.client_loop_id = current_loop_id
    return _tl.clients


def _next_client():
    return _tl.clients[next(_tl.client_cycle)]


async def cleanup_clients():
    """Close AsyncOpenAI clients and aiohttp session. Call at end of training."""
    clients = getattr(_tl, 'clients', None)
    if clients:
        for client in clients:
            await client.close()
        _tl.clients = None
        _tl.server_urls = None
    session = getattr(_tl, 'rm_session', None)
    if session and not session.closed:
        await session.close()
        _tl.rm_session = None


def _get_tactic_info(prompts_dir):
    global _tactic_info, _tactic_info_dir
    with _init_lock:
        if _tactic_info is None or _tactic_info_dir != prompts_dir:
            _tactic_info = load_tactic_info(prompts_dir)
            _tactic_info_dir = prompts_dir
        return _tactic_info


# _get_tagger_semaphore removed: semaphore now created in _compute_score_async


def _extract_think_block(text):
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Sample logging
# ---------------------------------------------------------------------------

def _buffer_sample(result, response, think_text, curr_counts, hist_counts,
                    log_interval, log_dir, prompt_text):
    global _jsonl_buffer, _jsonl_call_counter
    if log_interval <= 0 or not log_dir:
        return

    sample = {
        "prompt": prompt_text,
        "response": response,
        "think": think_text,
        "score": result["score"],
        "think_length": result.get("think_length", 0),
    }
    # Include available metrics
    for key in ("quality_raw", "quality_norm", "diversity_norm",
                "kl_raw", "kl_norm", "entropy_raw", "entropy_norm",
                "bigram_raw", "trigram_raw", "num_tactics",
                "tactic_overlap_pct", "new_tactic_pct"):
        if key in result:
            sample[key] = result[key]

    with _jsonl_lock:
        _jsonl_buffer.append(sample)
        _jsonl_call_counter += 1
        if _jsonl_call_counter % log_interval == 0:
            _flush_jsonl(log_dir)


def _flush_jsonl(log_dir):
    global _jsonl_buffer
    if not _jsonl_buffer:
        return
    try:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"reward_samples_pid{os.getpid()}.jsonl")
        with open(path, "a") as f:
            for sample in _jsonl_buffer:
                f.write(json.dumps(sample) + "\n")
    except Exception as e:
        print(f"[reward_verl] WARNING: JSONL sample logging failed: {e}")
    _jsonl_buffer = []


# ---------------------------------------------------------------------------
# Tagger request helpers
# ---------------------------------------------------------------------------

async def _tag_one(
    tactic_name,
    messages,
    temperature,
    max_tokens,
    semaphore,
    max_retries=6,
    base_delay=1.0,
):
    global _tagger_fail_count
    async with semaphore:
        for attempt in range(max_retries):
            client = _next_client()
            try:
                resp = await client.chat.completions.create(
                    model=tactic_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                )
                return parse_tactic_score(resp.choices[0].message.content)
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), 30.0)
                    await asyncio.sleep(delay)
                else:
                    with _tagger_fail_lock:
                        _tagger_fail_count += 1
                        if _tagger_fail_count <= 3:
                            print(f"[reward_verl] WARNING: tagger request failed after {max_retries} retries ({tactic_name}): {e}")
                        elif _tagger_fail_count == 100:
                            print(f"[reward_verl] WARNING: {_tagger_fail_count} tagger failures so far. Server may be down.")
                    return 0


def _build_tag_requests(sentences, tactic_info, response):
    requests = []
    for sent_idx, sentence in enumerate(sentences):
        for tactic_name, info in tactic_info.items():
            user_msg = (
                info["user_template"]
                .replace("{Full_Response}", response)
                .replace("{Sentence}", sentence)
            )
            messages = [
                {"role": "system", "content": info["system_prompt"]},
                {"role": "user", "content": user_msg},
            ]
            requests.append((sent_idx, tactic_name, messages))
    return requests


# ---------------------------------------------------------------------------
# Per-sample raw computation
# ---------------------------------------------------------------------------

async def _compute_single_raw(solution_str, extra_info,
                               reward_mode, rm_server_url,
                               tagger_server_url, prompts_dir,
                               bigram_gamma, trigram_gamma, smoothing_alpha,
                               temperature, max_tokens, semaphore,
                               rm_timeout_s, rm_max_retries, rm_retry_base_delay_s,
                               tagger_timeout_s, tagger_max_retries, tagger_retry_base_delay_s):
    """Compute raw metrics for a single sample. Returns dict with raw values."""
    needs_rm = reward_mode != "diversity_only"
    needs_tagger = reward_mode in ("diversity_only", "quality_plus_diversity", "quality_times_diversity")

    extra = extra_info or {}
    prompt_text = extra.get("prompt_text", "")

    raw_text = solution_str or ""
    think_text = _extract_think_block(raw_text)
    think_length = len(think_text)

    if needs_tagger:
        response = extract_response_from_thinking_model(raw_text)
    else:
        response = _strip_think_blocks(raw_text)

    result = {
        "prompt_text": prompt_text,
        "response": response,
        "think_text": think_text,
        "think_length": think_length,
        "num_tokens": 0,
        "has_format_violation": False,
    }

    if not response.strip():
        if needs_rm:
            result["quality_raw"] = 0.0
        if needs_tagger:
            result.update({"kl_raw": 0.0, "entropy_raw": 0.0, "bigram_raw": 0.0,
                           "trigram_raw": 0.0, "curr_counts": {}, "hist_counts": {},
                           "num_tactics": 0})
        return result

    tokenizer = _get_tokenizer()
    num_tokens = _count_tokens(response, tokenizer)
    result["num_tokens"] = num_tokens

    # RM quality score
    rm_task = None
    if needs_rm:
        messages_for_rm = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response},
        ]
        formatted_text = tokenizer.apply_chat_template(
            messages_for_rm, tokenize=False, add_generation_prompt=False
        )
        rm_task = asyncio.create_task(
            _call_rm(
                rm_server_url,
                formatted_text,
                timeout_s=rm_timeout_s,
                max_retries=rm_max_retries,
                base_delay=rm_retry_base_delay_s,
            )
        )

    # Tagger diversity
    if needs_tagger:
        _, hist_bigrams, hist_trigrams = clean_history_structs(
            extra.get("tactic_history_counts"),
            extra.get("tactic_history_bigrams"),
            extra.get("tactic_history_trigrams"),
        )
        last_turn_raw = extra.get("tactic_last_turn_counts") or {}
        hist_counts = {k: v for k, v in last_turn_raw.items() if v is not None}

        sentences = sent_tokenize(response)
        _get_clients(tagger_server_url, tagger_timeout_s)
        tactic_info = _get_tactic_info(prompts_dir)
        all_tag_requests = _build_tag_requests(sentences, tactic_info, response)

        tag_tasks = []
        tag_meta = []
        for sent_idx, tactic_name, msgs in all_tag_requests:
            tag_tasks.append(asyncio.create_task(
                _tag_one(
                    tactic_name,
                    msgs,
                    temperature,
                    max_tokens,
                    semaphore,
                    max_retries=tagger_max_retries,
                    base_delay=tagger_retry_base_delay_s,
                )
            ))
            tag_meta.append((sent_idx, tactic_name))

    # Await RM
    if rm_task is not None:
        result["quality_raw"] = await rm_task

    # Await tagger
    if needs_tagger:
        tag_results = await asyncio.gather(*tag_tasks)
        sentence_tactics = [set() for _ in sentences]
        for (sent_idx, tactic_name), score in zip(tag_meta, tag_results):
            if score == 1:
                sentence_tactics[sent_idx].add(tactic_name)
        curr_counts = dict(Counter(t for s_set in sentence_tactics for t in s_set))

        # Compute tactic overlap and new-tactic rate vs previous turn
        curr_tactic_set = set(curr_counts.keys())
        prev_tactic_set = set(hist_counts.keys())
        if curr_tactic_set:
            reused = curr_tactic_set & prev_tactic_set
            result["tactic_overlap_pct"] = len(reused) / len(curr_tactic_set) * 100
            result["new_tactic_pct"] = len(curr_tactic_set - prev_tactic_set) / len(curr_tactic_set) * 100
        else:
            result["tactic_overlap_pct"] = 0.0
            result["new_tactic_pct"] = 0.0

        result["kl_raw"] = compute_kl_divergence(hist_counts, curr_counts, smoothing_alpha)
        result["entropy_raw"] = compute_within_turn_entropy(curr_counts)
        result["bigram_raw"] = (
            compute_bigram_surprisal(hist_bigrams, sentence_tactics, smoothing_alpha)
            if bigram_gamma > 0 else 0.0
        )
        result["trigram_raw"] = (
            compute_trigram_surprisal(hist_trigrams, sentence_tactics, smoothing_alpha)
            if trigram_gamma > 0 else 0.0
        )
        result["curr_counts"] = curr_counts
        result["hist_counts"] = hist_counts
        result["num_tactics"] = len(curr_counts)
        result["has_format_violation"] = bool(_TACTIC_LABEL_RE.search(response))

    return result


# ---------------------------------------------------------------------------
# Batch compute_score (VERL reward_manager.name=batch interface)
# ---------------------------------------------------------------------------

def compute_score(data_sources, solution_strs, ground_truths,
                   extra_infos=None, **kwargs):
    """
    VERL batch reward with configurable reward_mode.

    Args:
        data_sources: List of dataset identifiers (unused)
        solution_strs: List of generated response texts
        ground_truths: List (unused)
        extra_infos: List of dicts with prompt_text, tactic_history_counts, etc.
        **kwargs: Reward config from Hydra (must include reward_mode)

    Returns:
        List of dicts, each with "score" key + auxiliary metrics for logging
    """
    return asyncio.run(_compute_score_async(
        data_sources, solution_strs, ground_truths, extra_infos, **kwargs
    ))


async def _compute_score_async(data_sources, solution_strs, ground_truths,
                                extra_infos=None, **kwargs):
    reward_mode = str(kwargs.get("reward_mode", "diversity_only"))
    if reward_mode not in VALID_REWARD_MODES:
        raise ValueError(f"Unknown reward_mode={reward_mode!r}. Valid: {VALID_REWARD_MODES}")

    needs_rm = reward_mode != "diversity_only"
    needs_tagger = reward_mode in ("diversity_only", "quality_plus_diversity", "quality_times_diversity")

    rm_server_url = str(kwargs["rm_server_url"]).rstrip("/") if needs_rm else ""
    tagger_server_url = str(kwargs.get("tactic_tagger_server_url", ""))
    prompts_dir = str(kwargs.get("tactic_tagger_prompts_dir", ""))
    kl_gamma = float(kwargs.get("kl_gamma", 1.0))
    bigram_gamma = float(kwargs.get("bigram_gamma", 0.0))
    trigram_gamma = float(kwargs.get("trigram_gamma", 0.0))
    entropy_gamma = float(kwargs.get("entropy_gamma", 1.0))
    diversity_weight = float(kwargs.get("diversity_weight", 1.0))
    smoothing_alpha = float(kwargs.get("smoothing_alpha", 0.1))
    temperature = float(kwargs.get("tactic_tagger_temperature", 0.1))
    max_tokens = int(kwargs.get("tactic_tagger_max_tokens", 64))
    max_concurrent = max(1, int(kwargs.get("tactic_tagger_max_concurrent", 32)))
    rm_timeout_s = float(kwargs.get("rm_timeout_s", 20.0))
    rm_max_retries = max(1, int(kwargs.get("rm_max_retries", 6)))
    rm_retry_base_delay_s = float(kwargs.get("rm_retry_base_delay_s", 1.0))
    tagger_timeout_s = float(kwargs.get("tactic_tagger_timeout_s", 20.0))
    tagger_max_retries = max(1, int(kwargs.get("tactic_tagger_max_retries", 6)))
    tagger_retry_base_delay_s = float(kwargs.get("tactic_tagger_retry_base_delay_s", 1.0))
    sample_log_interval = int(kwargs.get("sample_log_interval", 384))
    sample_log_dir = str(kwargs.get("sample_log_dir", ""))

    n = len(solution_strs)
    if extra_infos is None:
        extra_infos = [None] * n

    # 1. Compute raw metrics for all samples concurrently
    semaphore = asyncio.Semaphore(max_concurrent)
    raw_list = await asyncio.gather(*[
        _compute_single_raw(
            solution_strs[i], extra_infos[i],
            reward_mode, rm_server_url, tagger_server_url, prompts_dir,
            bigram_gamma, trigram_gamma, smoothing_alpha,
            temperature, max_tokens, semaphore,
            rm_timeout_s, rm_max_retries, rm_retry_base_delay_s,
            tagger_timeout_s, tagger_max_retries, tagger_retry_base_delay_s,
        )
        for i in range(n)
    ])

    # 2. Per-group min-max normalization
    groups = {}
    for i, raw in enumerate(raw_list):
        groups.setdefault(raw["prompt_text"], []).append(i)

    q_normed = [0.0] * n
    kl_normed = [0.0] * n
    ent_normed = [0.0] * n

    for indices in groups.values():
        if needs_rm:
            q_vals = [raw_list[i]["quality_raw"] for i in indices]
            q_lo, q_hi = min(q_vals), max(q_vals)
            q_range = q_hi - q_lo
            for i in indices:
                q_normed[i] = (raw_list[i]["quality_raw"] - q_lo) / q_range if q_range > 0 else 0.5

        if needs_tagger:
            kl_vals = [raw_list[i]["kl_raw"] for i in indices]
            ent_vals = [raw_list[i]["entropy_raw"] for i in indices]
            kl_lo, kl_hi = min(kl_vals), max(kl_vals)
            ent_lo, ent_hi = min(ent_vals), max(ent_vals)
            kl_range = kl_hi - kl_lo
            ent_range = ent_hi - ent_lo
            if kl_range <= 1e-8:
                print(f"[reward_verl] WARNING: collapsed KL range ({kl_lo:.4f}-{kl_hi:.4f}) in group of {len(indices)} rollouts")
            if ent_range <= 1e-8:
                print(f"[reward_verl] WARNING: collapsed entropy range ({ent_lo:.4f}-{ent_hi:.4f}) in group of {len(indices)} rollouts")
            for i in indices:
                kl_normed[i] = (raw_list[i]["kl_raw"] - kl_lo) / kl_range if kl_range > 1e-8 else 0.5
                ent_normed[i] = (raw_list[i]["entropy_raw"] - ent_lo) / ent_range if ent_range > 1e-8 else 0.5

    # 3. Assemble final results per reward_mode
    results = []
    for i in range(n):
        raw = raw_list[i]
        length_scale = _compute_length_scale(raw["num_tokens"])

        if reward_mode == "diversity_only":
            diversity = kl_gamma * kl_normed[i] + entropy_gamma * ent_normed[i]
            total = diversity * length_scale
            if raw["has_format_violation"]:
                total += _FORMAT_VIOLATION_PENALTY

        elif reward_mode == "quality_plus_diversity":
            diversity_norm = kl_gamma * kl_normed[i] + entropy_gamma * ent_normed[i]
            total = q_normed[i] + diversity_weight * diversity_norm
            total *= length_scale
            if raw["has_format_violation"]:
                total += _FORMAT_VIOLATION_PENALTY

        elif reward_mode == "quality_times_diversity":
            diversity_norm = kl_gamma * kl_normed[i] + entropy_gamma * ent_normed[i]
            total = q_normed[i] * diversity_norm
            total *= length_scale
            if raw["has_format_violation"]:
                total += _FORMAT_VIOLATION_PENALTY

        elif reward_mode in ("quality_only", "quality_x_r1_zero_div"):
            total = q_normed[i] * length_scale

        result = {"score": total, "length_scale": round(length_scale, 4),
                  "num_tokens": raw["num_tokens"], "think_length": raw["think_length"]}

        if needs_rm:
            result["quality_raw"] = round(raw["quality_raw"], 4)
            result["quality_norm"] = round(q_normed[i], 4)

        if needs_tagger:
            diversity_norm_val = kl_gamma * kl_normed[i] + entropy_gamma * ent_normed[i]
            result["diversity_norm"] = round(diversity_norm_val, 4)
            result["kl_raw"] = round(raw["kl_raw"], 4)
            result["kl_norm"] = round(kl_normed[i], 4)
            result["entropy_raw"] = round(raw["entropy_raw"], 4)
            result["entropy_norm"] = round(ent_normed[i], 4)
            result["bigram_raw"] = round(raw["bigram_raw"], 4)
            result["trigram_raw"] = round(raw["trigram_raw"], 4)
            result["tactic_counts"] = json.dumps(raw["curr_counts"])
            result["num_tactics"] = raw["num_tactics"]
            result["tactic_overlap_pct"] = round(raw.get("tactic_overlap_pct", 0.0), 2)
            result["new_tactic_pct"] = round(raw.get("new_tactic_pct", 0.0), 2)

        _buffer_sample(result, raw["response"], raw["think_text"],
                       raw.get("curr_counts", {}), raw.get("hist_counts", {}),
                       sample_log_interval, sample_log_dir, raw["prompt_text"])
        results.append(result)

    return results