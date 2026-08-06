from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from .data import MathProblem, load_semantically_unique_dapo_problems
from .verifier import DAPOMathVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc base-policy K=32 evaluation matching a completed "
            "Meta-RLVR validation run."
        )
    )
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vllm-base-urls", required=True)
    parser.add_argument("--reference-step", type=int, required=True)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    return parser.parse_args()


def _request_json(
    method: str,
    url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}{path}",
        data=data,
        headers={} if data is None else {"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"vLLM {method} {url}{path} returned HTTP {error.code}: {detail}"
        ) from error
    if not body:
        return None
    return json.loads(body)


def _generation_seed(
    *,
    base_seed: int,
    step: int,
    phase: str,
    rank: int,
    problem_uid: str,
) -> int:
    payload = f"{base_seed}|{step}|{phase}|{rank}|{problem_uid}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _load_adapted_records(
    reference_run_dir: Path,
    *,
    step: int,
) -> list[dict[str, Any]]:
    records = []
    for path in sorted(reference_run_dir.glob("rollouts-rank-*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("step") == step and record.get("phase") == (
                    "validation_adapted"
                ):
                    records.append(record)
    if not records:
        raise ValueError(
            f"No validation_adapted records for step {step} in "
            f"{reference_run_dir}."
        )
    return records


def _summarize(records: list[dict[str, Any]]) -> dict[str, float | int]:
    by_problem: dict[str, list[float]] = {}
    for record in records:
        by_problem.setdefault(record["problem_uid"], []).append(
            float(record["correct"])
        )
    correct = sum(sum(values) for values in by_problem.values())
    responses = sum(len(values) for values in by_problem.values())
    passed = sum(any(value == 1 for value in values) for values in by_problem.values())
    return {
        "accuracy": correct / responses,
        "pass_at_group": passed / len(by_problem),
        "num_unique_problems": len(by_problem),
        "responses": responses,
        "group_size": responses // len(by_problem),
    }


def _wake(url: str, timeout: float) -> None:
    _request_json("POST", url, "/wake_up?tags=weights", timeout=timeout)
    _request_json("POST", url, "/wake_up?tags=kv_cache", timeout=timeout)
    state = _request_json("GET", url, "/is_sleeping", timeout=timeout)
    if state != {"is_sleeping": False}:
        raise RuntimeError(f"vLLM did not wake: {url}: {state!r}")


def _sleep(url: str, timeout: float) -> None:
    _request_json("POST", url, "/sleep?level=1", timeout=timeout)
    state = _request_json("GET", url, "/is_sleeping", timeout=timeout)
    if state != {"is_sleeping": True}:
        raise RuntimeError(f"vLLM did not sleep: {url}: {state!r}")


def _generate_problem(
    *,
    problem: MathProblem,
    url: str,
    tokenizer,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    timeout: float,
) -> tuple[MathProblem, list[list[int]]]:
    prompt = tokenizer.apply_chat_template(
        problem.conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response = _request_json(
        "POST",
        url,
        "/v1/completions",
        {
            "model": "meta-rlvr-base",
            "prompt": prompt_ids,
            "n": group_size,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "stream": False,
            "add_special_tokens": False,
            "logprobs": 0,
            "return_token_ids": True,
        },
        timeout=timeout,
    )
    choices = sorted(response["choices"], key=lambda choice: choice["index"])
    if len(choices) != group_size:
        raise ValueError(
            f"vLLM returned {len(choices)} responses for {problem.uid}; "
            f"expected {group_size}."
        )
    return problem, [choice["token_ids"] for choice in choices]


def main() -> None:
    args = parse_args()
    run_config_path = args.reference_run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    group_size = int(run_config["validation_query_group_size"])
    validation_limit = int(run_config["validation_max_problems"])
    reference_world_size = int(run_config["distributed_world_size"])
    base_seed = int(run_config["seed"])

    problems = load_semantically_unique_dapo_problems(
        run_config["validation_parquet"]
    )[:validation_limit]
    adapted_records = _load_adapted_records(
        args.reference_run_dir,
        step=args.reference_step,
    )
    adapted_summary = _summarize(adapted_records)
    if adapted_summary["num_unique_problems"] != len(problems):
        raise ValueError("Reference adapted records do not match validation problems.")
    if adapted_summary["group_size"] != group_size:
        raise ValueError("Reference adapted group size does not match run_config.json.")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    verifier = DAPOMathVerifier(strict_box_verify=True)
    urls = [url.strip() for url in args.vllm_base_urls.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rollout_path = args.output_dir / "base-query-k32.jsonl"
    problem_metric_path = args.output_dir / "problem-metrics.jsonl"
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        list(executor.map(lambda url: _wake(url, args.request_timeout), urls))
    try:
        futures = {}
        with ThreadPoolExecutor(max_workers=len(problems)) as executor:
            for problem_index, problem in enumerate(problems):
                original_rank = problem_index % reference_world_size
                seed = _generation_seed(
                    base_seed=base_seed,
                    step=args.reference_step,
                    phase="validation_adapted",
                    rank=original_rank,
                    problem_uid=problem.uid,
                )
                future = executor.submit(
                    _generate_problem,
                    problem=problem,
                    url=urls[problem_index % len(urls)],
                    tokenizer=tokenizer,
                    group_size=group_size,
                    max_new_tokens=int(run_config["max_new_tokens"]),
                    temperature=float(run_config["temperature"]),
                    top_p=float(run_config["top_p"]),
                    top_k=int(run_config["top_k"]),
                    seed=seed,
                    timeout=args.request_timeout,
                )
                futures[future] = (problem_index, seed)

            generated: dict[int, tuple[MathProblem, list[list[int]], int]] = {}
            progress = tqdm(
                total=len(problems) * group_size,
                desc="post-hoc base query K=32",
                unit="response",
            )
            for future in as_completed(futures):
                problem_index, seed = futures[future]
                problem, completion_ids = future.result()
                generated[problem_index] = (problem, completion_ids, seed)
                progress.update(group_size)
            progress.close()
    finally:
        with ThreadPoolExecutor(max_workers=len(urls)) as executor:
            list(executor.map(lambda url: _sleep(url, args.request_timeout), urls))

    base_records = []
    adapted_by_problem: dict[str, list[dict[str, Any]]] = {}
    for record in adapted_records:
        adapted_by_problem.setdefault(record["problem_uid"], []).append(record)

    with rollout_path.open("w", encoding="utf-8") as rollout_stream, \
        problem_metric_path.open("w", encoding="utf-8") as metric_stream:
        for problem_index in range(len(problems)):
            problem, completion_ids, seed = generated[problem_index]
            responses = tuple(
                tokenizer.decode(token_ids, skip_special_tokens=True)
                for token_ids in completion_ids
            )
            verification = verifier(
                responses,
                problem.ground_truth,
                device=torch.device("cpu"),
            )
            base_correct = verification.correctness.tolist()
            adapted_correct = [
                float(record["correct"])
                for record in adapted_by_problem[problem.uid]
            ]
            metric_stream.write(
                json.dumps(
                    {
                        "problem_uid": problem.uid,
                        "base_accuracy": sum(base_correct) / group_size,
                        "adapted_accuracy": sum(adapted_correct) / group_size,
                        "base_pass_at_32": float(any(base_correct)),
                        "adapted_pass_at_32": float(any(adapted_correct)),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            for response_index, (response, token_ids) in enumerate(
                zip(responses, completion_ids, strict=True)
            ):
                record = {
                    "step": args.reference_step,
                    "phase": "validation_base_query_posthoc",
                    "problem_uid": problem.uid,
                    "data_source": problem.data_source,
                    "ground_truth": problem.ground_truth,
                    "response_index": response_index,
                    "seed": seed,
                    "completion_tokens": len(token_ids),
                    "ended_with_eos": token_ids[-1] == tokenizer.eos_token_id,
                    "hit_max_new_tokens": (
                        len(token_ids) == int(run_config["max_new_tokens"])
                        and token_ids[-1] != tokenizer.eos_token_id
                    ),
                    "prediction": verification.predictions[response_index],
                    "reward": verification.rewards[response_index].item(),
                    "correct": verification.correctness[response_index].item(),
                    "response": response,
                }
                base_records.append(record)
                rollout_stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    base_summary = _summarize(base_records)
    summary = {
        "event": "posthoc_base_vs_adapted_k32",
        "reference_run_dir": str(args.reference_run_dir),
        "reference_step": args.reference_step,
        "seed_phase": "validation_adapted",
        "base_query": base_summary,
        "adapted_query": adapted_summary,
        "accuracy_delta": (
            adapted_summary["accuracy"] - base_summary["accuracy"]
        ),
        "pass_at_32_delta": (
            adapted_summary["pass_at_group"] - base_summary["pass_at_group"]
        ),
        "seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
