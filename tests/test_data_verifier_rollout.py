import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from meta_rlvr.data import (
    ChatMessage,
    MathProblem,
    load_semantically_unique_dapo_problems,
    parse_dapo_row,
    rank_shard,
)
from meta_rlvr.functional import trainable_parameter_state
from meta_rlvr.rollout import TransformersRolloutEngine, VLLMHybridRolloutEngine
from meta_rlvr.train import _cache_rollout_group
from meta_rlvr.types import RolloutGroup
from meta_rlvr.verifier import DAPOMathVerifier


def test_parse_dapo_row_and_rank_shard_are_strict() -> None:
    row = {
        "prompt": [{"role": "user", "content": "Compute 6*7. Answer: ..."}],
        "reward_model": {"ground_truth": "42"},
        "extra_info": {"index": 17},
        "data_source": "math_dapo",
    }
    problem = parse_dapo_row(row)
    assert problem.uid == "17"
    assert problem.ground_truth == "42"
    assert rank_shard([problem], rank=0, world_size=1) == [problem]

    malformed = dict(row)
    malformed["extra_info"] = {}
    with pytest.raises(TypeError, match="index"):
        parse_dapo_row(malformed)


def test_evaluation_loader_collapses_replicated_prompts(tmp_path, monkeypatch) -> None:
    parquet = tmp_path / "aime24.parquet"
    parquet.touch()
    first = {
        "prompt": [{"role": "user", "content": "problem one"}],
        "reward_model": {"ground_truth": "1"},
        "extra_info": {"index": 0},
        "data_source": "aime24",
    }
    duplicate = {
        **first,
        "extra_info": {"index": 1},
    }
    second = {
        "prompt": [{"role": "user", "content": "problem two"}],
        "reward_model": {"ground_truth": "2"},
        "extra_info": {"index": 2},
        "data_source": "aime24",
    }
    fake_datasets = ModuleType("datasets")
    fake_datasets.load_dataset = lambda *args, **kwargs: [
        first,
        duplicate,
        second,
    ]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    problems = load_semantically_unique_dapo_problems(parquet)
    assert [problem.ground_truth for problem in problems] == ["1", "2"]


def test_official_dapo_verifier_wrapper_separates_rewards_and_correctness() -> None:
    verifier = DAPOMathVerifier()
    verification = verifier(
        ("Reasoning. \\boxed{42}", "Reasoning. \\boxed{41}"),
        "42",
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(verification.rewards, torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(verification.correctness, torch.tensor([1.0, 0.0]))
    assert verification.predictions == ("42", "41")


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 6

    def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
        assert conversation == [{"role": "user", "content": "problem"}]
        assert not tokenize
        assert add_generation_prompt
        return "prompt"

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        assert text == "prompt"
        assert not add_special_tokens
        if return_tensors is None:
            return {"input_ids": [1, 2]}
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, token_ids, skip_special_tokens):
        assert skip_special_tokens
        return "response"


class ToyGeneratingPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(7, 4)
        self.base = nn.Linear(4, 7, bias=False)
        self.adapter = nn.Parameter(torch.zeros(4, 7))
        self.embedding.weight.requires_grad_(False)
        self.base.weight.requires_grad_(False)
        self.config = SimpleNamespace(max_position_embeddings=16)

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=self.base(hidden) + hidden @ self.adapter)

    def generate(self, input_ids, attention_mask, **kwargs):
        completion = torch.tensor(
            [[3, 6]], dtype=input_ids.dtype, device=input_ids.device
        ).expand(input_ids.shape[0], -1)
        return torch.cat((input_ids, completion), dim=1)


def test_transformers_rollout_builds_aligned_masks_and_old_logprobs() -> None:
    policy = ToyGeneratingPolicy()
    fast = trainable_parameter_state(policy)
    engine = TransformersRolloutEngine(
        policy,
        ToyTokenizer(),
        group_size=3,
        max_new_tokens=2,
        temperature=1.0,
        top_p=0.7,
        generation_micro_batch_size=2,
    )
    problem = MathProblem(
        uid="1",
        messages=(ChatMessage(role="user", content="problem"),),
        ground_truth="42",
        data_source="unit_test",
    )
    group = engine.generate(problem, fast)

    assert group.input_ids.shape == (3, 4)
    assert group.completion_mask.shape == (3, 3)
    assert torch.all(group.completion_mask[:, 1:])
    assert torch.all(~group.completion_mask[:, :1])
    assert group.old_logprobs.shape == group.completion_mask.shape
    assert group.texts == ("response", "response", "response")


def test_rollout_group_device_transfer_preserves_all_fields_and_aliases() -> None:
    input_ids = torch.tensor([[1, 2, 3], [1, 3, 4]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    completion_mask = torch.tensor([[False, True], [False, True]], dtype=torch.bool)
    old_logprobs = torch.tensor([[0.0, -0.2], [0.0, -0.3]])
    group = RolloutGroup(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        old_logprobs=old_logprobs,
        texts=("first", "second"),
        verifier_rewards=torch.tensor([1.0, -1.0]),
        correctness_labels=torch.tensor([1.0, 0.0]),
        reference_logprobs=old_logprobs,
        rollout_logprobs=torch.tensor([[0.0, -0.1], [0.0, -0.4]]),
    )

    moved = group.to("cpu")

    assert moved.device.type == "cpu"
    assert moved.reference_logprobs is moved.old_logprobs
    assert moved.texts == group.texts
    for field in (
        "input_ids",
        "attention_mask",
        "completion_mask",
        "old_logprobs",
        "verifier_rewards",
        "correctness_labels",
        "rollout_logprobs",
    ):
        torch.testing.assert_close(getattr(moved, field), getattr(group, field))

    cached = _cache_rollout_group(group)
    assert cached.device.type == "cpu"
    assert cached.texts == ("", "")
    assert cached.rollout_logprobs is None
    assert cached.reference_logprobs is cached.old_logprobs
    torch.testing.assert_close(cached.correctness_labels, group.correctness_labels)


def test_vllm_rollout_preserves_returned_token_ids_and_choice_order() -> None:
    policy = ToyGeneratingPolicy()
    engine = object.__new__(VLLMHybridRolloutEngine)
    TransformersRolloutEngine.__init__(
        engine,
        policy,
        ToyTokenizer(),
        group_size=2,
        max_new_tokens=3,
        temperature=1.0,
        top_p=0.7,
        generation_micro_batch_size=2,
    )
    completion_ids, completion_logprobs = engine._completion_data(
        {
            "choices": [
                {
                    "index": 1,
                    "token_ids": [5, 6],
                    "logprobs": {"token_logprobs": [-0.5, -0.6]},
                },
                {
                    "index": 0,
                    "token_ids": [3, 4, 6],
                    "logprobs": {"token_logprobs": [-0.3, -0.4, -0.6]},
                },
            ]
        }
    )
    assert completion_ids == [[3, 4, 6], [5, 6]]
    assert completion_logprobs == [[-0.3, -0.4, -0.6], [-0.5, -0.6]]

    sequences = engine._sequences_from_completion_ids([1, 2], completion_ids)
    assert sequences.tolist() == [[1, 2, 3, 4, 6], [1, 2, 5, 6, 0]]
    group = engine._build_group(sequences, prompt_length=2)
    assert group.completion_mask.tolist() == [
        [False, True, True, True],
        [False, True, True, False],
    ]
    rollout_logprobs = engine._aligned_rollout_logprobs(
        group,
        completion_logprobs,
    )
    torch.testing.assert_close(
        rollout_logprobs,
        torch.tensor(
            [
                [0.0, -0.3, -0.4, -0.6],
                [0.0, -0.5, -0.6, 0.0],
            ]
        ),
    )


def test_vllm_rollout_uses_staged_hybrid_lifecycle(tmp_path) -> None:
    policy = ToyGeneratingPolicy()
    fast = trainable_parameter_state(policy)
    engine = object.__new__(VLLMHybridRolloutEngine)
    TransformersRolloutEngine.__init__(
        engine,
        policy,
        ToyTokenizer(),
        group_size=2,
        max_new_tokens=3,
        temperature=1.0,
        top_p=0.7,
        top_k=0,
        generation_micro_batch_size=2,
    )
    engine.adapter_root = tmp_path
    engine.request_timeout = 30.0
    engine.control_timeout = 5.0
    engine._save_adapter = lambda adapter_dir, fast_parameters: None
    engine._unadapted_logprobs = lambda group, **kwargs: torch.zeros_like(
        group.old_logprobs
    )

    calls = []
    sleeping = True
    loaded_adapter = None

    def request_json(method, path, payload=None, *, timeout=None):
        nonlocal sleeping, loaded_adapter
        calls.append((method, path, payload, timeout))
        if path == "/wake_up?tags=weights":
            return None
        if path == "/wake_up?tags=kv_cache":
            sleeping = False
            return None
        if path == "/is_sleeping":
            return {"is_sleeping": sleeping}
        if path == "/v1/models":
            model_ids = ["meta-rlvr-base"]
            if loaded_adapter is not None:
                model_ids.append(loaded_adapter)
            return {"data": [{"id": model_id} for model_id in model_ids]}
        if path == "/v1/completions":
            assert payload["add_special_tokens"] is False
            assert payload["logprobs"] == 0
            assert payload["top_k"] == 0
            return {
                "choices": [
                    {
                        "index": 0,
                        "token_ids": [3, 6],
                        "logprobs": {"token_logprobs": [-0.3, -0.6]},
                    },
                    {
                        "index": 1,
                        "token_ids": [5, 6],
                        "logprobs": {"token_logprobs": [-0.5, -0.6]},
                    },
                ]
            }
        if path == "/sleep?level=1":
            sleeping = True
            return None
        raise AssertionError(f"Unexpected request: {method} {path}")

    def request_text(method, path, payload=None, *, timeout=None):
        nonlocal loaded_adapter
        calls.append((method, path, payload, timeout))
        if path == "/v1/load_lora_adapter":
            loaded_adapter = payload["lora_name"]
            return f"Success: LoRA adapter '{loaded_adapter}' added successfully."
        if path == "/v1/unload_lora_adapter":
            adapter_name = payload["lora_name"]
            loaded_adapter = None
            return f"Success: LoRA adapter '{adapter_name}' removed successfully."
        raise AssertionError(f"Unexpected request: {method} {path}")

    engine._request_json = request_json
    engine._request_text = request_text
    problem = MathProblem(
        uid="1",
        messages=(ChatMessage(role="user", content="problem"),),
        ground_truth="42",
        data_source="unit_test",
    )
    group = engine.generate(problem, fast)

    paths = [path for _, path, _, _ in calls]
    assert paths[:6] == [
        "/wake_up?tags=weights",
        "/v1/load_lora_adapter",
        "/v1/models",
        "/wake_up?tags=kv_cache",
        "/is_sleeping",
        "/v1/completions",
    ]
    assert paths[-4:] == [
        "/v1/unload_lora_adapter",
        "/v1/models",
        "/sleep?level=1",
        "/is_sleeping",
    ]
    assert group.rollout_logprobs is not None
    assert sleeping
    assert loaded_adapter is None


@pytest.mark.parametrize(
    ("distinct_adapters", "expected_adapter_count"),
    ((False, 1), (True, 2)),
)
def test_vllm_problem_batch_uses_one_hybrid_transaction(
    tmp_path,
    distinct_adapters,
    expected_adapter_count,
) -> None:
    policy = ToyGeneratingPolicy()
    first_fast = trainable_parameter_state(policy)
    second_fast = (
        {
            name: value.detach().clone().requires_grad_(True)
            for name, value in first_fast.items()
        }
        if distinct_adapters
        else first_fast
    )
    engine = object.__new__(VLLMHybridRolloutEngine)
    TransformersRolloutEngine.__init__(
        engine,
        policy,
        ToyTokenizer(),
        group_size=2,
        max_new_tokens=3,
        temperature=1.0,
        top_p=0.7,
        top_k=0,
        generation_micro_batch_size=2,
    )
    engine.adapter_root = tmp_path
    engine.request_timeout = 30.0
    engine.control_timeout = 5.0
    engine._save_adapter = lambda adapter_dir, fast_parameters: None
    engine._unadapted_logprobs = lambda group, **kwargs: torch.zeros_like(
        group.old_logprobs
    )

    calls = []
    sleeping = True
    loaded_adapters = set()

    def request_json(method, path, payload=None, *, timeout=None):
        nonlocal sleeping
        calls.append((method, path, payload, timeout))
        if path == "/wake_up?tags=weights":
            return None
        if path == "/wake_up?tags=kv_cache":
            sleeping = False
            return None
        if path == "/is_sleeping":
            return {"is_sleeping": sleeping}
        if path == "/v1/models":
            model_ids = ["meta-rlvr-base", *sorted(loaded_adapters)]
            return {"data": [{"id": model_id} for model_id in model_ids]}
        if path == "/v1/completions":
            assert payload["model"] in loaded_adapters
            return {
                "choices": [
                    {
                        "index": 0,
                        "token_ids": [3, 6],
                        "logprobs": {"token_logprobs": [-0.3, -0.6]},
                    },
                    {
                        "index": 1,
                        "token_ids": [5, 6],
                        "logprobs": {"token_logprobs": [-0.5, -0.6]},
                    },
                ]
            }
        if path == "/sleep?level=1":
            sleeping = True
            return None
        raise AssertionError(f"Unexpected request: {method} {path}")

    def request_text(method, path, payload=None, *, timeout=None):
        calls.append((method, path, payload, timeout))
        adapter_name = payload["lora_name"]
        if path == "/v1/load_lora_adapter":
            loaded_adapters.add(adapter_name)
            return f"Success: LoRA adapter '{adapter_name}' added successfully."
        if path == "/v1/unload_lora_adapter":
            loaded_adapters.remove(adapter_name)
            return f"Success: LoRA adapter '{adapter_name}' removed successfully."
        raise AssertionError(f"Unexpected request: {method} {path}")

    engine._request_json = request_json
    engine._request_text = request_text
    problems = [
        MathProblem(
            uid=str(index),
            messages=(ChatMessage(role="user", content="problem"),),
            ground_truth="42",
            data_source="unit_test",
        )
        for index in range(2)
    ]
    groups = engine.generate_batch(
        problems,
        [first_fast, second_fast],
    )

    paths = [path for _, path, _, _ in calls]
    assert paths.count("/wake_up?tags=weights") == 1
    assert paths.count("/wake_up?tags=kv_cache") == 1
    assert paths.count("/v1/completions") == 2
    assert paths.count("/v1/load_lora_adapter") == expected_adapter_count
    assert paths.count("/v1/unload_lora_adapter") == expected_adapter_count
    assert paths.count("/sleep?level=1") == 1
    assert len(groups) == 2
    assert all(group.rollout_logprobs is not None for group in groups)
    assert sleeping
    assert not loaded_adapters


def test_vllm_http_500_fails_immediately() -> None:
    class FailingHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"deliberate failure")

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), FailingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    engine = object.__new__(VLLMHybridRolloutEngine)
    engine.base_url = f"http://127.0.0.1:{server.server_port}"
    engine.control_timeout = 5.0
    try:
        with pytest.raises(RuntimeError, match="HTTP 500: deliberate failure"):
            engine._request_json("GET", "/health")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_vllm_plain_text_response_is_not_parsed_as_json() -> None:
    response_body = b"Success: LoRA adapter 'test' added successfully."

    class PlainTextHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), PlainTextHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    engine = object.__new__(VLLMHybridRolloutEngine)
    engine.base_url = f"http://127.0.0.1:{server.server_port}"
    engine.control_timeout = 5.0
    try:
        assert engine._request_text("POST", "/v1/load_lora_adapter") == (
            response_body.decode("utf-8")
        )
        with pytest.raises(RuntimeError, match="returned non-JSON content"):
            engine._request_json("POST", "/v1/load_lora_adapter")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not thread.is_alive()
