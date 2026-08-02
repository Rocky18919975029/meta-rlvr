import sys
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


def test_evaluation_loader_collapses_replicated_prompts(
    tmp_path, monkeypatch
) -> None:
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
    torch.testing.assert_close(
        verification.correctness, torch.tensor([1.0, 0.0])
    )
    assert verification.predictions == ("42", "41")


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 6

    def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
        assert conversation == [{"role": "user", "content": "problem"}]
        assert not tokenize
        assert add_generation_prompt
        return "prompt"

    def __call__(self, text, return_tensors, add_special_tokens):
        assert text == "prompt"
        assert return_tensors == "pt"
        assert not add_special_tokens
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
    completion_ids = engine._completion_ids(
        {
            "choices": [
                {"index": 1, "token_ids": [5, 6]},
                {"index": 0, "token_ids": [3, 4, 6]},
            ]
        }
    )
    assert completion_ids == [[3, 4, 6], [5, 6]]

    sequences = engine._sequences_from_completion_ids([1, 2], completion_ids)
    assert sequences.tolist() == [[1, 2, 3, 4, 6], [1, 2, 5, 6, 0]]
    group = engine._build_group(sequences, prompt_length=2)
    assert group.completion_mask.tolist() == [
        [False, True, True, True],
        [False, True, True, False],
    ]
