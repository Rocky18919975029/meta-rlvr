import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
)
from safetensors.torch import load_file
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Model

from meta_rlvr.bilevel import BilevelGRPO
from meta_rlvr.confidence import SequenceConfidenceModel
from meta_rlvr.config import (
    AdvantageConfig,
    FastOptimizerConfig,
    GRPOLossConfig,
    InnerLoopConfig,
    MetaLossConfig,
)
from meta_rlvr.functional import token_logprobs, trainable_parameter_state
from meta_rlvr.rollout import VLLMHybridRolloutEngine
from meta_rlvr.types import RolloutGroup


def tiny_qwen_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        attention_dropout=0.0,
        pad_token_id=0,
        eos_token_id=2,
        use_cache=False,
    )


def make_group(policy, rewards) -> RolloutGroup:
    correctness = rewards
    verifier_rewards = 2.0 * correctness - 1.0
    input_ids = torch.tensor(
        [
            [1, 4, 5, 6, 7, 2],
            [1, 4, 8, 9, 3, 2],
            [1, 4, 7, 5, 6, 2],
        ]
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    completion_mask = torch.tensor(
        [
            [False, True, True, True, True],
            [False, True, True, True, True],
            [False, True, True, True, True],
        ]
    )
    provisional = RolloutGroup(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        old_logprobs=torch.zeros((3, 5)),
        texts=("a", "b", "c"),
        verifier_rewards=verifier_rewards,
        correctness_labels=correctness,
    )
    old = token_logprobs(policy, provisional).detach()
    return RolloutGroup(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        old_logprobs=old,
        texts=("a", "b", "c"),
        verifier_rewards=verifier_rewards,
        correctness_labels=correctness,
    )


def test_tiny_qwen_peft_supports_differentiable_bilevel_update() -> None:
    torch.manual_seed(5)
    config = tiny_qwen_config()
    policy = get_peft_model(
        Qwen2ForCausalLM(config),
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ),
    )
    policy.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    policy.enable_input_require_grads()
    policy.train()
    confidence = SequenceConfidenceModel(
        Qwen2Model(config),
        hidden_size=config.hidden_size,
    )
    initial_fast = trainable_parameter_state(policy, required_name_substring="lora_")
    support = make_group(policy, torch.tensor([1.0, 0.0, 0.0]))
    query = make_group(policy, torch.tensor([0.0, 1.0, 0.0]))

    algorithm = BilevelGRPO(
        policy,
        confidence,
        InnerLoopConfig(
            num_iterations=2,
            optimizer=FastOptimizerConfig(name="sgd", learning_rate=0.05),
        ),
        MetaLossConfig(),
        AdvantageConfig(),
        GRPOLossConfig(),
        policy_micro_batch_size=1,
        confidence_micro_batch_size=1,
    )
    output = algorithm.outer_loss(support, query, initial_fast)
    output.loss.backward()

    assert confidence.score[-1].weight.grad is not None
    assert torch.isfinite(confidence.score[-1].weight.grad).all()


def test_task_fast_parameters_export_as_peft_adapter(tmp_path) -> None:
    policy = get_peft_model(
        Qwen2ForCausalLM(tiny_qwen_config()),
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ),
    )
    fast = trainable_parameter_state(policy, required_name_substring="lora_")
    with torch.no_grad():
        for index, value in enumerate(fast.values(), start=1):
            value.copy_(
                torch.arange(value.numel(), dtype=value.dtype).reshape_as(value) / index
            )
    engine = object.__new__(VLLMHybridRolloutEngine)
    engine.model = policy
    engine._save_adapter(tmp_path, fast)

    assert (tmp_path / "adapter_config.json").is_file()
    assert (tmp_path / "adapter_model.safetensors").is_file()
    exported = load_file(tmp_path / "adapter_model.safetensors")
    expected = get_peft_model_state_dict(
        policy,
        state_dict=fast,
        adapter_name="default",
    )
    assert exported.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(exported[name], value.cpu())
