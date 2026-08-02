from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .confidence import SequenceConfidenceModel
from .functional import ParameterDict, trainable_parameter_state


@dataclass(frozen=True)
class PolicyBundle:
    model: nn.Module
    tokenizer: Any
    initial_fast_parameters: ParameterDict


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name!r}")
    return mapping[name]


def load_policy_with_lora(
    model_name_or_path: str,
    *,
    lora_rank: int,
    lora_alpha: int,
    target_modules: tuple[str, ...],
    dtype: str = "bfloat16",
    gradient_checkpointing: bool = True,
    trust_remote_code: bool = False,
    model_kwargs: dict[str, Any] | None = None,
) -> PolicyBundle:
    if not model_name_or_path:
        raise ValueError("model_name_or_path cannot be empty.")
    if lora_rank <= 0 or lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive.")
    if not target_modules:
        raise ValueError("target_modules cannot be empty.")

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = dict(model_kwargs or {})
    forbidden = {"torch_dtype", "trust_remote_code"}.intersection(kwargs)
    if forbidden:
        raise ValueError(f"model_kwargs contains reserved keys: {sorted(forbidden)}")
    policy = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=_torch_dtype(dtype),
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    policy.config.use_cache = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=list(target_modules),
        bias="none",
        init_lora_weights=True,
    )
    policy = get_peft_model(policy, lora_config)
    if gradient_checkpointing:
        policy.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if not hasattr(policy, "enable_input_require_grads"):
            raise TypeError(
                "Policy model must support enable_input_require_grads when gradient "
                "checkpointing is enabled."
            )
        policy.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Qwen tokenizer must define pad_token_id and eos_token_id.")

    initial_fast_parameters = trainable_parameter_state(
        policy,
        required_name_substring="lora_",
    )
    # Qwen enables gradient checkpointing only in training mode. LoRA dropout is
    # fixed to zero, and the rollout engine switches to eval mode for sampling.
    policy.train()
    return PolicyBundle(
        model=policy,
        tokenizer=tokenizer,
        initial_fast_parameters=initial_fast_parameters,
    )


def load_confidence_model(
    model_name_or_path: str,
    *,
    dtype: str = "bfloat16",
    zero_init_output: bool = True,
    gradient_checkpointing: bool = True,
    trust_remote_code: bool = False,
    model_kwargs: dict[str, Any] | None = None,
) -> SequenceConfidenceModel:
    kwargs = dict(model_kwargs or {})
    forbidden = {"torch_dtype", "trust_remote_code"}.intersection(kwargs)
    if forbidden:
        raise ValueError(f"model_kwargs contains reserved keys: {sorted(forbidden)}")
    model = SequenceConfidenceModel.from_pretrained(
        model_name_or_path,
        zero_init_output=zero_init_output,
        torch_dtype=_torch_dtype(dtype),
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    if gradient_checkpointing:
        if not hasattr(model.backbone, "gradient_checkpointing_enable"):
            raise TypeError("Confidence backbone does not support gradient checkpointing.")
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model
