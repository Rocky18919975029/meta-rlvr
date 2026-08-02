from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class SequenceConfidenceModel(nn.Module):
    """Qwen-style sequence reward model with a two-layer scalar head."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        self.backbone = backbone
        if not hasattr(backbone, "config"):
            raise TypeError("Confidence backbone must expose a config object.")
        self.config = backbone.config
        self._no_split_modules = getattr(backbone, "_no_split_modules", None)
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        initializer_range = getattr(self.config, "initializer_range", None)
        if (
            not isinstance(initializer_range, (int, float))
            or initializer_range <= 0
        ):
            raise ValueError(
                "Confidence backbone config must define a positive "
                "initializer_range."
            )
        for module in self.score.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=float(initializer_range),
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **model_kwargs: Any,
    ) -> "SequenceConfidenceModel":
        if not model_name_or_path:
            raise ValueError("model_name_or_path cannot be empty.")
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
        hidden_size = getattr(backbone.config, "hidden_size", None)
        if not isinstance(hidden_size, int):
            raise ValueError("Backbone config must define an integer hidden_size.")
        model = cls(
            backbone,
            hidden_size,
        )
        try:
            reference_parameter = next(backbone.parameters())
        except StopIteration as error:
            raise ValueError("Confidence backbone has no parameters.") from error
        model.score.to(
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        return model

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids.")
        if attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must be torch.bool.")
        if torch.any(attention_mask.sum(dim=1) == 0):
            raise ValueError("Every confidence input must contain at least one token.")

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            raise TypeError("Confidence backbone must return last_hidden_state.")
        if hidden_states.ndim != 3 or hidden_states.shape[:2] != input_ids.shape:
            raise ValueError("Unexpected confidence backbone hidden-state shape.")

        positions = torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).unsqueeze(0)
        last_indices = positions.masked_fill(~attention_mask, -1).max(dim=1).values
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        pooled = hidden_states[batch_indices, last_indices]
        logits = self.score(pooled).squeeze(-1)
        if logits.shape != (input_ids.shape[0],):
            raise ValueError("Confidence head must return one scalar per sequence.")
        return logits
