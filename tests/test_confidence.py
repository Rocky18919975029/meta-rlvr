from types import SimpleNamespace

import pytest
import torch
from torch import nn

from meta_rlvr.confidence import SequenceConfidenceModel


class ToyBackbone(nn.Module):
    def __init__(self, vocabulary_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            initializer_range=0.02,
        )

    def forward(self, input_ids, attention_mask, return_dict):
        assert return_dict
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_confidence_model_pools_last_non_padding_token() -> None:
    torch.manual_seed(0)
    backbone = ToyBackbone(vocabulary_size=10, hidden_size=4)
    model = SequenceConfidenceModel(backbone, hidden_size=4)
    input_ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
    attention_mask = torch.tensor(
        [[True, True, False], [True, False, False]]
    )
    logits = model(input_ids, attention_mask)
    assert logits.shape == (2,)


def test_head_uses_qwen_random_initialization_and_zero_biases() -> None:
    torch.manual_seed(7)
    model = SequenceConfidenceModel(
        ToyBackbone(vocabulary_size=10, hidden_size=4),
        hidden_size=4,
    )
    for layer in (model.score[0], model.score[2]):
        assert torch.any(layer.weight != 0)
        torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))


def test_confidence_model_rejects_non_boolean_mask() -> None:
    model = SequenceConfidenceModel(
        ToyBackbone(vocabulary_size=10, hidden_size=4),
        hidden_size=4,
    )
    with pytest.raises(TypeError, match="torch.bool"):
        model(torch.tensor([[1, 2]]), torch.tensor([[1, 1]]))
