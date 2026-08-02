from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor, nn

from .data import MathProblem
from .functional import materialized_fast_parameters, token_logprobs
from .types import RolloutGroup


class TransformersRolloutEngine:
    """Correctness-first rollout backend for task-specific PEFT adapters."""

    _RESERVED_GENERATION_KEYS = {
        "input_ids",
        "attention_mask",
        "num_return_sequences",
        "return_dict_in_generate",
        "synced_gpus",
        "use_cache",
    }

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        *,
        group_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        generation_micro_batch_size: int,
        logprob_micro_batch_size: int = 1,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if group_size < 2:
            raise ValueError("group_size must be at least two.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        if generation_micro_batch_size <= 0:
            raise ValueError("generation_micro_batch_size must be positive.")
        if logprob_micro_batch_size <= 0:
            raise ValueError("logprob_micro_batch_size must be positive.")
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id and eos_token_id.")
        if not hasattr(tokenizer, "apply_chat_template"):
            raise TypeError("Tokenizer must implement apply_chat_template.")

        self.model = model
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generation_micro_batch_size = generation_micro_batch_size
        self.logprob_micro_batch_size = logprob_micro_batch_size
        self.generation_kwargs = dict(generation_kwargs or {})
        overlap = self._RESERVED_GENERATION_KEYS.intersection(self.generation_kwargs)
        if overlap:
            raise ValueError(
                f"generation_kwargs contains reserved keys: {sorted(overlap)}"
            )

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration as error:
            raise ValueError("Policy model has no parameters.") from error

    def generate(
        self,
        problem: MathProblem,
        fast_parameters: Mapping[str, Tensor],
        *,
        show_progress: bool = False,
        progress_description: str = "rollout",
    ) -> RolloutGroup:
        prompt = self.tokenizer.apply_chat_template(
            problem.conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Chat template must return a non-empty string.")
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        if not {"input_ids", "attention_mask"}.issubset(encoded):
            raise ValueError("Tokenizer output must contain input_ids/attention_mask.")
        prompt_ids = encoded["input_ids"]
        prompt_mask = encoded["attention_mask"]
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError("A single problem must tokenize to shape [1, prompt_length].")
        if prompt_mask.shape != prompt_ids.shape:
            raise ValueError("Tokenized attention_mask must match input_ids.")
        if not torch.all(prompt_mask == 1):
            raise ValueError("A single unpadded prompt must have an all-one attention mask.")
        prompt_ids = prompt_ids.to(self.device)
        prompt_mask = prompt_mask.to(self.device)
        prompt_length = prompt_ids.shape[1]
        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        if not isinstance(max_positions, int):
            raise ValueError("Policy config must define integer max_position_embeddings.")
        if prompt_length + self.max_new_tokens > max_positions:
            raise ValueError(
                f"prompt_length ({prompt_length}) + max_new_tokens "
                f"({self.max_new_tokens}) exceeds max_position_embeddings "
                f"({max_positions})."
            )

        generated_batches: list[Tensor] = []
        was_training = self.model.training
        self.model.eval()
        try:
            with materialized_fast_parameters(self.model, fast_parameters):
                with torch.no_grad():
                    starts = range(
                        0,
                        self.group_size,
                        self.generation_micro_batch_size,
                    )
                    progress_bar = None
                    if show_progress:
                        from tqdm.auto import tqdm

                        progress_bar = tqdm(
                            total=self.group_size,
                            desc=f"{progress_description}: generate",
                            unit="response",
                            leave=True,
                        )
                    for start in starts:
                        batch_size = min(
                            self.generation_micro_batch_size,
                            self.group_size - start,
                        )
                        batch_ids = prompt_ids.expand(batch_size, -1)
                        batch_mask = prompt_mask.expand(batch_size, -1)
                        sequences = self.model.generate(
                            input_ids=batch_ids,
                            attention_mask=batch_mask,
                            do_sample=True,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            max_new_tokens=self.max_new_tokens,
                            num_return_sequences=1,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True,
                            # The frozen policy is replicated, not FSDP-sharded;
                            # ranks therefore decode independently without waiting
                            # for the longest completion on another rank.
                            synced_gpus=False,
                            return_dict_in_generate=False,
                            **self.generation_kwargs,
                        )
                        if not isinstance(sequences, Tensor) or sequences.ndim != 2:
                            raise TypeError("model.generate must return a rank-two tensor.")
                        if sequences.shape[0] != batch_size:
                            raise ValueError("model.generate returned an unexpected batch size.")
                        generated_batches.append(sequences)
                        if progress_bar is not None:
                            progress_bar.update(batch_size)
                    if progress_bar is not None:
                        progress_bar.close()

                sequences = self._right_pad_and_concatenate(generated_batches)
                group = self._build_group(sequences, prompt_length)
                with torch.no_grad():
                    old_logprobs = self._unadapted_logprobs(
                        group,
                        show_progress=show_progress,
                        progress_description=(
                            f"{progress_description}: old log-probabilities"
                        ),
                    )
                group = replace(group, old_logprobs=old_logprobs)
        finally:
            self.model.train(was_training)
        return group

    def add_reference_logprobs(
        self,
        group: RolloutGroup,
        reference_fast_parameters: Mapping[str, Tensor],
        *,
        show_progress: bool = False,
        progress_description: str = "reference log-probabilities",
    ) -> RolloutGroup:
        was_training = self.model.training
        self.model.eval()
        try:
            with materialized_fast_parameters(self.model, reference_fast_parameters):
                with torch.no_grad():
                    reference = self._unadapted_logprobs(
                        group,
                        show_progress=show_progress,
                        progress_description=progress_description,
                    )
        finally:
            self.model.train(was_training)
        return group.with_reference_logprobs(reference)

    def _unadapted_logprobs(
        self,
        group: RolloutGroup,
        *,
        show_progress: bool,
        progress_description: str,
    ) -> Tensor:
        starts = range(0, group.group_size, self.logprob_micro_batch_size)
        if show_progress:
            from tqdm.auto import tqdm

            starts = tqdm(
                starts,
                total=(group.group_size + self.logprob_micro_batch_size - 1)
                // self.logprob_micro_batch_size,
                desc=progress_description,
                unit="microbatch",
                leave=True,
            )
        chunks = []
        for start in starts:
            chunks.append(
                token_logprobs(
                    self.model,
                    group,
                    row_start=start,
                    row_end=min(
                        start + self.logprob_micro_batch_size,
                        group.group_size,
                    ),
                ).detach()
            )
        return torch.cat(chunks, dim=0)

    def _right_pad_and_concatenate(self, batches: list[Tensor]) -> Tensor:
        if not batches:
            raise ValueError("Generation produced no batches.")
        max_length = max(batch.shape[1] for batch in batches)
        padded: list[Tensor] = []
        for batch in batches:
            if batch.shape[1] == max_length:
                padded.append(batch)
                continue
            pad = torch.full(
                (batch.shape[0], max_length - batch.shape[1]),
                self.tokenizer.pad_token_id,
                dtype=batch.dtype,
                device=batch.device,
            )
            padded.append(torch.cat((batch, pad), dim=1))
        return torch.cat(padded, dim=0)

    def _build_group(self, sequences: Tensor, prompt_length: int) -> RolloutGroup:
        if sequences.shape[0] != self.group_size:
            raise ValueError("Generated sequence count does not match group_size.")
        if sequences.shape[1] <= prompt_length:
            raise ValueError("Every generated sequence must contain completion tokens.")

        attention_mask = torch.zeros_like(sequences, dtype=torch.bool)
        attention_mask[:, :prompt_length] = True
        completion_mask = torch.zeros(
            (sequences.shape[0], sequences.shape[1] - 1),
            dtype=torch.bool,
            device=sequences.device,
        )
        texts: list[str] = []
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        for row in range(sequences.shape[0]):
            generated = sequences[row, prompt_length:]
            eos_positions = torch.nonzero(generated == eos_id, as_tuple=False).flatten()
            if eos_positions.numel() > 0:
                completion_length = int(eos_positions[0].item()) + 1
            elif pad_id != eos_id:
                pad_positions = torch.nonzero(generated == pad_id, as_tuple=False).flatten()
                completion_length = (
                    int(pad_positions[0].item()) if pad_positions.numel() > 0 else generated.numel()
                )
            else:
                completion_length = generated.numel()
            if completion_length <= 0:
                raise ValueError("Generation returned an empty completion.")

            end = prompt_length + completion_length
            attention_mask[row, :end] = True
            completion_mask[row, prompt_length - 1 : end - 1] = True
            response_ids = sequences[row, prompt_length:end]
            texts.append(
                self.tokenizer.decode(response_ids, skip_special_tokens=True)
            )

        return RolloutGroup(
            input_ids=sequences,
            attention_mask=attention_mask,
            completion_mask=completion_mask,
            old_logprobs=torch.zeros(
                completion_mask.shape,
                dtype=torch.float32,
                device=sequences.device,
            ),
            texts=tuple(texts),
        )
