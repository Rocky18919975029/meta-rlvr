from __future__ import annotations

import gc
import json
import math
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

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
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "max_new_tokens",
        "pad_token_id",
        "eos_token_id",
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
        top_k: int = 0,
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
        if top_k < 0:
            raise ValueError("top_k must be non-negative.")
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
        self.top_k = top_k
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
            raise ValueError(
                "A single problem must tokenize to shape [1, prompt_length]."
            )
        if prompt_mask.shape != prompt_ids.shape:
            raise ValueError("Tokenized attention_mask must match input_ids.")
        if not torch.all(prompt_mask == 1):
            raise ValueError(
                "A single unpadded prompt must have an all-one attention mask."
            )
        prompt_ids = prompt_ids.to(self.device)
        prompt_mask = prompt_mask.to(self.device)
        prompt_length = prompt_ids.shape[1]
        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        if not isinstance(max_positions, int):
            raise ValueError(
                "Policy config must define integer max_position_embeddings."
            )
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
                            top_k=self.top_k,
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
                            raise TypeError(
                                "model.generate must return a rank-two tensor."
                            )
                        if sequences.shape[0] != batch_size:
                            raise ValueError(
                                "model.generate returned an unexpected batch size."
                            )
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
                pad_positions = torch.nonzero(
                    generated == pad_id, as_tuple=False
                ).flatten()
                completion_length = (
                    int(pad_positions[0].item())
                    if pad_positions.numel() > 0
                    else generated.numel()
                )
            else:
                completion_length = generated.numel()
            if completion_length <= 0:
                raise ValueError("Generation returned an empty completion.")

            end = prompt_length + completion_length
            attention_mask[row, :end] = True
            completion_mask[row, prompt_length - 1 : end - 1] = True
            response_ids = sequences[row, prompt_length:end]
            texts.append(self.tokenizer.decode(response_ids, skip_special_tokens=True))

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


class VLLMHybridRolloutEngine(TransformersRolloutEngine):
    """vLLM continuous-batching rollout with verl-style sleep/resume.

    vLLM runs in a colocated server process on the same GPU.  Every generation
    transaction wakes weights, loads the task's detached LoRA, wakes the KV
    cache, generates the complete group, unregisters the adapter, and returns
    the server to level-1 sleep before PyTorch computes log-probabilities or
    gradients.  The staged weights -> LoRA -> KV ordering matches verl's
    hybrid-engine lifecycle and avoids allocating KV memory during adapter
    transfer.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        *,
        base_url: str,
        adapter_root: Path,
        request_timeout: float,
        control_timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, tokenizer, **kwargs)
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("vLLM base_url must start with http:// or https://.")
        if request_timeout <= 0:
            raise ValueError("vLLM request_timeout must be positive.")
        if control_timeout <= 0:
            raise ValueError("vLLM control_timeout must be positive.")
        self.base_url = base_url.rstrip("/")
        self.adapter_root = adapter_root
        self.request_timeout = request_timeout
        self.control_timeout = control_timeout
        self.adapter_root.mkdir(parents=True, exist_ok=True)
        status = self._request_json("GET", "/is_sleeping")
        if status != {"is_sleeping": True}:
            raise RuntimeError(
                f"vLLM server must be sleeping before training starts, got {status!r}."
            )
        models = self._request_json("GET", "/v1/models")
        self._require_model_registration(models, "meta-rlvr-base", present=True)

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
        encoded = self.tokenizer(prompt, add_special_tokens=False)
        prompt_ids = encoded.get("input_ids")
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError("Tokenizer must return a non-empty input_ids list.")
        if not all(isinstance(token_id, int) for token_id in prompt_ids):
            raise TypeError("Tokenized prompt IDs must be integers.")
        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        if not isinstance(max_positions, int):
            raise ValueError(
                "Policy config must define integer max_position_embeddings."
            )
        if len(prompt_ids) + self.max_new_tokens > max_positions:
            raise ValueError(
                f"prompt_length ({len(prompt_ids)}) + max_new_tokens "
                f"({self.max_new_tokens}) exceeds max_position_embeddings "
                f"({max_positions})."
            )

        adapter_name = f"meta-rlvr-{uuid4().hex}"
        adapter_dir = Path(
            tempfile.mkdtemp(prefix=f"{adapter_name}-", dir=self.adapter_root)
        )
        loaded = False
        weights_awake = False
        kv_awake = False
        started = time.monotonic()
        progress_bar = None
        if show_progress:
            from tqdm.auto import tqdm

            progress_bar = tqdm(
                total=self.group_size,
                desc=f"{progress_description}: vLLM generate",
                unit="response",
                leave=True,
            )
        try:
            self._save_adapter(adapter_dir, fast_parameters)
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
            self._request_json("POST", "/wake_up?tags=weights")
            weights_awake = True
            load_response = self._request_text(
                "POST",
                "/v1/load_lora_adapter",
                {"lora_name": adapter_name, "lora_path": str(adapter_dir)},
            )
            loaded = True
            expected_load_response = (
                f"Success: LoRA adapter '{adapter_name}' added successfully."
            )
            if load_response != expected_load_response:
                raise RuntimeError(
                    "Unexpected vLLM LoRA load response: "
                    f"{load_response!r}; expected {expected_load_response!r}."
                )
            models = self._request_json("GET", "/v1/models")
            self._require_model_registration(models, adapter_name, present=True)
            self._request_json("POST", "/wake_up?tags=kv_cache")
            kv_awake = True
            status = self._request_json("GET", "/is_sleeping")
            if status != {"is_sleeping": False}:
                raise RuntimeError(f"vLLM failed to wake up fully: {status!r}.")
            response = self._request_json(
                "POST",
                "/v1/completions",
                {
                    "model": adapter_name,
                    "prompt": prompt_ids,
                    "n": self.group_size,
                    "max_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "stream": False,
                    "add_special_tokens": False,
                    "logprobs": 0,
                    "return_token_ids": True,
                },
                timeout=self.request_timeout,
            )
            completion_ids, completion_logprobs = self._completion_data(response)
            if progress_bar is not None:
                progress_bar.update(self.group_size)
                progress_bar.set_postfix_str(
                    f"elapsed={time.monotonic() - started:.1f}s"
                )
        except BaseException as error:
            cleanup_errors = self._finish_generation_transaction(
                adapter_name=adapter_name,
                loaded=loaded,
                awake=weights_awake or kv_awake,
            )
            if cleanup_errors:
                cleanup_summary = "; ".join(str(item) for item in cleanup_errors)
                raise RuntimeError(
                    "vLLM generation failed and cleanup also failed: "
                    f"{cleanup_summary}"
                ) from error
            raise
        else:
            cleanup_errors = self._finish_generation_transaction(
                adapter_name=adapter_name,
                loaded=loaded,
                awake=weights_awake or kv_awake,
            )
            if cleanup_errors:
                error = RuntimeError("vLLM generation succeeded but cleanup failed.")
                raise error from cleanup_errors[0]
        finally:
            if progress_bar is not None:
                progress_bar.close()
            shutil.rmtree(adapter_dir)

        sequences = self._sequences_from_completion_ids(prompt_ids, completion_ids)
        group = self._build_group(sequences, len(prompt_ids))
        rollout_logprobs = self._aligned_rollout_logprobs(
            group,
            completion_logprobs,
        )
        was_training = self.model.training
        self.model.eval()
        try:
            with materialized_fast_parameters(self.model, fast_parameters):
                with torch.no_grad():
                    old_logprobs = self._unadapted_logprobs(
                        group,
                        show_progress=show_progress,
                        progress_description=(
                            f"{progress_description}: old log-probabilities"
                        ),
                    )
        finally:
            self.model.train(was_training)
        group = replace(
            group,
            old_logprobs=old_logprobs,
            rollout_logprobs=rollout_logprobs,
        )
        if show_progress:
            selected = group.completion_mask
            delta = group.rollout_logprobs[selected] - group.old_logprobs[selected]
            print(
                json.dumps(
                    {
                        "stage": progress_description,
                        "vllm_raw_vs_pytorch_raw/mean_delta": delta.mean().item(),
                        "vllm_raw_vs_pytorch_raw/mean_absolute_delta": (
                            delta.abs().mean().item()
                        ),
                        "vllm_raw_vs_pytorch_raw/max_absolute_delta": (
                            delta.abs().max().item()
                        ),
                        "vllm_raw_vs_pytorch_raw/token_count": delta.numel(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return group

    def _finish_generation_transaction(
        self,
        *,
        adapter_name: str,
        loaded: bool,
        awake: bool,
    ) -> list[Exception]:
        errors: list[Exception] = []
        if loaded:
            try:
                unload_response = self._request_text(
                    "POST",
                    "/v1/unload_lora_adapter",
                    {"lora_name": adapter_name},
                )
                expected_unload_response = (
                    f"Success: LoRA adapter '{adapter_name}' removed successfully."
                )
                if unload_response != expected_unload_response:
                    raise RuntimeError(
                        "Unexpected vLLM LoRA unload response: "
                        f"{unload_response!r}; expected "
                        f"{expected_unload_response!r}."
                    )
                models = self._request_json("GET", "/v1/models")
                self._require_model_registration(
                    models,
                    adapter_name,
                    present=False,
                )
            except Exception as error:
                errors.append(error)
        if awake:
            try:
                self._request_json("POST", "/sleep?level=1")
                status = self._request_json("GET", "/is_sleeping")
                if status != {"is_sleeping": True}:
                    raise RuntimeError(f"vLLM failed to enter sleep mode: {status!r}.")
            except Exception as error:
                errors.append(error)
        return errors

    def _save_adapter(
        self,
        adapter_dir: Path,
        fast_parameters: Mapping[str, Tensor],
    ) -> None:
        if not fast_parameters:
            raise ValueError("fast_parameters cannot be empty.")
        if not hasattr(self.model, "save_pretrained"):
            raise TypeError("The policy must implement PEFT save_pretrained().")
        state_dict = {
            name: value.detach().to(device="cpu").contiguous()
            for name, value in fast_parameters.items()
        }
        self.model.save_pretrained(
            adapter_dir,
            state_dict=state_dict,
            safe_serialization=True,
            selected_adapters=["default"],
        )
        expected = {"adapter_config.json", "adapter_model.safetensors"}
        missing = [name for name in expected if not (adapter_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"PEFT did not write adapter files: {missing}.")

    def _completion_data(
        self,
        response: Any,
    ) -> tuple[list[list[int]], list[list[float]]]:
        if not isinstance(response, dict) or not isinstance(
            response.get("choices"), list
        ):
            raise TypeError("vLLM completion response must contain a choices list.")
        choices = sorted(response["choices"], key=lambda choice: choice["index"])
        if len(choices) != self.group_size:
            raise ValueError(
                f"vLLM returned {len(choices)} choices, expected {self.group_size}."
            )
        completions: list[list[int]] = []
        completion_logprobs: list[list[float]] = []
        for expected_index, choice in enumerate(choices):
            if choice.get("index") != expected_index:
                raise ValueError("vLLM completion choice indices are not contiguous.")
            token_ids = choice.get("token_ids")
            if not isinstance(token_ids, list) or not token_ids:
                raise ValueError("Every vLLM completion must contain token_ids.")
            if not all(isinstance(token_id, int) for token_id in token_ids):
                raise TypeError("vLLM completion token IDs must be integers.")
            logprobs = choice.get("logprobs")
            if not isinstance(logprobs, dict):
                raise TypeError("Every vLLM completion must contain logprobs.")
            token_logprobs = logprobs.get("token_logprobs")
            if not isinstance(token_logprobs, list):
                raise TypeError("vLLM token_logprobs must be a list.")
            if len(token_logprobs) != len(token_ids):
                raise ValueError(
                    "vLLM token_ids and token_logprobs must have equal length."
                )
            if not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in token_logprobs
            ):
                raise ValueError("vLLM token_logprobs must be finite numbers.")
            completions.append(token_ids)
            completion_logprobs.append([float(value) for value in token_logprobs])
        return completions, completion_logprobs

    def _sequences_from_completion_ids(
        self,
        prompt_ids: list[int],
        completion_ids: list[list[int]],
    ) -> Tensor:
        max_completion = max(len(tokens) for tokens in completion_ids)
        sequences = torch.full(
            (self.group_size, len(prompt_ids) + max_completion),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        sequences[:, : len(prompt_ids)] = prompt_tensor
        for row, tokens in enumerate(completion_ids):
            sequences[row, len(prompt_ids) : len(prompt_ids) + len(tokens)] = (
                torch.tensor(
                    tokens,
                    dtype=torch.long,
                    device=self.device,
                )
            )
        return sequences

    def _aligned_rollout_logprobs(
        self,
        group: RolloutGroup,
        completion_logprobs: list[list[float]],
    ) -> Tensor:
        if len(completion_logprobs) != group.group_size:
            raise ValueError("vLLM log-probability rows must match group size.")
        aligned = torch.zeros_like(group.old_logprobs)
        for row, values in enumerate(completion_logprobs):
            positions = group.completion_mask[row].nonzero(as_tuple=False).flatten()
            if positions.numel() != len(values):
                raise ValueError(
                    "vLLM log-probability length does not match completion mask."
                )
            aligned[row, positions] = torch.tensor(
                values,
                dtype=aligned.dtype,
                device=aligned.device,
            )
        return aligned

    @staticmethod
    def _require_model_registration(
        response: Any,
        model_name: str,
        *,
        present: bool,
    ) -> None:
        if not isinstance(response, dict) or not isinstance(response.get("data"), list):
            raise TypeError("vLLM /v1/models response must contain a data list.")
        model_ids = {
            item.get("id")
            for item in response["data"]
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        if (model_name in model_ids) != present:
            expectation = "present" if present else "absent"
            raise RuntimeError(
                f"Expected vLLM model {model_name!r} to be {expectation}; "
                f"registered models are {sorted(model_ids)}."
            )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        body = self._request_body(
            method,
            path,
            payload,
            timeout=timeout,
        )
        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError as error:
            rendered = body.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM {method} {path} returned non-JSON content: {rendered!r}."
            ) from error

    def _request_text(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> str:
        body = self._request_body(
            method,
            path,
            payload,
            timeout=timeout,
        )
        if not body:
            raise RuntimeError(f"vLLM {method} {path} returned an empty response.")
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(
                f"vLLM {method} {path} returned non-UTF-8 content."
            ) from error

    def _request_body(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> bytes:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.control_timeout if timeout is None else timeout,
            ) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM {method} {path} failed with HTTP {error.code}: {body}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"vLLM {method} {path} failed: {error}") from error
        return body
