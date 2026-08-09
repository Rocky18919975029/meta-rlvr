from __future__ import annotations

import argparse
import json
from pathlib import Path


PROBE_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 0}


def inspect_fidelity_checkpoint(
    checkpoint: Path,
    *,
    expected_world_size: int | None = None,
) -> dict[str, object]:
    checkpoint = checkpoint.resolve()
    source_run_dir = checkpoint.parent
    config_path = source_run_dir / "run_config.json"
    state_path = checkpoint / "trainer_state.json"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing source run configuration: {config_path}")
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint trainer state: {state_path}")

    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    trainer_state = json.loads(state_path.read_text(encoding="utf-8"))
    if float(source_config.get("token_meta_coefficient", 0.0)) <= 0:
        raise ValueError("Fidelity probe requires a token-confidence checkpoint.")
    if source_config.get("attn_implementation") != "sdpa":
        raise ValueError("Fidelity probe requires the corrected SDPA token policy.")

    world_size = int(trainer_state["world_size"])
    if expected_world_size is not None and world_size != expected_world_size:
        raise ValueError(
            f"Checkpoint world size is {world_size}, expected {expected_world_size}."
        )
    source_sampling = {
        "temperature": float(source_config.get("temperature", 1.0)),
        "top_p": float(source_config.get("top_p", 1.0)),
        "top_k": int(source_config.get("top_k", 0)),
    }
    return {
        "event": "fidelity_checkpoint_preflight_passed",
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(trainer_state["completed_steps"]),
        "checkpoint_world_size": world_size,
        "source_sampling": source_sampling,
        "source_sampling_matches_probe": source_sampling == PROBE_SAMPLING,
        "probe_sampling": PROBE_SAMPLING,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a fidelity checkpoint before requesting GPUs."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    args = parser.parse_args()
    if args.expected_world_size <= 0:
        raise ValueError("expected_world_size must be positive.")
    print(
        json.dumps(
            inspect_fidelity_checkpoint(
                args.checkpoint,
                expected_world_size=args.expected_world_size,
            ),
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
