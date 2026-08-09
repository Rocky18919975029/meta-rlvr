from __future__ import annotations

import json
import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).parents[1] / "src/meta_rlvr/fidelity_preflight.py"
_SPEC = importlib.util.spec_from_file_location("fidelity_preflight", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
inspect_fidelity_checkpoint = _MODULE.inspect_fidelity_checkpoint


def test_fidelity_preflight_accepts_legacy_source_sampling(tmp_path) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoint-3"
    checkpoint.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "token_meta_coefficient": 1.0,
                "attn_implementation": "sdpa",
                "temperature": 1.0,
                "top_p": 0.7,
                "top_k": 0,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"completed_steps": 3, "world_size": 8}),
        encoding="utf-8",
    )

    result = inspect_fidelity_checkpoint(checkpoint, expected_world_size=8)

    assert result["source_sampling_matches_probe"] is False
    assert result["source_sampling"] == {
        "temperature": 1.0,
        "top_p": 0.7,
        "top_k": 0,
    }
    assert result["probe_sampling"] == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
    }
