import importlib.util
from pathlib import Path

from packaging.version import Version

import pytest


module_path = Path(__file__).parents[1] / "src/meta_rlvr/vllm_preflight.py"
launcher_path = Path(__file__).parents[1] / "scripts/vllm_hybrid_servers.sh"
spec = importlib.util.spec_from_file_location("vllm_preflight", module_path)
assert spec is not None and spec.loader is not None
vllm_preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vllm_preflight)


def test_vllm_preflight_accepts_validated_stack(monkeypatch) -> None:
    versions = {
        "vllm": Version("0.11.0"),
        "fastapi": Version("0.137.0"),
        "starlette": Version("0.52.1"),
        "prometheus-fastapi-instrumentator": Version("8.0.1"),
    }
    monkeypatch.setattr(
        vllm_preflight,
        "_installed_version",
        versions.__getitem__,
    )

    assert vllm_preflight.validate_vllm_environment() == {
        name: str(package_version) for name, package_version in versions.items()
    }


def test_vllm_preflight_rejects_broken_instrumentator(monkeypatch) -> None:
    versions = {
        "vllm": Version("0.11.0"),
        "fastapi": Version("0.137.0"),
        "starlette": Version("0.52.1"),
        "prometheus-fastapi-instrumentator": Version("7.1.0"),
    }
    monkeypatch.setattr(
        vllm_preflight,
        "_installed_version",
        versions.__getitem__,
    )

    with pytest.raises(RuntimeError, match=">=8.0.1"):
        vllm_preflight.validate_vllm_environment()


def test_vllm_sleep_process_does_not_inherit_expandable_segments() -> None:
    launcher = launcher_path.read_text(encoding="utf-8")
    server_environment = launcher.split("setsid env", maxsplit=1)[1].split(
        "python -m vllm.entrypoints.openai.api_server",
        maxsplit=1,
    )[0]

    assert "-u PYTORCH_CUDA_ALLOC_CONF" in server_environment
    assert "-u PYTORCH_ALLOC_CONF" in server_environment
