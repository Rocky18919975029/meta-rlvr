from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version


def _installed_version(distribution: str) -> Version:
    try:
        return Version(version(distribution))
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"Required vLLM rollout dependency is not installed: {distribution}."
        ) from error


def validate_vllm_environment() -> dict[str, str]:
    versions = {
        distribution: _installed_version(distribution)
        for distribution in (
            "vllm",
            "fastapi",
            "starlette",
            "prometheus-fastapi-instrumentator",
        )
    }
    if not Version("0.11.0") <= versions["vllm"] < Version("0.12.0"):
        raise RuntimeError(
            "The hybrid backend is validated only for vLLM 0.11.x; "
            f"found {versions['vllm']}."
        )
    if versions["prometheus-fastapi-instrumentator"] < Version("8.0.1"):
        raise RuntimeError(
            "prometheus-fastapi-instrumentator>=8.0.1 is required. Older "
            "versions crash vLLM's FastAPI routes with recent FastAPI releases. "
            "Install the 8.0.1 wheel into the offline HPC environment before "
            "submitting a GPU job."
        )
    return {name: str(package_version) for name, package_version in versions.items()}


def main() -> None:
    print(json.dumps(validate_vllm_environment(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
