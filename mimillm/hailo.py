"""Optional HailoRT discovery and HEF inspection helpers.

This module deliberately does not present Hailo as a mimiLLM tensor backend.
HailoRT executes a compiled HEF artifact; it cannot execute mimiLLM
SafeTensors or an arbitrary Python graph directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HailoRuntimeInfo:
    """Installed HailoRT version and PCIe devices visible to Python."""

    runtime_version: str | None
    device_ids: tuple[str, ...]
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.error is None and self.runtime_version is not None and bool(self.device_ids)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["device_ids"] = list(self.device_ids)
        result["available"] = self.available
        return result


@dataclass(frozen=True)
class HailoHefInfo:
    """Static names exposed by one compiled Hailo Executable Format file."""

    path: str
    network_groups: tuple[str, ...]
    networks: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for field in ("network_groups", "networks", "inputs", "outputs"):
            result[field] = list(result[field])
        return result


def _hailo_platform() -> Any:
    try:
        import hailo_platform
    except ImportError as exc:
        raise RuntimeError(
            "HailoRT Python package is unavailable. Install the HailoRT package "
            "matching the device firmware, or expose the system package to the venv."
        ) from exc
    return hailo_platform


def inspect_hailo_runtime() -> HailoRuntimeInfo:
    """Return HailoRT/device information without making Hailo a required dependency."""
    try:
        platform = _hailo_platform()
    except RuntimeError as exc:
        return HailoRuntimeInfo(None, (), str(exc))
    version = getattr(platform, "__version__", None)
    try:
        devices = tuple(str(value) for value in platform.Device.scan())
    except Exception as exc:  # HailoRT exposes its own extension exception types.
        return HailoRuntimeInfo(
            str(version) if version is not None else None,
            (),
            f"HailoRT device scan failed: {exc}",
        )
    return HailoRuntimeInfo(
        str(version) if version is not None else None,
        devices,
    )


def hailo_is_available() -> bool:
    """Return True when HailoRT imports and at least one device is visible."""
    return inspect_hailo_runtime().available


def _info_names(values: Any) -> tuple[str, ...]:
    return tuple(str(getattr(value, "name", value)) for value in values)


def inspect_hailo_hef(path: str | Path) -> HailoHefInfo:
    """Parse a ready `.hef` and report its network/input/output names."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"HEF file not found: {source}")
    platform = _hailo_platform()
    hef = platform.HEF(str(source))
    return HailoHefInfo(
        path=str(source),
        network_groups=tuple(str(value) for value in hef.get_network_group_names()),
        networks=tuple(str(value) for value in hef.get_networks_names()),
        inputs=_info_names(hef.get_input_vstream_infos()),
        outputs=_info_names(hef.get_output_vstream_infos()),
    )
