from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a dictionary.")
    return cfg


def get_section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return dict(section)


def deep_get(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def config_bool(config: Mapping[str, Any], dotted_key: str, default: bool) -> bool:
    value = deep_get(config, dotted_key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def annotation_track_enabled(config: Mapping[str, Any]) -> bool:
    return config_bool(config, "model.annotation_track", True)


def annotation_loss_enabled(config: Mapping[str, Any]) -> bool:
    return config_bool(config, "model.annotation_loss", True)


def validate_annotation_settings(config: Mapping[str, Any]) -> None:
    if annotation_loss_enabled(config) and not annotation_track_enabled(config):
        raise ValueError("model.annotation_loss=true requires model.annotation_track=true.")


def require(config: Mapping[str, Any], dotted_key: str) -> Any:
    sentinel = object()
    value = deep_get(config, dotted_key, sentinel)
    if value is sentinel:
        raise KeyError(f"Missing required config key: {dotted_key}")
    return value


def resolve_repo_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[2].joinpath(*parts)
