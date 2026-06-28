from __future__ import annotations

from pathlib import Path

from config.settings import get_settings, resolve_project_root

PROJECT_ROOT = resolve_project_root()
SETTINGS = get_settings()

__all__ = ["PROJECT_ROOT", "SETTINGS", "Path"]
