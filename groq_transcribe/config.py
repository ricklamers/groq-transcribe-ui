from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from platformdirs import PlatformDirs


APP_NAME = "groq-transcribe-2"
CONFIG_FILE_NAME = "config.json"
API_KEY_ENV = "GROQ_API_KEY"


def _config_dir() -> Path:
    dirs = PlatformDirs(appname=APP_NAME)
    path = Path(dirs.user_config_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _config_path() -> Path:
    return _config_dir() / CONFIG_FILE_NAME


@dataclass
class AppConfig:
    api_key: Optional[str] = None

    @staticmethod
    def load() -> "AppConfig":
        path = _config_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return AppConfig(api_key=data.get("api_key") or None)
            except Exception:
                return AppConfig()
        return AppConfig()

    def save(self) -> None:
        payload = {"api_key": self.api_key}
        _config_path().write_text(json.dumps(payload, indent=2))


def get_api_key() -> Optional[str]:
    # Prefer environment variable, then config file
    env_key = os.environ.get(API_KEY_ENV)
    if env_key:
        return env_key
    cfg = AppConfig.load()
    return cfg.api_key


def set_api_key(value: str) -> None:
    # Persist and set env var for current process
    cfg = AppConfig.load()
    cfg.api_key = value.strip() if value else None
    cfg.save()
    if cfg.api_key:
        os.environ[API_KEY_ENV] = cfg.api_key
    else:
        os.environ.pop(API_KEY_ENV, None)
