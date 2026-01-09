from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    catalogue_path: str = Field(default="catalogue.csv")
    queries_path: str = Field(default="queries.csv")
    artifacts_dir: str = Field(default="artifacts")
    index_version: str = Field(default="1.0.0")

    @property
    def versioned_artifacts_dir(self) -> str:
        return os.path.join(self.artifacts_dir, f"v{self.index_version}")

    def resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        root = Path(__file__).parent.parent.parent
        return str(root / path)


def get_settings() -> AppSettings:
    return AppSettings()
