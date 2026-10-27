"""Typed configuration for spine runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class SpineConfig(BaseModel):
    dataset_slug: str = Field("yelp-dataset")
    subset_dir: Path = Field(Path("companion/spine/data/subsets/yelp_100k"))
    output_dir: Path = Field(Path("companion/spine/outputs"))
    seed: int = Field(13, ge=0)
    use_gpu: bool = Field(False)
    run_id: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))

    @field_validator("subset_dir")
    @classmethod
    def _validate_subset_dir(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Subset directory does not exist: {value}")
        return value

    @field_validator("output_dir")
    @classmethod
    def _ensure_output_dir(cls, value: Path) -> Path:
        value.mkdir(parents=True, exist_ok=True)
        return value

    @property
    def run_dir(self) -> Path:
        run_path = self.output_dir / self.run_id
        run_path.mkdir(parents=True, exist_ok=True)
        return run_path
