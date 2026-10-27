"""Governance checklist utilities for spine runs."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GovernanceChecklist(BaseModel):
    data_quality_ok: bool
    model_eval_ok: bool
    policy_eval_ok: bool
    notes: list[str] = Field(default_factory=list)


def write_checklist(output_dir: Path, checklist: GovernanceChecklist) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "governance_checklist.json"
    path.write_text(json.dumps(checklist.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path
