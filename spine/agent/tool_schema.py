"""Tool schema definitions for spine agents."""

from __future__ import annotations

# region book:fnd.perception-tools.core-01
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str
    input: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None

    @model_validator(mode="after")
    def validate_payload(self, info: ValidationInfo) -> "ToolRequest":
        if info.context is None or "tool_schemas" not in info.context:
            raise ValueError("Missing tool schema context")
        schemas = info.context["tool_schemas"]
        schema = schemas.get(self.tool_name)
        if schema is None:
            raise ValueError(f"No input schema for tool: {self.tool_name}")
        self.input = schema.model_validate(self.input).model_dump()
        return self
# endregion


# region book:fnd.perception-tools.example-01
class ToolResponse(BaseModel):
    tool_name: str
    output: dict[str, Any] = Field(default_factory=dict)
    status: str = "ok"
# endregion
