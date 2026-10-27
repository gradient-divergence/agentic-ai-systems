"""Five canonical components; evaluation manifests are cross-references only."""

import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
from typing import cast

from pydantic import field_validator, model_validator

from refund_agent import contracts as c
from refund_agent.adaptation.prompt import StructuredPrompt
from refund_agent.memory.register import Snapshot, project
from refund_agent.release import extensions
from refund_agent.release.agent_evaluation_resources import IMPLEMENTATION_FILES
from refund_agent.release.candidate import PACKAGE, Candidate
from refund_agent.retrieval.clauses import policy_clauses
from refund_agent.retrieval.encoders import CountEncoder
from refund_agent.retrieval.index import PolicyIndex
from refund_agent.rules import _RISK_THRESHOLD
from refund_agent.selection import fingerprint

NAME = "mlops-mle.bundle"
ORDER = ("model", "prompt", "tools", "policy", "index")


def public_record(value):
    """Reject explicitly secret items before canonicalization or hashing."""
    if isinstance(value, dict):
        if value.get("secret") is True:
            raise ValueError("a secret is not a released artifact")
        for item in value.values():
            public_record(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            public_record(item)
    fingerprint(value)  # Reject non-JSON and nonfinite values.
    return json.loads(json.dumps(value))


class Component(c.Record):
    name: str
    items: dict

    @field_validator("items", mode="before")
    @classmethod
    def canonical_items(cls, value):
        return public_record(value)

    @model_validator(mode="after")
    def public_items(self):
        if self.name not in ORDER:
            raise ValueError("unknown bundle component")
        public_record(self.items)
        return self

    @property
    def fingerprint(self):
        return fingerprint(public_record(self.items))


class VersionBundle(c.Record):
    model: Component
    prompt: Component
    tools: Component
    policy: Component
    index: Component
    refs: dict[str, str]

    @model_validator(mode="after")
    def fixed_order(self):
        if tuple(part.name for part in self.components) != ORDER:
            raise ValueError("bundle needs five components in the fixed order")
        versions = self.policy.items["versions"]
        if not versions or versions != sorted(set(versions)):
            raise ValueError("declare distinct sorted policy versions")
        return self

    @property
    def components(self):
        return self.model, self.prompt, self.tools, self.policy, self.index

    @property
    def fingerprint(self):
        return fingerprint([part.fingerprint for part in self.components])

    def covers(self, version):
        return version in self.policy.items["versions"]

    def record(self):
        return {**self.model_dump(mode="json"), "fingerprint": self.fingerprint}

    @classmethod
    def from_record(cls, row):
        row = dict(row)
        expected = row.pop("fingerprint")
        bundle = cls.model_validate_json(json.dumps(row))
        if bundle.fingerprint != expected:
            raise ValueError("bundle fingerprint differs from its components")
        return bundle


@extensions.resource_guard(0)
def agent_code_files(candidate: Candidate) -> dict[str, str]:
    """Chapter 29's fixed core plus the *selected* component closure, no scan."""
    paths = set(IMPLEMENTATION_FILES)
    for item in extensions.records(candidate.tools_composition.get("components", ())):
        for name in item["modules"]:
            path = PACKAGE.joinpath(*name.split(".")[1:])
            path = (
                path.with_suffix(".py")
                if path.with_suffix(".py").is_file()
                else path / "__init__.py"
            )
            paths.add(str(path.relative_to(PACKAGE)))
        paths.update(item["files"])
    return {path: sha256((PACKAGE / path).read_bytes()).hexdigest() for path in sorted(paths)}


def tool_contract_fingerprint():
    from refund_agent.tools import TOOL_NAMES

    return fingerprint(
        {
            "schemas": {
                record.__name__: record.model_json_schema()
                for record in (c.ToolCall, c.ToolArguments, c.ToolResult)
            },
            "tools": sorted(TOOL_NAMES),
            "effects": {
                "order.lookup": "read",
                "shipment.status": "read",
                "policy.retrieve": "read",
                "refund.propose": "proposal",
                "approval.request": "review request",
                "refund.execute": "payment",
                "audit.append": "audit",
            },
        }
    )


def environment_fingerprint():
    root = PACKAGE.parent
    match = re.search(
        r'^requires-python\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(), re.M
    )
    if match is None:
        raise ValueError("environment must name its Python requirement")
    return fingerprint(
        {"lock": sha256((root / "uv.lock").read_bytes()).hexdigest(), "requires_python": match[1]}
    )


def snapshot_record(snapshot: Snapshot):
    def record(row):
        return {
            key: value.isoformat() if hasattr(value, "isoformat") else value
            for key, value in asdict(row).items()
        }

    return {
        "at": snapshot.at.isoformat(),
        "records": [record(r) for r in snapshot.records],
        "texts": dict(cast(Mapping[str, str], snapshot.texts)),
    }


def release_settings(snapshot: Snapshot, versions):
    policies = project(snapshot, snapshot.at)
    versions = sorted(set(versions))
    if not set(versions) <= {p.version for p in policies}:
        raise ValueError("declared version absent from register snapshot")
    clauses = policy_clauses(policies, snapshot.at)
    index = PolicyIndex.build(
        policies, snapshot.at, CountEncoder().fit([clause.text for clause in clauses])
    )
    return {
        "versions": versions,
        "register_snapshot": snapshot_record(snapshot),
        "index_manifest": index.manifest,
    }


@extensions.resource_guard(0)
def from_candidate(candidate, *, register_snapshot=None, versions=None, release_record=None):
    selected = candidate.tools_composition.get("components", ())
    settings = next((r["settings"] for r in selected if r["name"] == NAME), None)
    if register_snapshot is not None:
        supplied = release_settings(register_snapshot, versions)
        if settings is not None and settings != supplied:
            raise ValueError("loaded register or declared coverage differs from the candidate")
        settings = supplied
    if settings is None:
        raise ValueError("a bundle needs explicit register and policy coverage")
    candidate_refs = {"candidate": candidate.fingerprint, "suite": candidate.suite["fingerprint"]}
    if release_record is not None:
        if (release_record["candidate_fingerprint"], release_record["suite_fingerprint"]) != (
            candidate_refs["candidate"],
            candidate_refs["suite"],
        ):
            raise ValueError("release belongs to another candidate or evaluation manifest")
        candidate_refs["release"] = release_record["record_id"]
    parts = {
        "model": {
            "scripted_model": candidate.model,
            "risk_score": "text rule",
            "risk_threshold": _RISK_THRESHOLD,
            "live_model": None,
        },
        "prompt": {
            "template": StructuredPrompt().version,
            "fingerprint": StructuredPrompt().fingerprint,
            "code": sha256((PACKAGE / "adaptation/prompt.py").read_bytes()).hexdigest(),
        },
        "tools": {
            "agent": candidate.agent,
            "agent_code_files": agent_code_files(candidate),
            "core": candidate.tools_composition["fingerprint"],
            "tool_contracts": tool_contract_fingerprint(),
            "runtime": candidate.runtime,
            "adapter": candidate.adapter,
            "verifier": {
                key: candidate.tools_composition[key] for key in ("verifier", "verifier_version")
            },
            "environment": environment_fingerprint(),
            "components": [
                {key: row[key] for key in ("name", "version", "fingerprint", "settings")}
                for row in selected
                if row["name"] != NAME
            ],
        },
        "policy": {
            "versions": settings["versions"],
            "register_snapshot": fingerprint(settings["register_snapshot"]),
            "offer_rule": {key: candidate.offer_policy[key] for key in ("name", "policy")},
        },
        "index": settings["index_manifest"],
    }
    for row in selected:
        if row["name"] == "data-systems.contracts":
            parts["tools"]["data_contracts"] = {"version": row["version"], **row["settings"]}
        elif row["name"] == "batch-stream":
            parts["tools"]["event_time"] = {"version": row["version"], **row["settings"]}
    extra = extensions.contributions(candidate, "bundle_items")
    for contribution in extra.values():
        for part, items in contribution.items():
            if part not in parts or parts[part].keys() & items.keys():
                raise ValueError("bundle contribution conflicts with an owned item")
            parts[part].update(public_record(items))
    return VersionBundle(
        **{name: Component(name=name, items=items) for name, items in parts.items()},
        refs=candidate_refs,
    )
