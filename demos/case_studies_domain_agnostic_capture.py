"""Explicit source capture; the complete run never substitutes missing launch rows."""

import sys
from pathlib import Path

from demos import regions as _regions
from refund_agent.transfer.case_studies_domain_agnostic_facts import require_inventory

ROOT = Path(__file__).resolve().parents[1]
INDEPENDENT = tuple(
    f"fnd.case-studies-domain-agnostic.{name}-01"
    for name in ("packets", "triage", "hospital", "bank", "balance", "keys")
)
VERDICTS = "fnd.case-studies-domain-agnostic.verdicts-01"


def capture_chapter(destination: Path, *, ids: tuple[str, ...]) -> dict[str, str]:
    required = (*INDEPENDENT, VERDICTS) if VERDICTS in ids else INDEPENDENT
    require_inventory(ids, required, "capture request")
    extractor = vars(_regions)
    source = ROOT / "demos/case_studies_domain_agnostic_demo.py"
    regions = extractor["extract_snippets_from_text"](source.read_text(), source)
    require_inventory((s.snippet_id for s in regions), (*INDEPENDENT, VERDICTS), "capture source")
    snippets = {s.snippet_id: s for s in regions}
    selected = {name: snippets[name] for name in ids}
    outputs = extractor["capture_outputs"](selected, sys.executable, timeout=120)
    require_inventory(outputs, ids, "capture result")
    out = Path(destination) / "book_outputs"
    out.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (out / f"_{name}.txt").write_text(content)
    return outputs
