"""The architecture chapter prints the order record's contract, four releases of the
order system checked against it, and the baseline's drafts. These tests run the demo's
regions in order, in one namespace, and check what the chapter says about them."""

import ast
import contextlib
import io
import json
from pathlib import Path

import pytest

from demos import regions as _regions
from refund_agent.contracts import Order, Outcome

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/modern_ai_stack_demo.py"
PREFIX = "fnd.modern-ai-stack."
extract = vars(_regions)[
    "extract_snippets_from_text"
]


@pytest.fixture(scope="module")
def run() -> tuple[dict, dict[str, str]]:
    """Execute the whole demo once, capturing each output region's stdout."""
    text = DEMO.read_text(encoding="utf-8")
    regions = {s.snippet_id: s for s in extract(text, DEMO)}
    lines = text.splitlines(keepends=True)
    namespace: dict = {"__name__": "__main__", "__file__": str(DEMO)}
    printed: dict[str, str] = {}
    previous = 0
    for snippet in sorted(regions.values(), key=lambda s: s.start_line):
        setup = "\n" * previous + "".join(lines[previous : snippet.start_line - 1])
        exec(compile(setup, str(DEMO), "exec"), namespace)
        body = "\n" * (snippet.start_line - 1) + snippet.content
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(body, str(DEMO), "exec"), namespace)
        printed[snippet.snippet_id.removeprefix(PREFIX)] = stdout.getvalue()
        previous = snippet.end_line
    return namespace, printed


def test_contract_excerpt_preserves_package_statements_within_the_measure(run: tuple[dict, dict]) -> None:
    printed = run[1]["contract-01"].splitlines()
    source = (ROOT / "refund_agent/contracts.py").read_text(encoding="utf-8")
    assert printed[0] == "# refund_agent/contracts.py, excerpt"
    assert all(len(line) <= 79 for line in printed)
    package = ast.parse(source)
    excerpt = ast.parse("\n".join(printed))
    canonical = {
        node.name if isinstance(node, ast.ClassDef) else node.targets[0].id: node
        for node in package.body
        if isinstance(node, ast.ClassDef)
        or isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    assert [
        node.name if isinstance(node, ast.ClassDef) else node.targets[0].id
        for node in excerpt.body
    ] == ["Cents", "Record", "Item", "Order"]
    for node in excerpt.body:
        if isinstance(node, ast.Assign):
            assert ast.dump(node) == ast.dump(canonical["Cents"])
            continue
        original = canonical[node.name]
        if node.name in {"Item", "Order"}:
            assert ast.dump(node) == ast.dump(original)
        assert [ast.dump(base) for base in node.bases] == [ast.dump(base) for base in original.bases]
        for statement in node.body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                assert statement.value.value is Ellipsis
            else:
                assert ast.dump(statement) in [ast.dump(item) for item in original.body]
    record = next(node for node in excerpt.body if isinstance(node, ast.ClassDef) and node.name == "Record")
    config = next(node.value for node in record.body if isinstance(node, ast.Assign))
    assert ast.literal_eval(config) == Order.model_config
    assert "Cents = Annotated[int, Field(strict=True, gt=0)]" in printed
    assert "    paid_cents: Cents" in printed


def test_sent_is_the_generated_order_with_the_chapters_amounts(run: tuple[dict, dict]) -> None:
    ns = run[0]
    generated = ns["generated"].model_dump(mode="json")
    sent = ns["sent"]
    changed = {key for key in generated if generated[key] != sent[key]}
    assert changed == {"items", "paid_cents"}
    assert sent["paid_cents"] == 4200 and generated["prior_refunds"] == []
    assert Order.model_validate_json(json.dumps(sent)).paid_cents == 4200


def test_three_releases_are_rejected_and_dollars_pass(run: tuple[dict, dict]) -> None:
    out = run[1]["compat-01"]
    assert "renamed field: rejected" in out
    assert "added field: rejected" in out
    assert "naive time: rejected" in out
    assert "in dollars: accepted as 42 cents" in out
    assert Order.model_validate_json(json.dumps(run[0]["sent"])).observed_at.utcoffset() is not None


def test_baseline_approves_as_sent_and_denies_in_dollars(run: tuple[dict, dict]) -> None:
    ns, printed = run
    out = printed["drafts-01"]
    assert "as sent: balance 4200 cents; approve, tier 1, eligible" in out
    assert "renamed field: data load stopped, 0 of 600 drafted" in out
    assert "in dollars: balance 42 cents; deny, tier 1, amount_over_policy" in out
    changed = ns["load"](ns["releases"]["in dollars"])
    request = next(r for r in changed.requests if r.request_id == "r-7-0007")
    decision = ns["baseline"](changed, [request])[0]
    assert decision.outcome == Outcome.DENY and not decision.executed


def test_scorecard_before_and_after_the_dollars_release(run: tuple[dict, dict]) -> None:
    rows = [row.strip("|").split("|") for row in run[1]["scorecard-01"].splitlines()]
    before = [cell.strip() for cell in rows[2][1:]]
    after = [cell.strip() for cell in rows[3][1:]]
    assert before == ["40/40", "4", "22", "14", "0"]
    assert after == ["16/40", "0", "35", "5", "0"]


def test_invariant_holds_on_every_generated_order_and_fails_in_dollars(
    run: tuple[dict, dict],
) -> None:
    out = run[1]["invariant-01"]
    assert "items' total: 2000 of 2000" in out
    assert "paid_cents 42, items 4200" in out


def test_invariant_accepts_release_that_moves_every_amount_to_dollars(
    run: tuple[dict, dict],
) -> None:
    ns, printed = run
    assert "all amounts in dollars: paid_cents 42, items 42, accepted" in printed["invariant-01"]
    order = Order.model_validate_json(json.dumps(ns["all_dollars"]))
    assert order.paid_cents == 42
    assert [(item.quantity, item.unit_cents) for item in order.items] == [(1, 42)]
    assert order.prior_refunds == []
    assert ns["sent"]["paid_cents"] == ns["sent"]["items"][0]["unit_cents"] == 4200


def test_items_total_multiplies_quantities_and_sums_items(run: tuple[dict, dict]) -> None:
    order = {
        "items": [
            {"name": "lamp", "quantity": 2, "unit_cents": 2100},
            {"name": "bulb", "quantity": 3, "unit_cents": 200},
        ]
    }
    assert run[0]["items_total"](order) == 4800


def test_invariant_rejects_schema_valid_mismatched_payment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate = Order.model_validate_json

    def mismatched_payment(payload: str) -> Order:
        record = json.loads(payload)
        record["paid_cents"] = 43
        return validate(json.dumps(record))

    monkeypatch.setattr(Order, "model_validate_json", mismatched_payment)
    region = next(
        s for s in extract(DEMO.read_text(encoding="utf-8"), DEMO)
        if s.snippet_id == PREFIX + "invariant-01"
    )
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(region.content, str(DEMO), "exec"), {})
    assert stdout.getvalue().splitlines()[-1] == (
        "all amounts in dollars: paid_cents 43, items 42, rejected"
    )


def test_invariant_region_runs_on_its_own(run: tuple[dict, dict]) -> None:
    region = next(
        s for s in extract(DEMO.read_text(encoding="utf-8"), DEMO)
        if s.snippet_id == PREFIX + "invariant-01"
    )
    assert max(map(len, region.content.splitlines())) <= 66
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(region.content, str(DEMO), "exec"), {})
    assert stdout.getvalue() == run[1]["invariant-01"]


def test_exercise_three_key(run: tuple[dict, dict]) -> None:
    out = run[1]["exercise-03"]
    assert "paid as text: rejected" in out
    assert "payment pending: rejected" in out
    assert "local time: balance 4200 cents; approve, tier 1, eligible" in out
    assert "hundredths: balance 420000 cents; approve, tier 1, eligible" in out
    local = Order.model_validate_json(json.dumps(run[0]["exercise"]["local time"]))
    assert local.observed_at.isoformat() == "2026-03-22T15:07:27+00:00"


def test_printed_lines_fit_the_code_measure(run: tuple[dict, dict]) -> None:
    for name in ("compat-01", "drafts-01", "scorecard-01", "invariant-01", "exercise-03"):
        assert max(map(len, run[1][name].splitlines())) <= 66, name
    listing = next(
        s
        for s in extract(DEMO.read_text(encoding="utf-8"), DEMO)
        if s.snippet_id == PREFIX + "compat-01"
    )
    assert max(map(len, listing.content.splitlines())) <= 66
