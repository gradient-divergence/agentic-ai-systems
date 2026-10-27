"""The perception and tools demo, checked against the values its chapter's brief worked by hand.

The functions the chapter prints are run here from the demo's own regions; the refund
agent's values are recomputed from the package; and the committed outputs that the
chapter includes are checked to carry the same numbers.
"""

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from refund_agent import rules
from refund_agent.agent import Agent
from refund_agent.data.generate import generate
from refund_agent.tools import Tools

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos" / "perception_tools_demo.py"
OUTPUTS = ROOT / "book_outputs"


@pytest.mark.parametrize("name", ["freshness-01", "contracts-01", "execute-01"])
def test_source_outputs_fit_measure_and_remain_valid_python(name: str) -> None:
    text = (OUTPUTS / f"_fnd.perception-tools.{name}.txt").read_text()
    assert max(map(len, text.splitlines())) <= 79
    ast.parse(text)


def region(name: str, namespace: dict | None = None) -> dict:
    """Execute one region of the demo, the code the chapter prints or quotes."""
    text = DEMO.read_text(encoding="utf-8")
    start = text.index(f"# region book:fnd.perception-tools.{name}")
    end = text.index(f"# endregion book:fnd.perception-tools.{name}")
    if namespace is None:
        namespace = {"__file__": str(DEMO)}
    exec(compile(text[start:end], str(DEMO), "exec"), namespace)
    return namespace


def output(name: str) -> str:
    return (OUTPUTS / f"_fnd.perception-tools.{name}.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def conv() -> dict:
    return region("conv-01")


@pytest.fixture(scope="module")
def corpus():
    return generate(7)


def test_cross_correlation_and_convolution(conv: dict) -> None:
    cross_correlate = conv["cross_correlate"]
    x, w = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 0], [0, -1]]
    assert cross_correlate(x, w).tolist() == [[-4, -4], [-4, -4]]
    assert cross_correlate(x, np.flip(w)).tolist() == [[4, 4], [4, 4]]
    line = np.zeros((5, 5))
    line[1:4, 2] = 1
    z = cross_correlate(line, [[0, 1, 0]] * 3, pad=1)
    assert z.shape == (5, 5)
    assert z[:, 2].tolist() == [1, 2, 3, 2, 1]
    assert not np.delete(z, 2, axis=1).any()


@pytest.mark.parametrize("shape", [(3, 5), (5, 3)])
def test_cross_correlation_of_a_non_square_image(conv: dict, shape: tuple[int, int]) -> None:
    image = np.arange(shape[0] * shape[1]).reshape(shape)
    result = conv["cross_correlate"](image, [[1, 0], [0, -1]])
    expected = image[:-1, :-1] - image[1:, 1:]
    np.testing.assert_array_equal(result, expected)


def test_sizes_and_parameters(conv: dict) -> None:
    size, parameters = conv["output_size"], conv["parameters"]
    assert size(224, 3, stride=2, pad=1) == 112
    assert size(224, 3, pad=1, dil=2) == 222
    assert parameters(3, 3, 32) == 896
    assert 224 * 224 * 3 * 112 * 112 * 32 + 112 * 112 * 32 == 60_423_544_832
    for n, k, stride, pad, dil in [(9, 3, 2, 1, 1), (10, 3, 1, 1, 2), (32, 5, 1, 2, 1)]:
        z = conv["cross_correlate"](np.ones((n, n)), np.ones((k, k)), 0, stride, pad, dil)
        assert z.shape == (size(n, k, stride, pad, dil),) * 2


def test_stride_keeps_shifts_only_by_its_multiples(conv: dict) -> None:
    rng = np.random.default_rng(0)
    image, kernel = rng.normal(size=(12, 12)), rng.normal(size=(3, 3))
    base = conv["cross_correlate"](image, kernel)
    shifted = conv["cross_correlate"](np.roll(image, 1, axis=1), kernel)
    assert np.allclose(shifted[:, 1:], base[:, :-1])
    strided = conv["cross_correlate"](image, kernel, stride=2)
    by_one = conv["cross_correlate"](np.roll(image, 1, axis=1), kernel, stride=2)
    by_two = conv["cross_correlate"](np.roll(image, 2, axis=1), kernel, stride=2)
    assert np.allclose(by_two[:, 1:], strided[:, :-1])
    assert not np.allclose(by_one[:, 1:], strided[:, :-1])


def test_receptive_fields_and_base_rates(capsys: pytest.CaptureFixture[str]) -> None:
    vision = region("vision-01")
    capsys.readouterr()
    field = vision["receptive_field"]
    assert field([(3, 1, 1)] * 3) == 7
    assert field([(3, 2, 1), (3, 1, 1), (3, 1, 1)]) == 11
    assert field([(3, 1, 1), (3, 1, 2), (3, 1, 1)]) == 9
    assert field([(3, 1, 1), (3, 1, 2), (3, 1, 4)]) == 15  # Yu and Koltun, Figure 1
    share = vision["flagged_defective"]
    assert round(share(0.02, 0.9, 0.1), 6) == 0.155172
    assert round(share(0.02, 0.9, 0.3), 6) == 0.057692
    assert round(share(0.02, 0.9, 0.25), 3) == 0.068


def test_reads_of_the_default_request(corpus) -> None:
    request = next(r for r in corpus.requests if r.request_id == "r-7-0007")
    trace = Agent(Tools(corpus)).run(request)
    ages = {s.phase.value: s.result.age_seconds for s in trace.steps if s.result.age_seconds}
    assert ages == {"order": 60.0, "shipment": 120.0, "policy": 3600.0}
    margins = {source: rules._FRESHNESS_LIMITS[source] - age for source, age in ages.items()}
    assert margins == {"order": 1740, "shipment": 1680, "policy": 2588400}
    assert trace.decision.executed and trace.receipt is not None


def test_stale_carrier_is_referred_one_second_over(corpus) -> None:
    request = next(r for r in corpus.requests if r.failure == "stale_carrier")
    tools = Tools(corpus)
    trace = Agent(tools).run(request)
    shipment = next(s for s in trace.steps if s.phase.value == "shipment")
    assert shipment.result.age_seconds == 1801.0
    assert trace.decision.reason == "stale_shipment"
    assert len(tools.audits) == 1 and not tools.payments


def test_freshness_boundary() -> None:
    def check(age: float | None) -> str | None:
        ages = {"order": 60.0, "policy": 3600.0}
        if age is not None:
            ages["shipment"] = age
        return rules.freshness_problem(ages)

    assert check(1800.0) is None
    assert check(1801.0) == "stale_shipment"
    assert check(-1.0) == "future_shipment"
    assert check(None) == "missing_shipment_freshness"
    assert check(math.nan) == "invalid_shipment_freshness"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("stale-01", ["r-7-0002: 2499 cents", "1,801 s", "stale_shipment", "money moved: false"]),
        ("reads-01", ["1,740 s", "1,680 s", "2,588,400 s", "1,680 s (28 min)", "by shipment"]),
        ("boundary-01", ["1,800 s       -> fresh", "1,801 s       -> stale_shipment"]),
        ("versions-01", ["non_current_policy", "2,592,001 s", "stale_policy"]),
        ("calls-01", ["{'amount_cents': 2499}", "invalid_tool_arguments", "unknown_tool"]),
        (
            "keys-01",
            ["idempotency_conflict", "request_already_executed", "idempotency_key_required"],
        ),
        ("conv-02", ["[-4 -4] [-4 -4]", "[+4 +4] [+4 +4]", "[1, 2, 3, 2, 1]", "896 parameters"]),
        ("vision-01", ["0.155172", "0.057692", "1,160 flags", "3,120 flags", "180 of them"]),
        ("exercises-01", ["1771.0", "1831.0", "5311.0", "1680 s", "[+8 -1] [+0 +7]", "1,216"]),
    ],
)
def test_committed_outputs_carry_the_worked_values(name: str, expected: list[str]) -> None:
    text = output(name)
    for value in expected:
        assert value in text, (name, value)


def test_failures_table_has_one_audit_and_no_payment_per_row() -> None:
    rows = [line for line in output("failures-01").splitlines() if line.startswith("| `")]
    rows += [line for line in output("failures-01").splitlines() if "malformed" in line]
    reasons = [row.split("|")[4].strip(" `") for row in rows]
    assert reasons == [
        "ambiguous_order",
        "non_current_policy",
        "stale_shipment",
        "stale_policy",
        "suspicious_request",
        "invalid_tool_arguments",
    ]
    assert all(
        row.split("|")[5].strip() == "1" and row.split("|")[6].strip() == "no" for row in rows
    )


def test_keys_pay_once() -> None:
    lines = output("keys-01").splitlines()
    assert lines[0].endswith("refund-r-7-0007") and lines[1].endswith("refund-r-7-0007")
    assert [line for line in lines if line.startswith("payments")] == [
        "payments in the ledger: 1"
    ] * 2


@pytest.mark.parametrize("flagged, defects", [(45, 50), (450, 500), (36, 40)])
def test_slice_interval_inverts_the_binomial_score(flagged: int, defects: int) -> None:
    # A Wald interval fails the score equation at its endpoints.
    from demos.perception_tools_enrichment import parts_per_slice, wilson_interval

    low, high = wilson_interval(flagged, defects)
    estimate = flagged / defects
    for bound in (low, high):
        assert abs(estimate - bound) == pytest.approx(
            1.96 * math.sqrt(bound * (1 - bound) / defects)
        )
    assert low < estimate < high
    assert parts_per_slice(defects, 0.02) == {50: 2500, 500: 25000, 40: 2000}[defects]


@pytest.mark.parametrize("flagged, expected", [(0, (0.0, 0.07135)), (50, (0.92865, 1.0))])
def test_slice_interval_at_zero_or_all_flags(flagged: int, expected: tuple) -> None:
    from demos.perception_tools_enrichment import wilson_interval

    assert wilson_interval(flagged, 50) == pytest.approx(expected, abs=0.00001)


@pytest.mark.parametrize("flagged, defects", [(-1, 50), (51, 50), (0, 0)])
def test_slice_interval_rejects_impossible_counts(flagged: int, defects: int) -> None:
    from demos.perception_tools_enrichment import wilson_interval

    with pytest.raises(ValueError):
        wilson_interval(flagged, defects)


@pytest.mark.parametrize(
    "prediction, expected",
    [
        ((10, 10, 50, 50), 1.0),
        ((20, 20, 60, 60), 9 / 23),
        ((14, 14, 54, 54), 81 / 119),
        ((50, 10, 90, 50), 0.0),
        ((60, 10, 100, 50), 0.0),
        ((10, 10, 30, 30), 0.25),
    ],
)
def test_localization_uses_union_not_the_sum_of_areas(prediction: tuple, expected: float) -> None:
    from demos.perception_tools_enrichment import intersection_over_union

    assert intersection_over_union(prediction, (10, 10, 50, 50)) == pytest.approx(expected)


@pytest.mark.parametrize("offset, expected", [(10, (900, 2300)), (4, (1296, 1904))])
def test_localization_reports_the_actual_overlap_and_union(offset: int, expected: tuple) -> None:
    from demos.perception_tools_enrichment import box_overlap_and_union

    assert box_overlap_and_union(
        (10 + offset, 10 + offset, 50 + offset, 50 + offset), (10, 10, 50, 50)
    ) == expected


def test_page_age_uses_displayed_update_not_capture_time() -> None:
    from datetime import datetime

    from demos.perception_tools_enrichment import locate_element, page_age_seconds

    page = {
        "captured_at": "2026-10-15T10:30:00+00:00",
        "children": [
            {"id": "updated", "text": "2026-10-15T10:00:00+00:00"},
            {"id": "refresh", "box": (10, 10, 50, 50)},
        ],
    }
    used_at = datetime.fromisoformat("2026-10-15T10:30:01+00:00")
    assert page_age_seconds(page, used_at) == 1801
    page["captured_at"] = "2026-10-15T10:30:01+00:00"
    assert page_age_seconds(page, used_at) == 1801
    with pytest.raises(ValueError, match="localization_refused"):
        locate_element(page, "refresh", (20, 20, 60, 60))
    assert locate_element(page, "refresh", (14, 14, 54, 54))["id"] == "refresh"
    with pytest.raises(ValueError, match="element_not_found"):
        locate_element(page, "submit-refund", (14, 14, 54, 54))


@pytest.mark.parametrize("updated", ["2026-10-15T10:31:00+00:00", "2026-10-15T10:00:00"])
def test_page_age_refuses_future_or_unzoned_updates(updated: str) -> None:
    from datetime import datetime

    from demos.perception_tools_enrichment import page_age_seconds

    with pytest.raises(ValueError):
        page_age_seconds(
            {"children": [{"id": "updated", "text": updated}]},
            datetime.fromisoformat("2026-10-15T10:30:01+00:00"),
        )


def test_retry_probabilities_count_silent_retry_errors_and_stops() -> None:
    from demos.perception_tools_enrichment import action_probabilities

    # At p=q=1/2, the first/retry branches have probabilities 1/2, 1/8,
    # 1/4, 1/16, 1/16: correct, corrected, silent, silent retry, stop.
    assert action_probabilities(1, 0.5, 0.5) == pytest.approx(
        {"unchecked_all_right": 0.5, "checked_all_right": 0.625,
         "checked_silent_completed": 0.3125, "checked_stopped": 0.0625,
         "checked_any_silent": 0.3125}
    )
    assert action_probabilities(2, 0.5, 0.5) == pytest.approx(
        {"unchecked_all_right": 0.25, "checked_all_right": 0.390625,
         "checked_silent_completed": 0.48828125, "checked_stopped": 0.12109375,
         "checked_any_silent": 0.5078125}
    )


@pytest.mark.parametrize("p, catch", [(1.0, 0.9), (0.0, 0.0), (0.0, 1.0)])
def test_action_simulation_at_certain_outcomes(p: float, catch: float) -> None:
    from demos.perception_tools_enrichment import action_probabilities, simulate_actions

    counts = simulate_actions(20, p, catch, runs=25, seed=7)
    probabilities = action_probabilities(20, p, catch)
    assert counts == {key: int(value * 25) for key, value in probabilities.items()}


@pytest.mark.parametrize("actions, p, catch", [(20, 0.95, 0.9), (2, 0.5, 0.5)])
def test_seeded_action_simulation_agrees_with_exact_probabilities(
    actions: int, p: float, catch: float,
) -> None:
    from demos.perception_tools_enrichment import action_probabilities, simulate_actions

    counts = simulate_actions(actions, p, catch, runs=10000, seed=7)
    assert counts == simulate_actions(actions, p, catch, runs=10000, seed=7)
    assert sum(counts[key] for key in (
        "checked_all_right", "checked_silent_completed", "checked_stopped"
    )) == 10000
    assert counts["checked_any_silent"] >= counts["checked_silent_completed"]
    for key, probability in action_probabilities(actions, p, catch).items():
        assert counts[key] / 10000 == pytest.approx(probability, abs=0.02)


@pytest.mark.parametrize("name", ["vision-02", "screen-01", "listing-01"])
def test_enrichment_regions_run_independently_and_replay_stored_output(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    region(name)
    assert capsys.readouterr().out == output(name)


def test_generated_listing_retains_the_real_nullable_argument_schema(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    from refund_agent.contracts import ToolArguments

    listing = region("listing-01")
    capsys.readouterr()
    assert json.loads(listing["schema_text"]) == ToolArguments.model_json_schema()
    assert not listing["schema"].get("required")
    assert listing["schema"]["additionalProperties"] is False
    assert "idempotency_key_required" in listing["contract_row"]
    assert "the gate" in listing["contract_row"]


def test_enrichment_exercise_keys_are_run_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # Run the demo's real setup; its CLI artifacts belong in the temporary directory.
    monkeypatch.chdir(tmp_path)
    setup = DEMO.read_text(encoding="utf-8").split("# region book:", 1)[0]
    namespace = {"__file__": str(DEMO)}
    exec(compile(setup, str(DEMO), "exec"), namespace)
    capsys.readouterr()
    keys = region("keys-01", namespace)
    printed = capsys.readouterr().out
    assert len(printed.splitlines()) == 8
    assert printed == output("keys-01")
    assert len(keys["tools"].payments) == 1
    for name in ("conv-01", "conv-02", "vision-01"):
        region(name, namespace)
    capsys.readouterr()
    exercises = region("exercises-01", namespace)
    assert capsys.readouterr().out == output("exercises-01")
    assert exercises["first"].ok and exercises["first"].record_id == "refund-r-7-0007"
    assert exercises["repeat"] == exercises["first"]
    assert not exercises["missing"].ok and exercises["missing"].reason == "idempotency_key_required"
    assert not exercises["conflict"].ok and exercises["conflict"].reason == "idempotency_conflict"
    assert len(exercises["tools"].payments) == 1
    assert not exercises["stale_result"].ok and exercises["stale_result"].reason == "stale_shipment"
    assert len(exercises["stale_tools"].payments) == 0
