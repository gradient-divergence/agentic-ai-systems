"""The chapter's listings, frozen comparisons and generated evidence."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from refund_agent.data.generate import Dataset

ROOT = Path(__file__).resolve().parents[1]
COMPANION = ROOT
DEMO = COMPANION / "demos" / "semi_self_supervised_demo.py"
OUTPUTS = COMPANION / "book_outputs"
ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def demo():
    return importlib.import_module("demos.semi_self_supervised_demo")


def test_printed_listings_run_in_chapter_order_on_their_own(tmp_path):
    from demos.regions import extract_snippets_from_text

    names = ("kmeans-01", "selftrain-01", "contrast-01")
    snippets = {
        s.snippet_id: s for s in extract_snippets_from_text(DEMO.read_text(), source_path=DEMO)
    }
    script = tmp_path / "printed.py"
    script.write_text("\n\n".join(snippets[f"fnd.semi-self-supervised.{n}"].content for n in names))
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, env=ENV, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "".join(
        (OUTPUTS / f"_fnd.semi-self-supervised.{n}.txt").read_text() for n in names
    )
    assert list(tmp_path.iterdir()) == [script]  # freezing alone writes no journal


def test_hand_kmeans_and_contrastive_gradient():
    d = demo()
    groups, centers, before, after = d.lloyd_step([1, 2, 8, 9], [1, 8])
    assert groups == [[1.0, 2.0], [8.0, 9.0]]
    assert centers == [1.5, 8.5] and (before, after) == (2, 1)
    for tau in (1, 0.5):
        scores = np.array([2.0, 0.0, 0.0])
        loss, p, gradient = d.contrastive_loss(scores, tau)
        assert loss == pytest.approx(0.2395447662 if tau == 1 else 0.0359762997)
        eps = 1e-6
        numeric = [
            (
                d.contrastive_loss(scores + eps * row, tau)[0]
                - d.contrastive_loss(scores - eps * row, tau)[0]
            )
            / (2 * eps)
            for row in np.eye(3)
        ]
        np.testing.assert_allclose(gradient, numeric, atol=1e-9)
        assert p.sum() == pytest.approx(1)
    for n in (3, 8, 64):
        assert d.contrastive_loss(np.zeros(n))[0] == pytest.approx(np.log(n))


def test_a_group_with_no_cases_keeps_its_center():
    """K-means need not return K groups: three identical points and two centers give one."""
    import warnings

    from demos.regions import extract_snippets_from_text

    listing = next(
        s.content
        for s in extract_snippets_from_text(DEMO.read_text(), source_path=DEMO)
        if s.snippet_id == "fnd.semi-self-supervised.kmeans-01"
    )
    printed = {"__name__": "printed"}  # the listing's function, without its run of four points
    exec(compile(listing, "kmeans-01", "exec"), printed)
    for lloyd_step in (demo().lloyd_step, printed["lloyd_step"]):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # the mean of no cases warns, then gives NaN
            groups, centers, before, after = lloyd_step([5, 5, 5], [5, 9])
            again = lloyd_step([5, 5, 5], centers)
            far = lloyd_step([1, 2, 8, 9], [1, 100])
        assert groups == [[5.0, 5.0, 5.0], []]
        assert centers == [5.0, 9.0] and np.isfinite(centers).all()
        assert (before, after) == (0, 0)
        assert again == (groups, centers, 0, 0)  # a second step is defined and changes nothing
        # A center no case is nearest to stays where it was, and the update still lowers
        # the objective of the group that has cases.
        assert far == ([[1.0, 2.0, 8.0, 9.0], []], [5.0, 100.0], 114, 50)


def test_durable_log_continues_in_a_later_process(tmp_path):
    code = """
from demos.supervised_learning_demo import frozen, windows
from refund_agent.data.history import generate_history
g = frozen(windows([r for r in generate_history(rows=400) if r.paid])['test'], 'august')
g.persist({directory!r})
g.consult('reader', 'fixed comparison')
print(g.log[-1]['entry'])
""".format(directory=str(tmp_path))
    for entry in (1, 2):
        run = subprocess.run([sys.executable, "-c", code], env=ENV, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        assert run.stdout == f"{entry}\n"
    manifests = list(tmp_path.glob("*.manifest.json"))
    logs = list(tmp_path.glob("*.access.jsonl"))
    assert len(manifests) == len(logs) == 1
    manifest = json.loads(manifests[0].read_text())
    entries = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert [e["entry"] for e in entries] == [1, 2]
    assert {e["sha256"] for e in entries} == {manifest["sha256"]}


@pytest.fixture(scope="module")
def captured():
    from demos.regions import capture_outputs, extract_snippets_from_text

    snippets = {
        s.snippet_id: s for s in extract_snippets_from_text(DEMO.read_text(), source_path=DEMO)
    }
    return capture_outputs(snippets, sys.executable)


def test_every_output_regenerates_byte_for_byte(captured):
    assert len(captured) == 15
    for name, text in captured.items():
        assert text.encode() == (OUTPUTS / f"_{name}.txt").read_bytes(), name


def test_determinism_under_the_seed(captured):
    from demos.regions import capture_outputs, extract_snippets_from_text

    snippets = {
        s.snippet_id: s for s in extract_snippets_from_text(DEMO.read_text(), source_path=DEMO)
    }
    assert capture_outputs(snippets, sys.executable) == captured


def test_chapter_four_and_five_outputs_are_unchanged():
    from demos.regions import capture_outputs, extract_snippets_from_text

    snippets = {}
    for name in ("ml_data_evaluation_demo.py", "supervised_learning_demo.py"):
        source = COMPANION / "demos" / name
        snippets.update(
            {
                s.snippet_id: s
                for s in extract_snippets_from_text(source.read_text(), source_path=source)
            }
        )
    outputs = capture_outputs(snippets, sys.executable)
    assert len(outputs) == 30  # includes the complete trace, its two excerpts and bootstrap
    assert "fnd.ml-data-and-evaluation.trace-01" in outputs
    for name, text in outputs.items():
        assert text.encode() == (OUTPUTS / f"_{name}.txt").read_bytes(), name


def test_import_runs_nothing(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import demos.semi_self_supervised_demo"],
        cwd=tmp_path,
        env=ENV,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "" and list(tmp_path.iterdir()) == []


def test_inputs_never_read_unlabeled_outcomes_and_pools_are_time_safe():
    d = demo()
    window, pools, august, september, counts = d.prepare()
    assert (len(window["training"]), len(window["validation"])) == (13766, 988)
    assert counts == {
        "history": 36000,
        "never_paid": 18296,
        "refused_before": 16254,
        "refused_above": 6355,
    }
    assert [len(p) for p in pools.values()] == [985, 9899]
    assert august.manifest["sha256"].startswith("a7a7cf68978cc0c8")
    assert august.log == september.log == []
    assert august.intact() and september.intact()
    forbidden = {r.request_id for g in (august, september) for r in g.requests}
    for rows in [*window.values(), *pools.values()]:
        assert all(r.at < d.GO_LIVE and r.request_id not in forbidden for r in rows)
    assert all(r.label_final_at > d.GO_LIVE for r in pools["open"])
    assert all(not r.paid and r.amount_cents < 5000 for r in pools["refused tier 1"])

    class InputsOnly:
        def __init__(self, row):
            self.row = row

        def __getattr__(self, name):
            if name in ("found_false_30d", "label_final_at"):
                raise AssertionError(f"unlabeled outcome read: {name}")
            return getattr(self.row, name)

    for rows in pools.values():
        np.testing.assert_array_equal(d.inputs([InputsOnly(r) for r in rows]), d.inputs(rows))


def test_fits_and_feature_reads_precede_the_frozen_consultations(tmp_path):
    # Instrument actual fitting and feature extraction, not just the source order.
    code = f"""
import contextlib, io, json, runpy
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from refund_agent.split import GoldenSet
import refund_agent.model.history_baseline as features
from demos.supervised_learning_demo import GO_LIVE
state = {{'reads': [], 'fits': 0}}
for cls in (KMeans, HistGradientBoostingClassifier, KNeighborsClassifier, StandardScaler):
    original = cls.fit
    def fit(self, *args, _original=original, **kwargs):
        assert not state['reads'], 'fit after a frozen consultation'
        state['fits'] += 1
        return _original(self, *args, **kwargs)
    cls.fit = fit
original_features = features.feature_matrix
def matrix(rows):
    if any(r.at >= GO_LIVE for r in rows):
        assert state['reads'], 'held-out input used before consultation'
    return original_features(rows)
features.feature_matrix = matrix
import demos.supervised_learning_demo as supervised_demo
supervised_demo.feature_matrix = matrix
original_frozen = supervised_demo.frozen
class GuardedRows:
    def __init__(self, rows, name):
        self.rows, self.name = rows, name
    def __iter__(self):
        assert self.name in state['reads'], 'frozen rows or labels read without consultation'
        return iter(self.rows)
def frozen(rows, name):
    golden = original_frozen(rows, name)
    golden.requests = GuardedRows(golden.requests, name)
    golden.labels = GuardedRows(golden.labels, name)
    return golden
supervised_demo.frozen = frozen
original_consult = GoldenSet.consult
def consult(self, *args):
    state['reads'].append(self.manifest['name'])
    return original_consult(self, *args)
GoldenSet.consult = consult
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path({str(DEMO)!r}, run_name='__main__')
assert state['reads'] == ['history-august', 'history-september']
assert state['fits'] > 200
assert set(ns['window']) == {{'training', 'validation', 'open'}}
assert all(len(ns[name].log) == 2 for name in ('august', 'september'))
print(json.dumps(state))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**ENV, "BOOK_GOLDEN_DIRECTORY": str(tmp_path / "journal")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["fits"] > 200


def test_unit_scalers_and_centers_use_the_declared_inputs():
    d = demo()
    window, _, _, _, _ = d.prepare()
    two, scaler, raw, scaled, stability, figure = d.unit_partitions(window)
    np.testing.assert_allclose(scaler.mean_, d.inputs(window["training"])[:, [0, 3]].mean(axis=0))
    assert len(two) == 15739 and len(figure["points"]) == 1000
    assert raw.n_init == scaled.n_init == 10 and raw.n_clusters == scaled.n_clusters == 4
    assert raw.random_state == scaled.random_state == 7
    assert stability > 0.99
    # The plotted coordinates use dollars; only the fit uses cents.
    assert max(p[0] for p in figure["points"]) < 50


def test_choices_listing_is_independent_and_fits_the_page(tmp_path):
    from demos.regions import extract_snippets_from_text

    snippets = {
        s.snippet_id: s for s in extract_snippets_from_text(DEMO.read_text(), source_path=DEMO)
    }
    name = "fnd.semi-self-supervised.choose-01"
    assert name in snippets, "the clustering-choices listing is missing"
    listing = snippets[name].content
    assert len(listing.splitlines()) <= 15
    assert max(map(len, listing.splitlines())) <= 79
    script = tmp_path / "choices.py"
    script.write_text(listing)
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, env=ENV, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) <= 20
    lines = result.stdout.splitlines()
    assert lines[:5] == [
        "silhouettes: [0.867, 0.846, 0.846, 0.867]; mean 0.856",
        "silhouettes: [0.857, 0.833, 0.0, 0.0]; mean 0.423",
        "Pr(first | x=4, x=5): [0.852, 0.5]",
        "density radius 1.5, min count 2: [[1, 2], [8, 9]]; noise []",
        "density radius 1.5, min count 3: []; noise [1, 2, 8, 9]",
    ]
    assert result.stdout == (OUTPUTS / f"_{name}.txt").read_text()
    assert list(tmp_path.iterdir()) == [script]  # no frozen-set journal or other side effect


def test_refund_choices_use_plot_rows_and_full_partitions_without_outcomes(monkeypatch):
    d = demo()
    assert hasattr(d, "refund_choices"), "the refund-plane K sweep is missing"
    history = d.generate_history(seed=7)
    before = [r for r in history if r.paid and r.at < d.GO_LIVE]
    window = d.windows(before)
    two, scaler, _, _, stability, _ = d.unit_partitions(window)
    plane = scaler.transform(two)
    pick = np.linspace(0, len(plane) - 1, 1000, dtype=int)
    calls = []
    silhouette = d.silhouette_score

    def checked_silhouette(X, labels):
        np.testing.assert_array_equal(X, plane[pick])
        assert len(labels) == 1000
        calls.append(len(X))
        return silhouette(X, labels)

    class InputsOnly:
        def __init__(self, row):
            self.row = row

        def __getattr__(self, name):
            if name == "found_false_30d":
                raise AssertionError("the clustering sweep read an outcome")
            if self.row.at >= d.GO_LIVE and name not in {"at", "paid"}:
                raise AssertionError("the clustering sweep read a frozen-month input")
            return getattr(self.row, name)

    def no_consultation(*args, **kwargs):
        raise AssertionError("the clustering sweep froze or consulted a month")

    monkeypatch.setattr(d, "generate_history", lambda seed: [InputsOnly(r) for r in history])
    monkeypatch.setattr(d, "frozen", no_consultation)
    monkeypatch.setattr(d, "consulted", no_consultation)
    monkeypatch.setattr(d, "silhouette_score", checked_silhouette)
    report = d.refund_choices()
    assert report["fit_rows"] == len(two) == 15739 and report["plot_rows"] == 1000
    assert calls == [1000] * 7
    rows = report["choices"]
    assert [r["K"] for r in rows] == list(range(2, 9))
    plotted = plane[pick]
    distances = np.linalg.norm(plotted[:, None, :] - plotted[None, :, :], axis=2)
    for row in rows:
        k = row["K"]
        first = d.KMeans(n_clusters=k, n_init=10, random_state=7).fit(plane)
        second = d.KMeans(n_clusters=k, n_init=10, random_state=17).fit(plane)
        assert row["objective"] == pytest.approx(first.inertia_)
        assert row["adjusted_rand"] == pytest.approx(
            d.adjusted_rand_score(first.labels_, second.labels_)
        )
        # Compute silhouettes independently from distances and the expected memberships.
        labels = first.labels_[pick]
        scores = []
        for i, group in enumerate(labels):
            own = labels == group
            own[i] = False
            if not own.any():
                scores.append(0.0)
                continue
            within = distances[i, own].mean()
            other = min(
                distances[i, labels == g].mean() for g in np.unique(labels) if g != group
            )
            scores.append((other - within) / max(within, other))
        assert row["silhouette"] == pytest.approx(np.mean(scores))
    assert rows[2]["adjusted_rand"] == pytest.approx(stability)
    assert f"{rows[2]['adjusted_rand']:.4f}" == "0.9968"


def test_self_training_retains_each_pseudo_label_and_never_adds_a_case_twice():
    d = demo()
    fits = []

    class Constant:
        classes_ = np.array([0, 1])

        def fit(self, X, y):
            fits.append((X.copy(), y.copy()))
            return self

        def predict_proba(self, X):
            return np.tile([0.95, 0.05], (len(X), 1))

    X = np.arange(4).reshape(-1, 1)
    pool = np.arange(10, 20).reshape(-1, 1)
    _, rows = d.self_train(X, np.array([0, 0, 0, 1]), pool, Constant)
    assert len(rows) == 1 and rows[0]["accepted"] == rows[0]["total"] == 10
    assert rows[0]["by_class"] == {0: 10, 1: 0}
    assert len(fits) == 2
    np.testing.assert_array_equal(fits[1][0], np.concatenate([X, pool]))
    np.testing.assert_array_equal(fits[1][1], [0, 0, 0, 1] + [0] * 10)
    # Rejected inputs produce no additional fit, including an exact confidence boundary.
    fits.clear()
    d.self_train(X, np.array([0, 0, 0, 1]), pool, Constant, confidence=0.99)
    assert len(fits) == 1


def test_ticket_holdout_is_random_disjoint_and_frozen_and_labels_are_matched():
    d = demo()
    X, y, Xh, yh = d.tickets()
    assert len(X) == 3000 and len(Xh) == 1000
    assert not Xh.flags.writeable and not yh.flags.writeable
    assert not set(map(tuple, X)) & set(map(tuple, Xh))
    np.testing.assert_array_equal(d.tickets()[2], Xh)
    for budget in (6, 12, 24, 48, 96):
        scores, _, run = d.ticket_run(7, budget, (X, y, Xh, yh))
        chosen, unlabeled, central = run[4:]
        assert len(chosen) == budget and len(unlabeled) + budget == len(X)
        assert not set(chosen) & set(unlabeled)
        assert list(scores) == [
            "random labels",
            "self-trained",
            "center labels",
            "center labels, spread",
        ]
        # Each pair shares its labels: the random ones with and without self-training,
        # and one label for the ticket nearest each center with and without spreading.
        assert len(set(central)) == budget
        alone = d.ticket_classifier().fit(X[central], y[central])
        assert scores["center labels"] == float((alone.predict(Xh) == yh).mean())
        drawn = d.ticket_classifier().fit(X[chosen], y[chosen])
        assert scores["random labels"] == float((drawn.predict(Xh) == yh).mean())


def test_fixture_has_one_evaluation_defect_and_an_honest_rerun():
    d = demo()
    fixture = json.loads(d.FIXTURE.read_text())
    report, honest = d.contaminated_report()
    assert fixture == report
    holdout = fixture["holdout"]
    assert holdout["source"] == "unlabeled pool" and holdout["cut_after_round"] == 1
    assert holdout["outcome_labels"] + holdout["pseudo_labels"] == holdout["requests"] == 1000
    assert holdout["pseudo_label_source"] == "the model being judged in each round"
    assert [row["round"] for row in fixture["rounds"]] == [1, 2, 3]
    for row in fixture["rounds"]:
        assert sum(row["accepted_by_queue"]) == row["accepted"]
    contaminated = [row["holdout_accuracy"] for row in fixture["rounds"]]
    assert contaminated[0] < contaminated[1] < contaminated[2]
    assert all(bad > good for bad, good in zip(contaminated, honest))
    assert honest == [0.611, 0.616, 0.617]


def test_figure_data_and_headline_differences_are_the_captured_results(captured):
    figures = json.loads(captured["fnd.semi-self-supervised.figures-01"])
    budget = figures["budget"]
    assert [r["labels"] for r in budget["tickets"]] == [6, 12, 24, 48, 96]
    assert [r["labels"] for r in budget["refunds"]] == [300, 1000, 3000]
    assert budget["draws"] == 20 and budget["refund_validation"] == 988
    for row in budget["refunds"]:
        assert row["self-trained"]["log_loss"][0] > row["supervised"]["log_loss"][0]
    assert figures["contrastive"]["loss"][20] == pytest.approx(np.log(3), abs=1e-6)
    assert figures["contrastive"]["loss"][40] == pytest.approx(0.2395447662, abs=1e-6)
    holdout = captured["fnd.semi-self-supervised.holdout-01"]
    assert "August, milliseconds | 999/999 (1.0000)" in holdout
    assert "August, seconds | 27/999 (0.0270) | 0" in holdout
    assert "September, seconds | 15/966 (0.0155) | 0" in holdout


def test_durable_slices_share_the_parent_journal_and_resumption_does_not_replay(tmp_path):
    from demos.supervised_learning_demo import frozen, windows
    from refund_agent.data.history import generate_history
    from refund_agent.split import freeze

    d = importlib.import_module("demos.supervised_learning_demo")
    rows = windows([r for r in generate_history(rows=400) if r.paid])["test"]
    parent = frozen(rows, "parent")
    parent.persist(tmp_path)
    parent.consult("reader", "first")
    child = freeze(
        cast(Dataset, d.Ledger(rows)), rows[:2], d.OUTCOME_POLICY, name="slice", parent=parent
    )
    child.consult("reader", "slice")
    assert child.log is parent.log and child.log_path == parent.log_path
    assert [e["entry"] for e in parent.log] == [1, 2]
    before = parent.log_path.read_bytes()
    resumed = frozen(rows, "parent")
    resumed.persist(tmp_path, prior_log=[parent.log[0]])
    assert resumed.log == parent.log and parent.log_path.read_bytes() == before
    resumed.consult("reader", "later")
    assert resumed.log[-1]["entry"] == 3
    assert parent.log_path.read_bytes().startswith(before)


def test_durable_log_rejects_corruption_without_overwriting_it(tmp_path):
    from demos.supervised_learning_demo import frozen, windows
    from refund_agent.data.history import generate_history

    rows = windows([r for r in generate_history(rows=400) if r.paid])["test"]
    g = frozen(rows, "august")
    g.persist(tmp_path)
    g.consult("reader", "first")
    with g.log_path.open("a") as stream:
        stream.write('{"sha256":"wrong", "entry":2}\n')
    corrupted = g.log_path.read_bytes()
    with pytest.raises(ValueError, match="invalid durable"):
        g.consult("reader", "second")
    assert g.log_path.read_bytes() == corrupted and len(g.log) == 1


def test_page_width_regions_repeat_the_full_record(captured):
    """record-01, groups-01, flags-01 and scores-01 print holdout-01 again at page width."""
    import re

    d = demo()
    out = {name.rsplit(".", 1)[1]: text for name, text in captured.items()}
    full = out["holdout-01"]

    # The record: August's fingerprint, and both log entries once the wrapping is undone.
    record = out["record-01"]
    assert "name: history-august\nrequests: 999\nsha256: a7a7cf68978cc0c8\n" in record
    assert max(map(len, record.splitlines())) <= 66
    entries = re.split(r"\n  (?=\d+: )", record.split("access log, every entry:\n")[1].rstrip())
    assert len(entries) == 2
    for entry in entries:
        assert f"  {' '.join(entry.split())}\n" in full

    # The groups: the same counts, centers and rates, the rates to three places.
    separation = out["units-01"].split("center separation, standardized range: ")[1]
    assert {part.split("=")[0] for part in separation.split(", ")[:4]} == set(d.SEPARATING)
    wide = re.findall(
        r"^\| (\d) \| (\d+) \| (.+?) \| (\d+) \| ([\d.]+) \(([\d.]+) to ([\d.]+)\) \|$", full, re.M
    )
    page = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in out["groups-01"].splitlines()
    ]
    assert all(len(line) <= 79 for line in out["groups-01"].splitlines())
    assert page[0] == ["Measure", "Group 1", "Group 2", "Group 3", "Group 4"]
    column = {row[0]: row[1:] for row in page[2:]}  # one column a group, one row a measure
    assert list(column) == [*d.SEPARATING.values(), "Wrong payments", "Rate", "95% lower", "Upper"]
    assert len(wide) == 4
    for j, (k, n, center, wrong, rate, low, high) in enumerate(wide):
        assert k == str(j + 1) and column["Wrong payments"][j] == f"{wrong} of {n}"
        for name, label in d.SEPARATING.items():
            assert f"{name}={column[label][j]}" in center
        low_page, high_page = column["95% lower"][j], column["Upper"][j]
        for exact, shown in ((rate, column["Rate"][j]), (low, low_page), (high, high_page)):
            assert abs(float(exact) - float(shown)) <= 0.00051

    # The flags: the same three rows.
    wide = re.findall(
        r"^\| (.+?) \| (\d+)/(\d+) \([\d.]+\) \| (\d+) \(([\d.]+), ([\d.]+) to ([\d.]+)\) "
        r"\| (\d+)/\d+ \([\d.]+\) \|$",
        full,
        re.M,
    )
    page = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in out["flags-01"].splitlines()[2:]
    ]
    assert len(wide) == len(page) == 3
    for (name, n, total, wrong, rate, low, high, all_wrong), row in zip(wide, page):
        assert row[:3] == [name, f"{n} of {total}", f"{wrong} of {all_wrong}"]
        low_page, high_page = row[4].split(" to ")
        for exact, shown in ((rate, row[3]), (low, low_page), (high, high_page)):
            assert abs(float(exact) - float(shown)) <= 0.00051

    # The scores: every cell of the transposed table is a cell of the full one.
    rows = [
        [cell.strip().replace(d.MINUS, "-") for cell in line.strip("|").split("|")]
        for line in out["scores-01"].splitlines()
    ]
    assert rows[0] == ["Measure", "Supervised", "Open pool", "Refused pool"]
    assert list(d.prepare()[1]) == ["open", "refused tier 1"]  # the two pools, in column order
    cells, last = {}, None
    for label, *values in rows[2:]:
        cells[f"{last} change" if label == "change" else label] = values
        last = label if label != "change" else last
    for column, name in enumerate(("supervised", "open", "refused tier 1")):

        def cell(measure):
            value = cells[measure][column]
            change = cells.get(f"{measure} change", [""] * 3)[column]
            return value, change

        mean, loss, cost = cell("Mean prediction"), cell("Log loss"), cell("Cost per request")
        if column:
            assert f"| {name} | {mean[0]} ({mean[1]}) | {loss[0]} ({loss[1]}) | " in full
            assert f" | {cost[0]} ({cost[1]}) | {cells['Training loss'][column]} |" in full
        else:
            assert mean[1] == loss[1] == cost[1] == ""
            assert f"| {name} | {mean[0]} (0.0000 " in full
            assert (
                f" | {cost[0]} (0.0000 (0.0000 to 0.0000)) | {cells['Training loss'][0]} |" in full
            )
        line = next(row for row in full.splitlines() if row.startswith(f"| {name} |"))
        referred, paid = (part.split(" (")[0].strip() for part in line.split("|")[4:6])
        assert [cells["Referred"][column], cells["Wrong payments paid"][column]] == [referred, paid]
    rounds = out["selftrain-01"]
    assert f"validation loss {cells['Validation loss'][0]}\n" in rounds
    finals = re.findall(r"^\s+3\s.*\s([\d.]+)$", rounds, re.M)  # each pool's last round
    assert cells["Validation loss"][1:] == finals
