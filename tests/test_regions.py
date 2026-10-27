"""Read marked code and capture selected outputs without repository helpers."""

from pathlib import Path

import pytest

from demos.regions import capture_outputs, extract_snippets_from_text


def test_reader_dedents_and_removes_only_comment_callouts():
    source = "# region book:sample output\n    text = '# (9)'\n    print(text)  # (1)\n# endregion book:sample\n"
    (region,) = extract_snippets_from_text(source, Path("demo.py"))
    assert region.content == "text = '# (9)'\nprint(text)\n"
    assert region.callouts == {2: 1}
    assert (region.start_line, region.end_line, region.output) == (2, 3, True)


@pytest.mark.parametrize(
    "source",
    [
        "# region book:a\npass\n# endregion book:b\n",
        "# region book:a\npass\n",
        "# region book:a\npass  # (2)\n# endregion\n",
    ],
)
def test_reader_rejects_invalid_regions(source):
    with pytest.raises(ValueError):
        extract_snippets_from_text(source, Path("demo.py"))


def test_capture_keeps_setup_and_stops_after_selected_region(tmp_path):
    import sys

    path = tmp_path / "demo.py"
    path.write_text(
        "value = 41\n# region book:answer output\nprint(value + 1)\n# endregion\nraise RuntimeError('must not run')\n"
    )
    regions = {r.snippet_id: r for r in extract_snippets_from_text(path.read_text(), path)}
    assert capture_outputs(regions, sys.executable) == {"answer": "42\n"}


@pytest.mark.parametrize(
    "name",
    [
        "agent_evaluation_capture",
        "batch_stream_capture",
        "cloud_native_deployment_capture",
        "capstone_blueprint_capture",
        "human_oversight_risk_capture",
        "audits_explanations_capture",
        "case_studies_domain_agnostic_capture",
        "org_adoption_demo",
    ],
)
def test_capture_helpers_read_only_explicit_companion_sources(name, tmp_path, monkeypatch):
    from importlib import import_module

    from demos import regions

    helper = import_module("demos." + name)
    root = Path(__file__).resolve().parents[1]
    assert helper.ROOT == root
    seen = []

    def capture(snippets, *args, **kwargs):
        for region in snippets.values():
            assert region.source_path.is_relative_to(root)
            assert region.source_path.is_file()
            seen.append(region.snippet_id)
        return {key: "captured fixture\n" for key, region in snippets.items() if region.output}

    monkeypatch.setattr(regions, "capture_outputs", capture)
    kwargs = {}
    if name == "capstone_blueprint_capture":
        kwargs["run_dir"] = tmp_path / "run"
    if name == "case_studies_domain_agnostic_capture":
        kwargs["ids"] = helper.INDEPENDENT
    run = helper.capture_org_adoption if name == "org_adoption_demo" else helper.capture_chapter
    assert run(tmp_path / "outputs", **kwargs)
    assert seen
