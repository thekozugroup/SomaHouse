import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

from somahouse.brain import ROOT


class References(HTMLParser):
    def __init__(self):
        super().__init__(); self.refs = []

    def handle_starttag(self, tag, attrs):
        for key, val in attrs:
            if key in ("src", "href"):
                self.refs.append(val)


def test_portable_case_study_has_no_missing_local_references():
    page = ROOT / "docs/index.html"
    if not page.exists():
        return
    parser = References(); parser.feed(page.read_text())
    for ref in parser.refs:
        if not ref.startswith(("http:", "https:", "#")):
            assert not ref.startswith("/"), ref
            assert (page.parent / ref).is_file(), ref


def test_saved_artifact_manifest_matches_every_file():
    path = ROOT / "reports/manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    assert len(manifest) >= 100
    for filename, info in manifest.items():
        p = ROOT / filename
        assert p.is_file(), filename
        assert p.stat().st_size == info["bytes"], filename
        assert hashlib.sha256(p.read_bytes()).hexdigest() == info["sha256"], filename


def test_surgery_claim_matches_frozen_validation_and_held_out_interval():
    path = ROOT / "reports/results.json"
    if not path.exists():
        return
    result = json.loads(path.read_text())
    selection = json.loads((ROOT / "reports/selection.json").read_text())
    assert result["primary"] == selection["primary_family"]
    assert result["selection_changed_after_test"] is False
    assert selection["duplicate_improves_all_validation_seeds"]
    comparison = next(c for c in result["comparisons"] if c["a"] == "duplicate" and c["b"] == "continue")
    assert comparison["ci95"][1] < 0


def test_preview_identifies_real_checkpoint_and_exact_recorded_test():
    path = ROOT / "reports/preview.json"
    if not path.exists():
        return
    meta = json.loads(path.read_text())
    assert meta["verified_decisions"] == 288
    assert meta["maximum_action_temperature_replay_error"] < 1e-6
    assert meta["simulated_hours"] * 3600 / meta["duration_seconds"] == meta["simulation_speed"] == 2400
    assert hashlib.sha256((ROOT / meta["checkpoint"]).read_bytes()).hexdigest() == meta["checkpoint_sha256"]
    assert hashlib.sha256((ROOT / "somahouse/static/preview.mp4").read_bytes()).hexdigest() == meta["video_sha256"]
    assert meta["online_learning"] is False


def test_teacher_superiority_claim_requires_every_registered_gate():
    path = ROOT / "reports/teacher/results.json"
    if not path.exists():
        return
    result = json.loads(path.read_text())
    selection = json.loads((ROOT / "reports/teacher/selection.json").read_text())
    assert selection["selected_before_test"] and not result["selection_changed_after_test"]
    assert result["outperforms_teacher"] == all(result["gates"].values())
    assert len(selection["test_seeds"]) == 100
    assert len(selection["runs"]) == 15
    assert hashlib.sha256((ROOT / "reports/teacher/selection.json").read_bytes()).hexdigest() == result["selection_sha256"]
