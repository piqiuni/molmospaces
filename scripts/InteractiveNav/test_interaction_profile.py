import json

import pytest

from scripts.InteractiveNav.evaluation.interaction_profile import profile_interaction


@pytest.mark.parametrize("fail", [False, True])
def test_profile_preserves_result_or_exception_and_writes_partial_data(tmp_path, fail):
    def run(timing_sink):
        timing_sink(0, "open", "before_step", 0.02)
        timing_sink(0, "open", "after_step", 0.01)
        if fail:
            raise ValueError("simulation error")
        return {"success": True}

    if fail:
        with pytest.raises(ValueError, match="simulation error"):
            profile_interaction(run, tmp_path)
    else:
        assert profile_interaction(run, tmp_path) == {"success": True}
    assert (tmp_path / "calls.pstats").stat().st_size > 0
    assert json.loads((tmp_path / "calls.json").read_text())
    rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert [r["operation"] for r in rows] == ["before_step", "after_step"]
