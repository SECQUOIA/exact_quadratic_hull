import shutil

import pytest

from exact_hull.experiment.runner import run


@pytest.mark.gams
@pytest.mark.skipif(shutil.which("gams") is None, reason="GAMS is not installed")
def test_smoke_run_with_gams(tmp_path):
    from pathlib import Path

    root = Path(__file__).parents[1]
    records = run(root / "configs" / "smoke.toml", tmp_path)
    assert len(records) == 1
    assert (tmp_path / "jobs" / records[0].run_id / "result.json").exists()
