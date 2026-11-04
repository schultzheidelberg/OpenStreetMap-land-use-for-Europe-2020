"""Every command must be discoverable and safe to import without ML/DB extras."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(ROOT.glob("[0-9][0-9]_*.py")) + [ROOT / "shp_to_lulc.py"]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_help_does_not_need_datasets_or_optional_services(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert list(tmp_path.iterdir()) == []


def test_imports_do_not_start_jobs_or_load_tensorflow(tmp_path):
    code = """
import runpy
import sys
from pathlib import Path
root = Path(sys.argv[1])
for script in sorted(root.glob('[0-9][0-9]_*.py')) + [root / 'shp_to_lulc.py', root / 'hsi_io.py']:
    runpy.run_path(str(script), run_name='import_check')
assert 'tensorflow' not in sys.modules
assert 'psycopg2' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stdout
    assert list(tmp_path.iterdir()) == []
