import compileall
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_compile():
    assert compileall.compile_dir(str(ROOT), quiet=1, force=True)
