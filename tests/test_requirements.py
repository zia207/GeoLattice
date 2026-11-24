"""requirements.txt is the full runtime dependency list and the installed package uses it."""

import importlib.util
import re
from importlib.metadata import PackageNotFoundError, requires
from pathlib import Path

import pytest

packaging = pytest.importorskip("packaging")
from packaging.requirements import Requirement  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "requirements.txt"


def _load_tool():
    spec = importlib.util.spec_from_file_location("make_requirements", ROOT / "tools" / "make_requirements.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lines(path=REQ):
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.strip().startswith("#")]


def _name(req: str) -> str:
    return Requirement(req).name.lower().replace("_", "-")


def test_every_line_is_a_plain_requirement():
    """setuptools reads this file for [project].dependencies: no -r, -e or pip options allowed."""
    for line in _lines():
        assert not line.startswith("-"), line
        Requirement(line)


def test_one_line_per_package():
    names = [_name(l) for l in _lines()]
    assert len(names) == len(set(names))


def test_up_to_date_with_pyproject():
    """Regenerating from pyproject.toml gives the committed file (python tools/make_requirements.py)."""
    pytest.importorskip("tomllib")
    assert _load_tool().render() == REQ.read_text()


def test_covers_everything_in_option_all():
    pytest.importorskip("tomllib")
    tool = _load_tool()
    have = {_name(l) for l in _lines()}
    for _, req in tool.closure("all", tool._extras()):
        assert _name(req) in have, req
    for req in tool.BASE:
        assert _name(req) in have, req


@pytest.mark.parametrize("pkg", ["cupy-cuda13x", "cupy-cuda12x", "pyspark", "apache-sedona", "dask-cuda",
                                 "dask-jobqueue", "h2o", "tensorflow", "pytest"])
def test_opt_in_groups_stay_out(pkg):
    """GPU, cluster, Spark and dev packages stay opt-in (they need hardware or conflict)."""
    assert pkg not in {_name(l) for l in _lines()}


def test_pyproject_reads_dependencies_from_file():
    text = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'^dynamic = \["dependencies"\]', text, re.M)
    assert 'dependencies = { file = ["requirements.txt"] }' in text


def test_installed_metadata_matches():
    """The installed wheel declares exactly these requirements (so pip installs them)."""
    try:
        reqs = requires("geolattice") or []
    except PackageNotFoundError:
        pytest.skip("geolattice is not installed")
    base = {Requirement(r).name.lower().replace("_", "-") for r in reqs if "extra ==" not in r}
    assert base == {_name(l) for l in _lines()}


def test_helper_files_include_requirements():
    assert _lines(ROOT / "requirements-dev.txt")[0] == "-r requirements.txt"
    gpu = _lines(ROOT / "requirements-gpu.txt")
    assert gpu[0] == "-r requirements.txt"
    assert not any("cuda11x" in l or "cuda12x" in l for l in gpu)


def test_snaphu_skipped_where_it_has_no_wheel():
    """snaphu has no wheel for Python 3.14 and its source fails with GCC 14+."""
    line = next(l for l in _lines() if _name(l) == "snaphu")
    req = Requirement(line)
    assert not req.marker.evaluate({"python_version": "3.14"})
    assert req.marker.evaluate({"python_version": "3.11"})
