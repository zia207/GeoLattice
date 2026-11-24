"""Run every Python example in the user guide's ``geolattice.distributed`` section.

Keeps docs/USER_GUIDE.md honest: each code block there builds its own data and must run
as it stands against the installed package.
"""

import re
import warnings
from pathlib import Path

import pytest

pytest.importorskip("dask")
pytest.importorskip("geopandas")

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "USER_GUIDE.md"


def _blocks():
    text = GUIDE.read_text()
    start = text.index("## Distributed computing (`geolattice.distributed`)")
    end = text.index("\n## ", start + 10)
    section = text[start:end]
    heading = None
    out = []
    for m in re.finditer(r"^### ([^\n]+)$|```python\n(.*?)```", section, flags=re.S | re.M):
        if m.group(1):
            heading = m.group(1).strip()
        else:
            out.append(pytest.param(m.group(2), id=re.sub(r"\W+", "-", heading or "intro").strip("-")))
    return out


@pytest.mark.parametrize("code", _blocks())
def test_user_guide_example(code):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(code, str(GUIDE), "exec"), {"__name__": "user_guide_example"})


def test_examples_found():
    assert len(_blocks()) >= 8
