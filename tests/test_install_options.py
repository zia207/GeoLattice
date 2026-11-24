"""Installation options (geolattice[core] ... geolattice[all]) and install hints."""

import pytest

from geolattice import install_options as io_
from geolattice.install_options import INSTALL_OPTIONS, install_hint, option_for


def test_options_are_cumulative():
    names = [n for n, _ in INSTALL_OPTIONS]
    assert names == ["core", "analysis", "statistics", "geostats", "remote-sensing", "all"]
    for smaller, larger in zip(names, names[1:]):
        assert io_._closure(smaller) <= io_._closure(larger)


@pytest.mark.parametrize("extra, option", [
    ("io", "core"),
    ("viz", "core"),
    ("network", "analysis"),
    ("whitebox", "analysis"),
    ("spatial-statistics", "statistics"),
    ("spatialcausal", "statistics"),
    ("geostatistics-plot", "geostats"),
    ("rsense-lidar", "remote-sensing"),
    ("rsense-dl", "all"),
    ("web", "all"),
    ("spatial-db-postgis", "all"),
    ("distributed", "all"),
    ("distributed-ml", "all"),
])
def test_smallest_option(extra, option):
    assert option_for(extra) == option


@pytest.mark.parametrize("extra", ["geostatistics-gpu", "rsense-gpu", "rsense-biopar", "geostatistics-h2o", "dev",
                                   "distributed-gpu", "distributed-hpc", "distributed-spark"])
def test_extras_left_out_of_all(extra):
    assert option_for(extra) is None
    assert "not part of" in install_hint(extra)


def test_hint_text():
    assert install_hint("geostatistics-plot") == (
        'pip install "geolattice[geostatistics-plot]" (included in installation option 4, "geolattice[geostats]")'
    )
    assert install_hint("all") == 'pip install "geolattice[all]"'


def test_metadata_matches_fallback():
    """The fallback table used without package metadata must agree with pyproject.toml."""
    graph = io_._extra_includes()
    for extra, includes in io_._FALLBACK.items():
        assert graph.get(extra) == includes, extra
