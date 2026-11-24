# Contributing to GeoLattice

Thank you for helping improve GeoLattice. This page explains how to set up a development
environment, where things live, how the tests work and what a change needs before it is
merged.

## Development setup

GeoLattice needs Python 3.10 or newer, a C compiler (for the suitability C extension) and GDAL
with Python bindings that match the GDAL library. Install GDAL and PROJ with conda or your
system package manager first; pip cannot install the GDAL library itself.

```bash
git clone <your fork of the GeoLattice repository>
cd GeoLattice
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python3 -m pip install -e ".[dev]"     # every runtime dependency plus pytest, black, flake8, mypy, sphinx
```

An editable install picks up changes in `src/` immediately. The installed package is called
`geolattice`, but the source lives in `src/` (`setup.py` maps `src/` onto the `geolattice` name), so
`geolattice/raster` in an import is `src/raster` on disk.

If GDAL lives in a conda-style prefix that you use without `conda activate`, export
`PROJ_DATA=<prefix>/share/proj` so GDAL finds its `proj.db`. See the
[user guide's troubleshooting sections](docs/USER_GUIDE.md#troubleshooting-a-gdal-environment-used-without-conda-activate).

## Where things live

| Path | Contents |
|---|---|
| `src/` | The `geolattice` package: `vector`, `raster`, `spatial_analysis`, `spatial_statistics`, `geostatistics`, `rsense`, `web`, `spatial_bd`, `distributed`, `viz` and the rest |
| `tests/` | The pytest suite, one file per module (`test_distributed.py`, `test_geostatistics.py`, ...) |
| `docs/USER_GUIDE.md` | Features, installation, troubleshooting and worked examples |
| `docs/API_REFERENCE.md` | Generated list of every public class and function |
| `Tutorials/notebooks/` | Tutorial notebooks by chapter |
| `requirements.txt` | Every runtime dependency; `pyproject.toml` reads `[project].dependencies` from it |
| `tools/make_requirements.py` | Regenerates `requirements.txt` from the extras in `pyproject.toml` |
| `CHANGELOG.md` | Changes in each release |

## Code style

Black formats the code (line length 100, set in `pyproject.toml`), flake8 checks it and
mypy checks types:

```bash
black src tests
flake8 src tests --max-line-length 100
mypy src
```

Match the code around your change: its naming, docstring style (NumPy-style sections or
short reStructuredText, as the module already uses) and comment density. Import optional
dependencies inside the functions that need them, and raise an `ImportError` that names the
extra to install; `geolattice.install_options.install_hint("rsense-lidar")` writes that message.

## Testing

```bash
python3 -m pytest                       # whole suite, with coverage (see [tool.pytest.ini_options])
python3 -m pytest tests/test_raster.py  # one file
python3 -m pytest -k "sjoin and gpu"    # by name
python3 -m pytest -o addopts=""         # without the coverage report
```

The tests import the installed `geolattice`. With an editable install that is your working copy;
with a wheel install, rebuild and reinstall after changing `src/` (see below), or the tests
run against the old code.

Tests that need a GPU, Java, a cluster scheduler or a large optional package skip themselves
when it is missing (`pytest.importorskip`, `pytest.mark.skipif`), so the suite passes on a
plain CPU machine. When you add a GPU code path, also test it against the CPU result;
`geolattice.distributed.check_equivalence` compares the two.

New functions need tests. Where a reference implementation exists (GeoPandas, rasterio,
scipy, libpysal, esda, NetworkX, scikit-learn), compare against it rather than against
hand-written expected numbers.

## Dependencies

Runtime dependencies are declared as extras in `pyproject.toml` (`[project.optional-dependencies]`).
`requirements.txt` is generated from them and is what a plain install of GeoLattice installs.
After adding or changing an extra:

```bash
python3 tools/make_requirements.py          # rewrite requirements.txt
python3 tools/make_requirements.py --check  # exit 1 if it is out of date
```

`tests/test_requirements.py` fails until the two agree. Packages that need CUDA, a cluster,
Java, or that pin versions conflicting with the rest stay out of `geolattice[all]`, and therefore
out of `requirements.txt`; give them their own extra and list it in
`src/install_options.py` and the user guide.

## Building

```bash
python3 -m pip install build
python3 -m build                        # dist/geolattice-<version>-cp<python>-*.whl and dist/geolattice-<version>.tar.gz
python3 -m pip install dist/geolattice-<version>-cp311-cp311-linux_x86_64.whl
```

The wheel contains a C extension, so it only works with the Python version that built it.
For another version, build from the sdist with that Python:
`python3 -m pip wheel --no-deps -w dist dist/geolattice-<version>.tar.gz`.

## Documentation

- Update `docs/USER_GUIDE.md` when you add a feature or change how something is used. The
  examples in its `geolattice.distributed` section run as tests (`tests/test_distributed_docs.py`).
- Regenerate the API reference from the installed package after changing the public API:
  `python3 docs/make_api_reference.py`.
- The Sphinx site builds with `python3 -m sphinx -b html docs/source docs/build/html`.
- Add a line to the top entry of `CHANGELOG.md` describing the change.

## Releasing

1. Bump the version in `pyproject.toml`, `src/__init__.py`, `src/rsense/_version.py`,
   `src/geostatistics/__init__.py` and `src/spatial_statistics/__init__.py`, and in the wheel
   file names in `README.md` and `docs/USER_GUIDE.md`.
2. Date the top `CHANGELOG.md` entry.
3. Rebuild the wheel and sdist, reinstall, and run the whole test suite.
4. Regenerate `docs/API_REFERENCE.md`.

## Submitting changes

1. Create a branch: `git checkout -b feature/your-feature-name`.
2. Make the change, with tests and documentation.
3. Run the formatter, linter and test suite.
4. Push the branch to your fork and open a pull request.

In the pull request, describe what changed and why, and say how you tested it, including
any GPU or cluster runs the CI machines cannot do.

## Reporting issues

Please include:

- Python, GeoLattice and GDAL versions (`python3 -c "import geolattice; from osgeo import gdal; print(geolattice.__version__, gdal.__version__)"`)
- Operating system, and GPU and CUDA version if the problem involves the GPU
- A minimal example that reproduces the problem
- The expected and actual behaviour, with the full error message and traceback

## License

By contributing to GeoLattice, you agree that your contributions are licensed under the MIT
License.
