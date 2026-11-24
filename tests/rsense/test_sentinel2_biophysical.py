import pytest
from shapely.geometry import box

from geolattice.rsense.optical.sentinel2_biophysical import (
    BIOPAR_NAMESPACE,
    BIOPAR_PROCESS,
    as_spatial_extent,
    as_temporal_extent,
    biopar_cube,
    normalize_variable,
)


class _Cube:
    def __init__(self):
        self.kwargs = None

    def datacube_from_process(self, process_id, **kwargs):
        self.kwargs = (process_id, kwargs)
        return self


def test_variable_names_and_water_alias():
    assert normalize_variable("lai") == "LAI"
    assert normalize_variable("CWY") == "CWC"
    assert normalize_variable("cw") == "CWC"
    assert normalize_variable("fvc") == "FCOVER"
    with pytest.raises(ValueError, match="Unknown biophysical"):
        normalize_variable("NDVI")


def test_extents():
    bbox = as_spatial_extent((5.2, 51.1, 5.4, 51.2))
    assert bbox == {"west": 5.2, "south": 51.1, "east": 5.4, "north": 51.2}
    projected = as_spatial_extent(
        {"west": 600000, "south": 5600000, "east": 610000, "north": 5610000, "crs": 32631})
    assert projected["crs"] == 32631
    with pytest.raises(ValueError, match="projected"):
        as_spatial_extent({"west": 600000, "south": 5600000, "east": 610000, "north": 5610000})
    geom = as_spatial_extent(box(5.0, 50.0, 5.5, 50.5))
    assert geom["west"] == pytest.approx(5.0) and geom["north"] == pytest.approx(50.5)
    assert as_temporal_extent(("2024-01-01", "2024-12-31")) == ["2024-01-01", "2024-12-31"]
    assert as_temporal_extent("2024-05-01/2024-05-31") == ["2024-05-01", "2024-05-31"]


def test_biopar_cube_arguments():
    connection = _Cube()
    biopar_cube(connection, "CWY", (4.0, 50.0, 4.2, 50.2), ("2024-06-01", "2024-06-15"))
    process_id, kwargs = connection.kwargs
    assert process_id == BIOPAR_PROCESS
    assert kwargs["namespace"] == BIOPAR_NAMESPACE
    assert kwargs["biopar_type"] == "CWC"
    assert kwargs["temporal_extent"] == ["2024-06-01", "2024-06-15"]
    assert kwargs["spatial_extent"]["east"] == pytest.approx(4.2)
