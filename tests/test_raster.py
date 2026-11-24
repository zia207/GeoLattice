"""
Tests for raster module.
"""

from geolattice.raster import RasterData, RasterOperationError, create, open_raster_data


class TestRasterModule:
    """Test raster module imports."""

    def test_raster_import(self):
        """Test that raster module can be imported."""
        assert RasterData is not None
        assert RasterOperationError is not None
        assert create is not None
        assert open_raster_data is not None
