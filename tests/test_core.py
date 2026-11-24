"""
Tests for core module.
"""

import pytest
from geolattice.core import (
    CRS,
    Bounds,
    CRSPromptRequired,
    CRSSelector,
    CRSSettings,
    CustomProjectionRegistry,
    DatumTransformationDialog,
    DatumTransformationRequired,
    Layer,
    LayoutMapGrid,
    Metadata,
    MissingCRSBehavior,
    Project,
    SpatialReference,
)


class TestCRS:
    """Test CRS functionality."""
    
    def test_crs_from_epsg(self):
        """Test creating CRS from EPSG code."""
        crs = CRS.from_epsg(4326)
        assert str(crs) == "EPSG:4326"
    
    def test_crs_from_string(self):
        """Test creating CRS from string."""
        crs = CRS("EPSG:4326")
        assert str(crs) == "EPSG:4326"
    
    def test_crs_from_int(self):
        """Test creating CRS from integer."""
        crs = CRS(4326)
        assert str(crs) == "EPSG:4326"
    
    def test_crs_equality(self):
        """Test CRS equality."""
        crs1 = CRS.from_epsg(4326)
        crs2 = CRS.from_epsg(4326)
        assert crs1 == crs2
    
    def test_crs_to_epsg(self):
        """Test extracting EPSG code."""
        crs = CRS.from_epsg(4326)
        assert crs.to_epsg() == 4326
    
    def test_crs_is_geographic(self):
        """Test geographic CRS detection."""
        crs = CRS.from_epsg(4326)
        assert crs.is_geographic
        assert not crs.is_projected

    def test_crs_definition_and_area(self):
        """Selector details: WKT, PROJ, and area of use."""
        crs = CRS.from_epsg(4326)
        assert crs.authority == "EPSG"
        assert crs.code == "4326"
        assert crs.name == "WGS 84"
        assert crs.wkt.startswith("GEOGCRS")
        assert crs.proj.startswith("+proj=longlat")
        assert crs.area_of_use is not None
        assert crs.area_of_use.name == "World."
        assert crs.area_of_use.bounds == (-180.0, -90.0, 180.0, 90.0)

    def test_crs_equivalent_wkt(self):
        """WKT for EPSG:4326 matches the authority CRS."""
        crs = CRS.from_epsg(4326)
        assert crs.equivalent(CRS.from_wkt(crs.wkt))

    def test_spatial_reference_formats(self):
        """SpatialReference.org encodings from the local PROJ database."""
        crs = CRS.from_epsg(4326)
        ref = crs.spatial_reference
        assert isinstance(ref, SpatialReference)
        assert ref.url == "https://spatialreference.org/ref/epsg/4326/"
        assert ref.format_url("projjson") == (
            "https://spatialreference.org/ref/epsg/4326/projjson.json"
        )
        assert ref.projjson.startswith("{")
        assert "GeographicCRS" in ref.projjson
        assert ref.ogcwkt.startswith("GEOGCS")
        assert ref.pretty_wkt2.startswith("GEOGCRS")
        assert "GCS_WGS_1984" in ref.esri_wkt
        assert ref.proj4.startswith("+proj=longlat")
        assert crs.equivalent(CRS.from_projjson(ref.projjson))

        custom = CRS(
            "+proj=longlat +datum=WGS84 +no_defs +type=crs",
            authority="USER",
            code="100000",
        )
        assert custom.spatial_reference.url is None


class TestCRSSelector:
    """CRS selector searches authorities and shows WKT, PROJ, and area of use."""

    def test_search_epsg_esri_and_iau(self):
        selector = CRSSelector(custom=CustomProjectionRegistry())
        wgs84 = selector.get("EPSG", 4326)
        assert wgs84.auth_code == "EPSG:4326"
        assert wgs84.wkt.startswith("GEOGCRS")
        assert "+proj=longlat" in wgs84.proj
        assert wgs84.area_of_use is not None
        assert wgs84.area_of_use.bounds == (-180.0, -90.0, 180.0, 90.0)
        assert "World" in wgs84.area_of_use.name

        esri = selector.search(authorities=["ESRI"], limit=1)
        assert esri[0].authority == "ESRI"
        assert esri[0].wkt
        assert esri[0].proj

        iau = selector.search("Sun", authorities=["IAU"], limit=5)
        assert iau
        assert iau[0].authority.startswith("IAU")
        assert "Sun" in iau[0].name

    def test_custom_projection_from_proj_and_wkt(self):
        registry = CustomProjectionRegistry()
        proj = registry.define("Local longlat", "+proj=longlat +datum=WGS84 +no_defs +type=crs")
        assert proj.authority == "USER"
        assert proj.identifier == "USER:100000"
        assert registry.get("Local longlat").format == "proj"

        wkt = CRS.from_epsg(3857).wkt
        web = registry.define("Web mercator copy", wkt)
        assert web.identifier == "USER:100001"
        assert registry.get("Web mercator copy").format == "wkt"
        assert web.equivalent(CRS.from_epsg(3857))

        with pytest.raises(ValueError):
            registry.define("Broken", "not-a-crs")

        found = CRSSelector(custom=registry).search("Local longlat", authorities=["USER"])
        assert [entry.auth_code for entry in found] == ["USER:100000"]
        assert found[0].proj.startswith("+proj=longlat")


class TestProjectCRS:
    """Project CRS, layer source CRS, transforms, and layout grids."""

    def test_project_reprojects_on_the_fly_and_flags_mismatch(self):
        settings = CRSSettings()
        project = Project(3857, settings=settings)
        layer = Layer("roads", 4326)
        project.add_layer(layer)

        assert project.status_bar_crs == "EPSG:3857"
        assert project.on_the_fly(layer)
        assert project.crs_mismatch(layer)
        assert project.mismatched_layers() == [layer]
        transform = project.display_transform(layer)
        assert transform is not None
        assert transform.available

        project.crs = 4326
        assert project.status_bar_crs == "EPSG:4326"
        assert not project.on_the_fly(layer)
        assert not project.crs_mismatch(layer)

    def test_layer_source_override_does_not_reproject(self):
        project = Project(4326, settings=CRSSettings())
        layer = Layer("parcels", 4326)
        project.add_layer(layer)
        layer.override_crs(3857)

        assert layer.crs_overridden
        assert layer.crs.to_epsg() == 3857
        assert project.crs.to_epsg() == 4326
        assert project.crs_mismatch(layer)

    def test_missing_crs_behaviors(self):
        prompt = CRSSettings()
        prompt.missing_crs_behavior = MissingCRSBehavior.PROMPT
        project = Project(4326, settings=prompt)
        with pytest.raises(CRSPromptRequired):
            project.add_layer(Layer("unknown"))
        prompted = project.add_layer(Layer("answered"), prompted_crs=4269)
        assert prompted.crs.to_epsg() == 4269
        assert not prompted.crs_overridden

        use_project = CRSSettings()
        use_project.missing_crs_behavior = MissingCRSBehavior.USE_PROJECT_CRS
        project = Project(3857, settings=use_project)
        layer = project.add_layer(Layer("filled"))
        assert layer.crs.to_epsg() == 3857

        use_default = CRSSettings()
        use_default.missing_crs_behavior = MissingCRSBehavior.USE_DEFAULT_CRS
        use_default.default_layer_crs = 4269
        project = Project(3857, settings=use_default)
        layer = project.add_layer(Layer("defaulted"))
        assert layer.crs.to_epsg() == 4269

        leave = CRSSettings()
        leave.missing_crs_behavior = "leave_unknown"
        project = Project(4326, settings=leave)
        layer = project.add_layer(Layer("blank"))
        assert layer.crs_is_unknown
        assert not project.crs_mismatch(layer)

    def test_first_layer_sets_project_crs(self):
        settings = CRSSettings()
        settings.new_project_uses_first_layer = True
        project = Project(settings=settings)
        assert project.crs is None
        assert project.status_bar_crs == "No CRS"
        project.add_layer(Layer("first", 3857))
        assert project.crs.to_epsg() == 3857

    def test_nad27_to_nad83_paths_and_missing_grids(self):
        settings = CRSSettings()
        dialog = DatumTransformationDialog(4267, 4269, settings)
        assert dialog.requires_choice
        assert any(not path.uses_grid and path.available for path in dialog.paths)
        assert any(path.uses_grid for path in dialog.paths)
        if any(not path.available for path in dialog.paths):
            assert dialog.offer_grid_download()

        with pytest.raises(DatumTransformationRequired) as raised:
            dialog.resolve()
        assert raised.value.dialog is dialog

        settings.ask_for_datum_transformation = False
        assert not dialog.requires_choice
        assert dialog.resolve() is not None

        chosen = dialog.paths[0]
        settings.ask_for_datum_transformation = True
        settings.prefer(4267, 4269, chosen.name)
        assert not dialog.requires_choice
        assert dialog.resolve().name == chosen.name

    def test_layout_grid_can_use_another_crs(self):
        grid = LayoutMapGrid(3857)
        assert grid.draws_in_map_crs
        grid.grid_crs = 4326
        assert grid.map_crs.to_epsg() == 3857
        assert grid.grid_crs.to_epsg() == 4326
        assert not grid.draws_in_map_crs


class TestBounds:
    """Test Bounds functionality."""
    
    def test_bounds_creation(self):
        """Test creating bounds."""
        bounds = Bounds(0, 0, 10, 10)
        assert bounds.minx == 0
        assert bounds.miny == 0
        assert bounds.maxx == 10
        assert bounds.maxy == 10
    
    def test_bounds_properties(self):
        """Test bounds properties."""
        bounds = Bounds(0, 0, 10, 10)
        assert bounds.width == 10
        assert bounds.height == 10
        assert bounds.area == 100
        assert bounds.centroid == (5, 5)
    
    def test_bounds_invalid(self):
        """Test invalid bounds raise error."""
        with pytest.raises(ValueError):
            Bounds(10, 10, 0, 0)
    
    def test_bounds_contains(self):
        """Test point containment."""
        bounds = Bounds(0, 0, 10, 10)
        assert (5, 5) in bounds
        assert (15, 15) not in bounds
    
    def test_bounds_intersects(self):
        """Test bounds intersection."""
        bounds1 = Bounds(0, 0, 10, 10)
        bounds2 = Bounds(5, 5, 15, 15)
        assert bounds1.intersects(bounds2)
        
        bounds3 = Bounds(20, 20, 30, 30)
        assert not bounds1.intersects(bounds3)
    
    def test_bounds_intersection(self):
        """Test bounds intersection calculation."""
        bounds1 = Bounds(0, 0, 10, 10)
        bounds2 = Bounds(5, 5, 15, 15)
        intersection = bounds1.intersection(bounds2)
        assert intersection.minx == 5
        assert intersection.miny == 5
        assert intersection.maxx == 10
        assert intersection.maxy == 10
    
    def test_bounds_union(self):
        """Test bounds union."""
        bounds1 = Bounds(0, 0, 10, 10)
        bounds2 = Bounds(5, 5, 15, 15)
        union = bounds1.union(bounds2)
        assert union.minx == 0
        assert union.miny == 0
        assert union.maxx == 15
        assert union.maxy == 15
    
    def test_bounds_buffer(self):
        """Test bounds buffering."""
        bounds = Bounds(0, 0, 10, 10)
        buffered = bounds.buffer(5)
        assert buffered.minx == -5
        assert buffered.miny == -5
        assert buffered.maxx == 15
        assert buffered.maxy == 15
    
    def test_bounds_from_points(self):
        """Test creating bounds from points."""
        points = [(0, 0), (10, 10), (5, 5)]
        bounds = Bounds.from_points(points)
        assert bounds.minx == 0
        assert bounds.miny == 0
        assert bounds.maxx == 10
        assert bounds.maxy == 10


class TestMetadata:
    """Test Metadata functionality."""
    
    def test_metadata_creation(self):
        """Test creating metadata."""
        metadata = Metadata()
        assert metadata.created is not None
        assert metadata.modified is not None
    
    def test_metadata_attributes(self):
        """Test custom attributes."""
        metadata = Metadata()
        metadata.set_attribute("key", "value")
        assert metadata.get_attribute("key") == "value"
        assert metadata.get_attribute("nonexistent", "default") == "default"
    
    def test_metadata_quality(self):
        """Test quality indicators."""
        metadata = Metadata()
        metadata.set_quality_indicator("accuracy", 0.95)
        assert metadata.get_quality_indicator("accuracy") == 0.95
    
    def test_metadata_provenance(self):
        """Test provenance tracking."""
        metadata = Metadata()
        metadata.add_provenance_entry("buffer", {"distance": 100})
        provenance = metadata.provenance
        assert len(provenance) == 1
        assert provenance[0]["operation"] == "buffer"
    
    def test_metadata_to_dict(self):
        """Test metadata serialization."""
        metadata = Metadata()
        metadata.set_attribute("key", "value")
        data = metadata.to_dict()
        assert "attributes" in data
        assert data["attributes"]["key"] == "value"
    
    def test_metadata_to_json(self):
        """Test metadata JSON serialization."""
        metadata = Metadata()
        metadata.set_attribute("key", "value")
        json_str = metadata.to_json()
        assert "key" in json_str
        assert "value" in json_str
