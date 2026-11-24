"""
Coordinate reference system management.

Definitions come from the PROJ database through pyproj, the same database
published at SpatialReference.org. Each authority CRS can be written as the
formats that site serves: WKT1, WKT2:2019, ESRI WKT, PROJJSON, and a PROJ
string.

* SpatialReference.org: https://spatialreference.org/
* pyproj: https://pyproj4.github.io/pyproj/stable/
* PROJ: https://proj.org/en/stable/

Project Properties → CRS sets the project CRS. Every layer is reprojected to it
on the fly. Layer Properties → Source shows or overrides the CRS assigned to a
single layer. The CRS selector searches EPSG, ESRI, IAU, and other authorities
and exposes each CRS's WKT, PROJ definition, and area of use. Custom projections
are defined from a PROJ string or WKT. CRS and transform settings choose the
default CRS, what happens when a layer has no CRS, and preferred datum
transformations. When several transformation paths exist, for example NAD27 to
NAD83 with or without NADCON grids, a datum transformation choice lists them
and can download missing grids.

Geometry and grid reprojection are not implemented here:

* Vector geometry reprojection → :mod:`geolattice.vector.vector_CRS`
  (``reproject_vector``, ``align_vector_crs``, datum choice for vectors)
* Raster warp / reproject → :mod:`geolattice.raster.raster_CRS`
  (``warp``, ``reproject_raster``, ``gdal.WarpOptions``, datum choice for rasters)

This module stays the shared CRS catalog, settings, and datum-transformation
dialog that both vector and raster helpers use.
"""

from __future__ import annotations

import re
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pyproj
from pyproj.database import get_authorities, query_crs_info
from pyproj.exceptions import CRSError
from pyproj.transformer import TransformerGroup

_AUTH_CODE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*):(\d+)$")
_CATALOG: Dict[str, List[Any]] = {}

SPATIALREFERENCE_HOME = "https://spatialreference.org/"
PYPROJ_DOCS = "https://pyproj4.github.io/pyproj/stable/"
PROJ_DOCS = "https://proj.org/en/stable/"

# Path suffixes used by https://spatialreference.org/ref/{authority}/{code}/
_SPATIAL_REFERENCE_FORMATS = {
    "ogcwkt": "ogcwkt/",
    "prettywkt": "prettywkt.txt",
    "wkt1": "prettywkt.txt",
    "prettywkt2": "prettywkt2.txt",
    "wkt2": "prettywkt2.txt",
    "esriwkt": "esriwkt.txt",
    "esri": "esriwkt.txt",
    "prj": "esriwkt.txt",
    "projjson": "projjson.json",
    "proj4": "proj4.txt",
}


class CRSPromptRequired(Exception):
    """A layer has no CRS and the settings say to prompt for one."""


class DatumTransformationRequired(Exception):
    """Several datum transformations exist and the user needs to choose one."""

    def __init__(self, dialog: "DatumTransformationDialog") -> None:
        self.dialog = dialog
        super().__init__(
            f"Several datum transformations exist from {dialog.source} to {dialog.target}"
        )


class MissingCRSBehavior(Enum):
    """What to do when a layer is created or loaded with no CRS."""

    PROMPT = "prompt"
    USE_PROJECT_CRS = "use_project"
    USE_DEFAULT_CRS = "use_default"
    LEAVE_UNKNOWN = "leave_unknown"


class AreaOfUse:
    """Geographic area where a CRS is appropriate."""

    def __init__(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        name: str = "",
    ) -> None:
        self.west = float(west)
        self.south = float(south)
        self.east = float(east)
        self.north = float(north)
        self.name = name

    @property
    def bounds(self) -> tuple:
        """Return (west, south, east, north)."""
        return (self.west, self.south, self.east, self.north)

    @classmethod
    def from_pyproj(cls, area: Any) -> Optional["AreaOfUse"]:
        """Build an area from a pyproj area of use, if one is present."""
        if area is None:
            return None
        return cls(
            west=float(area.west),
            south=float(area.south),
            east=float(area.east),
            north=float(area.north),
            name=str(area.name or ""),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AreaOfUse):
            return False
        return self.bounds == other.bounds and self.name == other.name

    def __repr__(self) -> str:
        return (
            f"AreaOfUse(west={self.west}, south={self.south}, "
            f"east={self.east}, north={self.north}, name={self.name!r})"
        )


def _as_crs(value: Union["CRS", str, int]) -> "CRS":
    if isinstance(value, CRS):
        return value
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(
            f"Expected a CRS, EPSG code, or CRS string, got {type(value).__name__}"
        )
    return CRS(value)


def _is_crs_error(exc: BaseException) -> bool:
    # Coverage can import pyproj twice, so the raised type may not be this CRSError.
    return isinstance(exc, CRSError) or type(exc).__name__ == "CRSError"


def _bind_pyproj(crs_string: str) -> Any:
    try:
        return pyproj.CRS.from_user_input(crs_string)
    except Exception as exc:
        if _is_crs_error(exc):
            return None
        raise


def _proj4(proj_crs: Any) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        # Pass the version as an int. The ProjVersion default breaks when test
        # coverage imports pyproj.enums a second time.
        return str(proj_crs.to_proj4(version=5))


def _export_wkt(proj_crs: Any, version: str, pretty: bool) -> Optional[str]:
    """Write WKT the way SpatialReference.org does, or None when PROJ cannot."""
    try:
        output_axis_rule = True if proj_crs.is_projected else None
        text = proj_crs.to_wkt(
            version=version,
            pretty=pretty,
            output_axis_rule=output_axis_rule,
        )
    except Exception:
        return None
    return str(text) if text else None


def _projjson(proj_crs: Any) -> Optional[str]:
    """PROJJSON text. Same encoding as spatialreference.org ``projjson.json``."""
    try:
        text = proj_crs.to_json(pretty=False)
    except Exception:
        return None
    return str(text) if text else None


def _spatial_reference_url(authority: Optional[str], code: Optional[str]) -> Optional[str]:
    """Page for this authority code on SpatialReference.org, when one exists."""
    if not authority or not code:
        return None
    if authority.upper() in {"USER", "CUSTOM"}:
        return None
    known = {name.upper() for name in get_authorities()}
    if authority.upper() not in known:
        return None
    return f"{SPATIALREFERENCE_HOME}ref/{authority.lower()}/{code}/"


def _authority_records(authority: str) -> List[Any]:
    records = _CATALOG.get(authority)
    if records is None:
        records = list(query_crs_info(auth_name=authority, allow_deprecated=True))
        _CATALOG[authority] = records
    return records


def _resolve_authorities(requested: Optional[Sequence[str]]) -> List[str]:
    available = list(get_authorities())
    if requested is None:
        return available
    resolved: List[str] = []
    for name in requested:
        key = name.strip().upper()
        if key in {"USER", "CUSTOM"}:
            continue
        matches = [
            auth
            for auth in available
            if auth.upper() == key or auth.upper().startswith(key + "_")
        ]
        if not matches:
            known = ", ".join(available)
            raise ValueError(f"Unknown CRS authority {name!r}. Known authorities: {known}")
        for auth in matches:
            if auth not in resolved:
                resolved.append(auth)
    return resolved


def _definition_format(definition: str) -> str:
    stripped = definition.lstrip()
    lowered = stripped.lower()
    if stripped.startswith("+") or lowered.startswith("proj="):
        return "proj"
    if stripped.startswith("{"):
        return "projjson"
    return "wkt"


def _code_sort(code: str) -> tuple:
    if code.isdigit():
        return (0, int(code))
    return (1, code)


def _accuracy(operation: Any) -> Optional[float]:
    value = getattr(operation, "accuracy", None)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def _description(operation: Any) -> str:
    for attr in ("description", "remarks", "scope", "method_name"):
        value = getattr(operation, attr, None)
        if value:
            return str(value)
    return ""


def _operation_definition(operation: Any) -> str:
    definition = getattr(operation, "definition", None)
    if definition:
        return str(definition)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        for method_name in ("to_proj4", "to_wkt"):
            method = getattr(operation, method_name, None)
            if method is None:
                continue
            try:
                if method_name == "to_wkt":
                    text = method(version="WKT2_2019")
                else:
                    text = method(version=5)
            except Exception:
                continue
            if text:
                return str(text)
    return ""


def _grids_from(operation: Any) -> List["TransformationGrid"]:
    found: List[TransformationGrid] = []
    seen = set()

    def add(obj: Any) -> None:
        ident = id(obj)
        if ident in seen:
            return
        seen.add(ident)
        for grid in getattr(obj, "grids", None) or []:
            found.append(
                TransformationGrid(
                    short_name=str(getattr(grid, "short_name", "") or ""),
                    url=str(getattr(grid, "url", "") or ""),
                    available=bool(getattr(grid, "available", False)),
                    direct_download=bool(getattr(grid, "direct_download", False)),
                    open_license=bool(getattr(grid, "open_license", False)),
                )
            )
        for child in getattr(obj, "operations", None) or []:
            if child is not obj:
                add(child)

    add(operation)
    return found


class CRS:
    """
    Coordinate reference system representation.

    This class wraps coordinate reference system information and provides
    methods for CRS validation and transformation.
    """

    def __init__(
        self,
        crs_string: Union[str, int],
        *,
        name: Optional[str] = None,
        authority: Optional[str] = None,
        code: Optional[Union[str, int]] = None,
    ) -> None:
        """
        Initialize CRS from string or EPSG code.

        Args:
            crs_string: CRS definition (EPSG code, WKT string, or PROJ string)
            name: Optional display name. Used for custom projections.
            authority: Optional authority, such as USER for a custom projection.
            code: Optional authority code paired with ``authority``.

        Examples:
            >>> crs = CRS("EPSG:4326")  # WGS84
            >>> crs = CRS(4326)  # Same as above
            >>> crs = CRS("PROJCS[...]")  # WKT string
        """
        if isinstance(crs_string, bool) or not isinstance(crs_string, (str, int)):
            raise TypeError(
                f"Expected a CRS string or EPSG code, got {type(crs_string).__name__}"
            )
        if isinstance(crs_string, int):
            self._crs_string = f"EPSG:{crs_string}"
        else:
            self._crs_string = crs_string

        self._name = name
        self._authority = authority
        self._code = None if code is None else str(code)
        self._validate_crs()
        self._proj = _bind_pyproj(self._crs_string)

    def _validate_crs(self) -> None:
        """Validate CRS string format."""
        if not self._crs_string:
            raise ValueError("CRS string cannot be empty")

        # Check for common patterns
        if not re.match(r"^(EPSG:\d+|PROJCS|GEOGCS|\+proj=)", self._crs_string):
            # Still accept it, but warn in production
            pass

    def __str__(self) -> str:
        """String representation of CRS."""
        return self._crs_string

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"CRS('{self._crs_string}')"

    def __eq__(self, other: object) -> bool:
        """Check CRS equality."""
        if not isinstance(other, CRS):
            return False
        return self._crs_string == other._crs_string

    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        return hash(self._crs_string)

    def equivalent(self, other: object) -> bool:
        """Return True when two definitions describe the same CRS."""
        if not isinstance(other, CRS):
            return False
        if self._crs_string == other._crs_string:
            return True
        if self._proj is not None and other._proj is not None:
            try:
                return bool(self._proj.equals(other._proj))
            except Exception:
                return False
        return False

    @property
    def name(self) -> str:
        """CRS name, for example ``WGS 84``."""
        if self._name:
            return self._name
        if self._proj is not None:
            return str(self._proj.name)
        return self._crs_string

    @property
    def authority(self) -> Optional[str]:
        """Authority name, such as EPSG, ESRI, IAU_2015, or USER."""
        if self._authority:
            return self._authority
        if self._proj is not None:
            auth = self._proj.to_authority()
            if auth:
                return str(auth[0])
        match = _AUTH_CODE.match(self._crs_string.strip())
        if match:
            return match.group(1)
        return None

    @property
    def code(self) -> Optional[str]:
        """Code within :attr:`authority`."""
        if self._code:
            return self._code
        if self._proj is not None:
            auth = self._proj.to_authority()
            if auth:
                return str(auth[1])
        match = _AUTH_CODE.match(self._crs_string.strip())
        if match:
            return match.group(2)
        return None

    @property
    def identifier(self) -> str:
        """Authority and code, or the original definition when neither is known."""
        if self.authority and self.code:
            return f"{self.authority}:{self.code}"
        return self._crs_string

    @property
    def wkt(self) -> Optional[str]:
        """Well-known text definition."""
        if self._proj is not None:
            return str(self._proj.to_wkt(version="WKT2_2019"))
        upper = self._crs_string.lstrip().upper()
        prefixes = (
            "GEOGCS",
            "PROJCS",
            "GEOGCRS",
            "PROJCRS",
            "COMPD_CS",
            "VERT_CS",
            "BOUNDCRS",
            "VERTCRS",
        )
        if upper.startswith(prefixes):
            return self._crs_string
        return None

    @property
    def proj(self) -> Optional[str]:
        """PROJ definition."""
        if self._proj is not None:
            return _proj4(self._proj)
        stripped = self._crs_string.lstrip()
        if stripped.startswith("+") or stripped.lower().startswith("proj="):
            return self._crs_string
        return None

    @property
    def projjson(self) -> Optional[str]:
        """PROJJSON definition, equivalent to WKT2:2019.

        This is the encoding documented by PROJ and served as
        ``projjson.json`` on SpatialReference.org.
        """
        if self._proj is None:
            stripped = self._crs_string.lstrip()
            if stripped.startswith("{"):
                return self._crs_string
            return None
        return _projjson(self._proj)

    def to_wkt(self, version: str = "WKT2_2019", pretty: bool = False) -> Optional[str]:
        """Export WKT.

        ``version`` is a PROJ WKT version such as ``WKT2_2019``, ``WKT1_GDAL``,
        or ``WKT1_ESRI``. Pretty WKT2 matches ``prettywkt2.txt`` on
        SpatialReference.org. ``WKT1_GDAL`` matches ``ogcwkt`` and
        ``prettywkt.txt``. ``WKT1_ESRI`` matches ``esriwkt.txt``.
        """
        if self._proj is None:
            if version.upper().startswith("WKT2") and self.wkt:
                return self.wkt
            return None
        return _export_wkt(self._proj, version, pretty)

    @property
    def spatial_reference(self) -> "SpatialReference":
        """SpatialReference.org formats for this CRS."""
        return SpatialReference(self)

    @property
    def area_of_use(self) -> Optional[AreaOfUse]:
        """Area where this CRS is intended to be used."""
        if self._proj is None:
            return None
        return AreaOfUse.from_pyproj(self._proj.area_of_use)

    @property
    def is_geographic(self) -> bool:
        """
        Check if CRS is geographic (lat/lon).

        Returns:
            True if geographic coordinate system
        """
        if self._proj is not None:
            return bool(self._proj.is_geographic)
        upper = self._crs_string.upper()
        return (
            "GEOGCS" in upper
            or "GEOGCRS" in upper
            or "EPSG:4326" in upper
            or "+PROJ=LONG" in upper
            or "+PROJ=LATLONG" in upper
        )

    @property
    def is_projected(self) -> bool:
        """
        Check if CRS is projected.

        Returns:
            True if projected coordinate system
        """
        if self._proj is not None:
            return bool(self._proj.is_projected)
        return not self.is_geographic

    def to_epsg(self) -> Optional[int]:
        """
        Extract EPSG code if available.

        Returns:
            EPSG code as integer, or None if not available
        """
        if self._proj is not None:
            try:
                code = self._proj.to_epsg()
            except Exception:
                code = None
            if code is not None:
                return int(code)
        match = re.search(r"EPSG:(\d+)", self._crs_string, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    @classmethod
    def from_epsg(cls, epsg_code: int) -> "CRS":
        """
        Create CRS from EPSG code.

        Args:
            epsg_code: EPSG code (e.g., 4326 for WGS84)

        Returns:
            CRS object
        """
        return cls(epsg_code)

    @classmethod
    def from_authority(cls, authority: str, code: Union[str, int]) -> "CRS":
        """
        Create a CRS from an authority and code.

        Args:
            authority: Authority name, for example EPSG, ESRI, or IAU_2015.
            code: Code published by that authority.

        Returns:
            CRS object
        """
        return cls(f"{authority}:{code}")

    @classmethod
    def from_wkt(cls, wkt_string: str) -> "CRS":
        """
        Create CRS from WKT string.

        Args:
            wkt_string: Well-Known Text representation

        Returns:
            CRS object
        """
        return cls(wkt_string)

    @classmethod
    def from_proj4(cls, proj4_string: str) -> "CRS":
        """
        Create CRS from PROJ.4 string.

        Args:
            proj4_string: PROJ.4 string

        Returns:
            CRS object
        """
        return cls(proj4_string)

    @classmethod
    def from_projjson(cls, projjson: str) -> "CRS":
        """
        Create a CRS from PROJJSON.

        PROJJSON is the JSON encoding of WKT2:2019 described at
        https://proj.org/en/stable/specifications/projjson.html and served by
        SpatialReference.org as ``projjson.json``.

        Args:
            projjson: PROJJSON text.

        Returns:
            CRS object
        """
        if not isinstance(projjson, str) or not projjson.lstrip().startswith("{"):
            raise ValueError("Expected a PROJJSON object")
        return cls(projjson)


class SpatialReference:
    """Formats SpatialReference.org publishes for one coordinate reference system.

    The text is produced locally with pyproj from the PROJ database. That is
    the same database the website is built from, so these strings match
    https://spatialreference.org/ref/{authority}/{code}/ without a network
    request. Authorities the site does not publish, including USER, have no URL.

    See https://spatialreference.org/, https://pyproj4.github.io/pyproj/stable/,
    and https://proj.org/en/stable/.
    """

    def __init__(self, crs: CRS) -> None:
        self.crs = crs

    @property
    def url(self) -> Optional[str]:
        """SpatialReference.org page, for example EPSG:4326."""
        return _spatial_reference_url(self.crs.authority, self.crs.code)

    def format_url(self, fmt: str) -> Optional[str]:
        """URL of one published encoding, such as ``projjson`` or ``ogcwkt``."""
        page = self.url
        if page is None:
            return None
        key = fmt.strip().lower()
        suffix = _SPATIAL_REFERENCE_FORMATS.get(key)
        if suffix is None:
            known = ", ".join(sorted(set(_SPATIAL_REFERENCE_FORMATS)))
            raise ValueError(f"Unknown spatial reference format {fmt!r}. Known formats: {known}")
        return page + suffix

    @property
    def projjson(self) -> Optional[str]:
        """PROJJSON, the ``projjson.json`` download."""
        return self.crs.projjson

    @property
    def ogcwkt(self) -> Optional[str]:
        """Single-line WKT1, the ``ogcwkt`` download."""
        return self.crs.to_wkt("WKT1_GDAL", pretty=False)

    @property
    def pretty_wkt(self) -> Optional[str]:
        """Indented WKT1, the ``prettywkt.txt`` download."""
        return self.crs.to_wkt("WKT1_GDAL", pretty=True)

    @property
    def pretty_wkt2(self) -> Optional[str]:
        """Indented WKT2:2019, the ``prettywkt2.txt`` download."""
        return self.crs.to_wkt("WKT2_2019", pretty=True)

    @property
    def esri_wkt(self) -> Optional[str]:
        """Esri WKT used for a shapefile ``.prj``, the ``esriwkt.txt`` download."""
        return self.crs.to_wkt("WKT1_ESRI", pretty=False)

    @property
    def proj4(self) -> Optional[str]:
        """Legacy PROJ string, the ``proj4.txt`` download."""
        return self.crs.proj

    def __repr__(self) -> str:
        return f"SpatialReference({self.crs.identifier!r}, url={self.url!r})"


class CRSCatalogEntry:
    """One CRS in the selector: authority, code, WKT, PROJ, and area of use."""

    def __init__(
        self,
        authority: str,
        code: str,
        name: str,
        area_of_use: Optional[AreaOfUse] = None,
        projection_method: Optional[str] = None,
        type_name: Optional[str] = None,
        deprecated: bool = False,
        crs: Optional[CRS] = None,
    ) -> None:
        self.authority = authority
        self.code = code
        self.name = name
        self.area_of_use = area_of_use
        self.projection_method = projection_method
        self.type_name = type_name
        self.deprecated = deprecated
        self._crs = crs

    @property
    def auth_code(self) -> str:
        """Authority and code, for example ``EPSG:4326``."""
        return f"{self.authority}:{self.code}"

    @property
    def crs(self) -> CRS:
        """CRS object for this catalog record."""
        if self._crs is None:
            self._crs = CRS.from_authority(self.authority, self.code)
        return self._crs

    @property
    def wkt(self) -> str:
        """WKT definition shown in the CRS selector."""
        return self.crs.wkt or ""

    @property
    def proj(self) -> str:
        """PROJ definition shown in the CRS selector."""
        return self.crs.proj or ""

    @property
    def projjson(self) -> str:
        """PROJJSON definition shown in the CRS selector."""
        return self.crs.projjson or ""

    @property
    def spatial_reference(self) -> SpatialReference:
        """SpatialReference.org formats for this catalog record."""
        return self.crs.spatial_reference

    def __repr__(self) -> str:
        return f"CRSCatalogEntry({self.auth_code!r}, {self.name!r})"


def _entry_from_info(info: Any) -> CRSCatalogEntry:
    type_name = getattr(getattr(info, "type", None), "value", None)
    if type_name is None and getattr(info, "type", None) is not None:
        type_name = str(info.type)
    method = getattr(info, "projection_method_name", None)
    return CRSCatalogEntry(
        authority=str(info.auth_name),
        code=str(info.code),
        name=str(info.name),
        area_of_use=AreaOfUse.from_pyproj(getattr(info, "area_of_use", None)),
        projection_method=str(method) if method else None,
        type_name=str(type_name) if type_name else None,
        deprecated=bool(getattr(info, "deprecated", False)),
    )


def _entry_matches(entry: CRSCatalogEntry, text: str) -> bool:
    area_name = entry.area_of_use.name if entry.area_of_use is not None else ""
    haystack = (
        entry.name,
        entry.code,
        entry.auth_code,
        entry.projection_method or "",
        area_name,
    )
    return any(text in part.lower() for part in haystack)


class CustomProjection:
    """A CRS defined from a PROJ string or WKT."""

    def __init__(self, name: str, definition: str, code: str, crs: CRS, format: str) -> None:
        self.name = name
        self.definition = definition
        self.code = code
        self.crs = crs
        self.format = format
        self.authority = "USER"

    @property
    def auth_code(self) -> str:
        """User authority code, for example ``USER:100000``."""
        return f"{self.authority}:{self.code}"

    def __repr__(self) -> str:
        return f"CustomProjection({self.name!r}, {self.auth_code})"


def _entry_from_custom(item: CustomProjection) -> CRSCatalogEntry:
    return CRSCatalogEntry(
        authority=item.authority,
        code=item.code,
        name=item.name,
        area_of_use=item.crs.area_of_use,
        type_name="USER_DEFINED",
        crs=item.crs,
    )


class CustomProjectionRegistry:
    """Settings → Custom Projections. Definitions are a PROJ string or WKT."""

    def __init__(self) -> None:
        self._items: Dict[str, CustomProjection] = {}
        self._by_code: Dict[str, CustomProjection] = {}
        self._next_code = 100000

    def define(self, name: str, definition: str) -> CRS:
        """
        Define a custom CRS.

        Args:
            name: Name shown for the projection.
            definition: PROJ string or WKT.

        Returns:
            The CRS assigned a USER authority code.

        Raises:
            ValueError: If the name is empty, already used, or the definition
                is not valid PROJ or WKT.
        """
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError("Custom projection name cannot be empty")
        if cleaned_name in self._items:
            raise ValueError(f"Custom projection {cleaned_name!r} already exists")
        cleaned = definition.strip()
        if not cleaned:
            raise ValueError("Custom projection definition cannot be empty")
        try:
            pyproj.CRS.from_user_input(cleaned)
        except Exception as exc:
            if not _is_crs_error(exc):
                raise
            raise ValueError(f"Invalid PROJ string or WKT: {exc}") from exc

        code = str(self._next_code)
        self._next_code += 1
        crs = CRS(cleaned, name=cleaned_name, authority="USER", code=code)
        item = CustomProjection(
            name=cleaned_name,
            definition=cleaned,
            code=code,
            crs=crs,
            format=_definition_format(cleaned),
        )
        self._items[cleaned_name] = item
        self._by_code[code] = item
        return crs

    def get(self, name: str) -> CustomProjection:
        """Return a custom projection by name."""
        try:
            return self._items[name]
        except KeyError as exc:
            raise KeyError(f"No custom projection named {name!r}") from exc

    def get_by_code(self, code: Union[str, int]) -> CustomProjection:
        """Return a custom projection by its USER code."""
        try:
            return self._by_code[str(code)]
        except KeyError as exc:
            raise KeyError(f"No custom projection with code {code}") from exc

    def remove(self, name: str) -> None:
        """Remove a custom projection by name."""
        item = self.get(name)
        del self._items[item.name]
        del self._by_code[item.code]

    def __iter__(self) -> Any:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)


custom_projections = CustomProjectionRegistry()


class CRSSelector:
    """
    CRS selector dialog.

    Searches EPSG, ESRI, IAU, and the other authorities in the PROJ database,
    plus custom projections. Each result exposes WKT, PROJ, and area of use.
    ``IAU`` matches the IAU authority shipped with PROJ (currently IAU_2015).
    """

    def __init__(self, custom: Optional[CustomProjectionRegistry] = None) -> None:
        self.custom = custom_projections if custom is None else custom

    def authorities(self) -> List[str]:
        """Authority names the selector can search, including USER."""
        names = list(get_authorities())
        if "USER" not in names:
            names.append("USER")
        return names

    def search(
        self,
        query: str = "",
        authorities: Optional[Sequence[str]] = None,
        *,
        limit: Optional[int] = None,
        allow_deprecated: bool = False,
    ) -> List[CRSCatalogEntry]:
        """
        Search CRS definitions.

        Args:
            query: Text matched against name, authority code, area of use,
                and projection method. An empty query returns every match.
            authorities: Authorities to search. ``None`` searches all of them.
                ``IAU`` selects IAU authorities such as ``IAU_2015``.
            limit: Maximum number of results.
            allow_deprecated: Include deprecated CRS definitions.

        Returns:
            Matching catalog entries, ordered by authority and code.
        """
        text = query.strip().lower()
        resolved = _resolve_authorities(authorities)
        include_user = authorities is None or any(
            name.strip().upper() in {"USER", "CUSTOM"} for name in authorities
        )
        entries: List[CRSCatalogEntry] = []
        for authority in resolved:
            for info in _authority_records(authority):
                if info.deprecated and not allow_deprecated:
                    continue
                entry = _entry_from_info(info)
                if text and not _entry_matches(entry, text):
                    continue
                entries.append(entry)
        if include_user:
            for item in self.custom:
                entry = _entry_from_custom(item)
                if text and not _entry_matches(entry, text) and text not in item.definition.lower():
                    continue
                entries.append(entry)
        entries.sort(key=lambda entry: (entry.authority, _code_sort(entry.code), entry.name))
        if limit is not None:
            if limit < 0:
                raise ValueError("limit cannot be negative")
            return entries[:limit]
        return entries

    def get(self, authority: str, code: Union[str, int]) -> CRSCatalogEntry:
        """Return one CRS by authority and code."""
        code_text = str(code)
        key = authority.strip().upper()
        if key in {"USER", "CUSTOM"}:
            return _entry_from_custom(self.custom.get_by_code(code_text))
        for auth in _resolve_authorities([authority]):
            for info in _authority_records(auth):
                if str(info.code) == code_text:
                    return _entry_from_info(info)
        raise KeyError(f"No CRS {authority}:{code_text}")


class PreferredDatumTransformation:
    """A datum transformation chosen ahead of time for a source and target CRS."""

    def __init__(self, source: CRS, target: CRS, operation: str) -> None:
        self.source = source
        self.target = target
        self.operation = operation

    def __repr__(self) -> str:
        return (
            f"PreferredDatumTransformation({self.source.identifier} -> "
            f"{self.target.identifier}, {self.operation!r})"
        )


class CRSSettings:
    """
    Settings → Options → CRS and Transforms.

    Sets the default CRS for new projects, the CRS used for new layers, what
    happens when a layer has no CRS, and the preferred datum transformations.
    """

    def __init__(self) -> None:
        self.default_crs: Union[CRS, str, int] = CRS.from_epsg(4326)
        self.default_layer_crs: Union[CRS, str, int] = CRS.from_epsg(4326)
        self.new_project_uses_first_layer = False
        self.ask_for_datum_transformation = True
        self.preferred_transformations: List[PreferredDatumTransformation] = []
        self.missing_crs_behavior = MissingCRSBehavior.PROMPT

    @property
    def missing_crs_behavior(self) -> MissingCRSBehavior:
        """Behavior used when a layer has no CRS."""
        return self._missing_crs_behavior

    @missing_crs_behavior.setter
    def missing_crs_behavior(self, value: Union[MissingCRSBehavior, str]) -> None:
        if isinstance(value, str):
            behavior = MissingCRSBehavior(value)
        elif isinstance(value, MissingCRSBehavior):
            behavior = value
        else:
            raise TypeError(
                "missing_crs_behavior must be a MissingCRSBehavior or its value string"
            )
        self._missing_crs_behavior = behavior

    def prefer(
        self,
        source: Union[CRS, str, int],
        target: Union[CRS, str, int],
        operation: str,
    ) -> PreferredDatumTransformation:
        """Remember the datum transformation to use between two CRSs."""
        item = PreferredDatumTransformation(_as_crs(source), _as_crs(target), operation)
        self.preferred_transformations = [
            current
            for current in self.preferred_transformations
            if not (
                current.source.equivalent(item.source) and current.target.equivalent(item.target)
            )
        ]
        self.preferred_transformations.append(item)
        return item

    def resolve_missing_layer_crs(
        self,
        project_crs: Optional[CRS],
        prompted_crs: Optional[Union[CRS, str, int]] = None,
    ) -> Optional[CRS]:
        """
        Decide the CRS for a layer that has none.

        Returns:
            The CRS to assign, or None when the layer is left unknown.

        Raises:
            CRSPromptRequired: When the setting is to prompt and no CRS was given.
        """
        behavior = self.missing_crs_behavior
        if behavior is MissingCRSBehavior.LEAVE_UNKNOWN:
            return None
        if behavior is MissingCRSBehavior.USE_PROJECT_CRS:
            if project_crs is None:
                raise CRSPromptRequired("Project CRS is not set")
            return project_crs
        if behavior is MissingCRSBehavior.USE_DEFAULT_CRS:
            return _as_crs(self.default_layer_crs)
        if prompted_crs is None:
            raise CRSPromptRequired("Layer has no CRS; prompt for a CRS")
        return _as_crs(prompted_crs)


crs_settings = CRSSettings()


class TransformationGrid:
    """A grid file used by a datum transformation, and whether it is installed."""

    def __init__(
        self,
        short_name: str,
        url: str = "",
        available: bool = False,
        direct_download: bool = False,
        open_license: bool = False,
    ) -> None:
        self.short_name = short_name
        self.url = url
        self.available = available
        self.direct_download = direct_download
        self.open_license = open_license

    def __repr__(self) -> str:
        state = "installed" if self.available else "missing"
        return f"TransformationGrid({self.short_name!r}, {state})"


class DatumTransformation:
    """One transformation path between two coordinate reference systems."""

    def __init__(
        self,
        name: str,
        description: str,
        accuracy: Optional[float],
        definition: str,
        grids: List[TransformationGrid],
        available: bool,
        source: CRS,
        target: CRS,
        method: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self.accuracy = accuracy
        self.definition = definition
        self.grids = grids
        self.available = available
        self.source = source
        self.target = target
        self.method = method

    @property
    def uses_grid(self) -> bool:
        """True when this path shifts coordinates with a grid such as NADCON."""
        return bool(self.grids)

    @property
    def missing_grids(self) -> List[TransformationGrid]:
        """Grid files this path needs that are not installed."""
        return [grid for grid in self.grids if not grid.available]

    def __repr__(self) -> str:
        return f"DatumTransformation({self.name!r}, available={self.available})"


class DatumTransformationDialog:
    """
    Choose a datum transformation when several paths exist.

    NAD27 (EPSG:4267) to NAD83 (EPSG:4269) is the usual example: a path with
    no grid, and NADCON or NTv2 paths that need grid files. Missing grids are
    listed so they can be downloaded.
    """

    def __init__(
        self,
        source: Union[CRS, str, int],
        target: Union[CRS, str, int],
        settings: Optional[CRSSettings] = None,
    ) -> None:
        self.source = _as_crs(source)
        self.target = _as_crs(target)
        self.settings = crs_settings if settings is None else settings
        self._group: Any = None
        self._paths: List[DatumTransformation] = []
        self._load()

    def _load(self) -> None:
        self._paths = []
        self._group = None
        if self.source.equivalent(self.target):
            return
        source_proj = self.source._proj
        target_proj = self.target._proj
        if source_proj is None or target_proj is None:
            raise ValueError(
                f"Cannot build a datum transformation from {self.source} to {self.target}"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self._group = TransformerGroup(source_proj, target_proj)
        paths: List[DatumTransformation] = []
        for transformer in self._group.transformers:
            paths.append(self._path_from(transformer, listed_available=True))
        for operation in self._group.unavailable_operations:
            paths.append(self._path_from(operation, listed_available=False))
        paths.sort(
            key=lambda path: (
                path.accuracy is None,
                path.accuracy if path.accuracy is not None else 0.0,
                not path.available,
                path.name,
            )
        )
        self._paths = paths

    def _path_from(self, operation: Any, listed_available: bool) -> DatumTransformation:
        grids = _grids_from(operation)
        available = listed_available and all(grid.available for grid in grids)
        method = getattr(operation, "method_name", None)
        return DatumTransformation(
            name=str(getattr(operation, "name", "") or ""),
            description=_description(operation),
            accuracy=_accuracy(operation),
            definition=_operation_definition(operation),
            grids=grids,
            available=available,
            source=self.source,
            target=self.target,
            method=str(method or ""),
        )

    @property
    def paths(self) -> List[DatumTransformation]:
        """Available paths and paths blocked on a missing grid."""
        return list(self._paths)

    @property
    def missing_grids(self) -> List[TransformationGrid]:
        """Grids referenced by any path that are not installed."""
        grids: List[TransformationGrid] = []
        seen = set()
        for path in self._paths:
            for grid in path.missing_grids:
                if grid.short_name in seen:
                    continue
                seen.add(grid.short_name)
                grids.append(grid)
        return grids

    def offer_grid_download(self) -> List[TransformationGrid]:
        """Missing grids that can be downloaded, as offered by the dialog."""
        return [grid for grid in self.missing_grids if grid.direct_download]

    def download_missing_grids(
        self,
        directory: Optional[Union[str, Path]] = None,
        open_license: bool = True,
    ) -> List[TransformationGrid]:
        """
        Download missing grids and refresh the transformation paths.

        Args:
            directory: Optional directory to store the grids.
            open_license: When True, download only grids with an open license.

        Returns:
            The grids that were requested.
        """
        offered = [
            grid
            for grid in self.missing_grids
            if grid.direct_download and (grid.open_license or not open_license)
        ]
        if not offered or self._group is None:
            return []
        self._group.download_grids(directory=directory, open_license=open_license)
        self._load()
        return offered

    def _preferred(self) -> Optional[DatumTransformation]:
        for item in self.settings.preferred_transformations:
            if not (
                item.source.equivalent(self.source) and item.target.equivalent(self.target)
            ):
                continue
            for path in self._paths:
                if path.name == item.operation or path.description == item.operation:
                    return path
        return None

    @property
    def requires_choice(self) -> bool:
        """True when the dialog should open because several paths exist."""
        if not self.settings.ask_for_datum_transformation:
            return False
        if self._preferred() is not None:
            return False
        return len(self._paths) > 1

    def select(self, operation: str) -> DatumTransformation:
        """Select a path by its operation name or description."""
        for path in self._paths:
            if path.name == operation or path.description == operation:
                return path
        known = ", ".join(path.name for path in self._paths) or "none"
        raise ValueError(f"No datum transformation {operation!r}. Available: {known}")

    def resolve(self) -> Optional[DatumTransformation]:
        """
        Pick the preferred path, or the best available path.

        Raises:
            DatumTransformationRequired: When several paths exist and settings
                say to ask which one to use.
        """
        if not self._paths:
            return None
        preferred = self._preferred()
        if preferred is not None:
            return preferred
        if self.requires_choice:
            raise DatumTransformationRequired(self)
        available = [path for path in self._paths if path.available]
        if available:
            return available[0]
        return self._paths[0]


class Layer:
    """
    A map layer and the CRS on Layer Properties → Source.

    Showing the CRS reports the authority assigned to this layer. Overriding
    it changes that assignment. Coordinates are not reprojected; they are
    interpreted in the new CRS.
    """

    def __init__(self, name: str, crs: Optional[Union[CRS, str, int]] = None) -> None:
        self.name = name
        self._crs = None if crs is None else _as_crs(crs)
        self.crs_overridden = False

    @property
    def crs(self) -> Optional[CRS]:
        """CRS assigned to this layer."""
        return self._crs

    @property
    def crs_is_unknown(self) -> bool:
        """True when this layer has no CRS."""
        return self._crs is None

    def assign_crs(
        self,
        crs: Optional[Union[CRS, str, int]],
        *,
        override: bool = False,
    ) -> None:
        """Assign a CRS. ``override`` records a Layer Properties → Source change."""
        self._crs = None if crs is None else _as_crs(crs)
        if override:
            self.crs_overridden = True

    def override_crs(self, crs: Union[CRS, str, int]) -> None:
        """
        Override the CRS assigned to this layer.

        This does not reproject coordinates.
        """
        self.assign_crs(crs, override=True)

    def __repr__(self) -> str:
        return f"Layer({self.name!r}, crs={self._crs!r})"


class Project:
    """
    A project whose CRS is set from Project Properties → CRS.

    Layers stay in their own CRS. For display, each layer is reprojected to
    the project CRS on the fly. ``status_bar_crs`` is the status-bar CRS
    button. ``crs_mismatch`` is the Layers-panel icon for a layer whose CRS
    differs from the project.
    """

    def __init__(
        self,
        crs: Optional[Union[CRS, str, int]] = None,
        settings: Optional[CRSSettings] = None,
    ) -> None:
        self.settings = crs_settings if settings is None else settings
        self._layers: List[Layer] = []
        self._take_crs_from_first_layer = crs is None and bool(
            self.settings.new_project_uses_first_layer
        )
        if crs is None and self._take_crs_from_first_layer:
            self._crs: Optional[CRS] = None
        elif crs is None:
            self._crs = _as_crs(self.settings.default_crs)
        else:
            self._crs = _as_crs(crs)

    @property
    def crs(self) -> Optional[CRS]:
        """Project CRS. Layers are reprojected to it on the fly."""
        return self._crs

    @crs.setter
    def crs(self, value: Union[CRS, str, int]) -> None:
        self._crs = _as_crs(value)
        self._take_crs_from_first_layer = False

    @property
    def layers(self) -> List[Layer]:
        """Layers in this project."""
        return list(self._layers)

    @property
    def status_bar_crs(self) -> str:
        """Label for the status bar CRS button."""
        if self._crs is None:
            return "No CRS"
        if self._crs.authority and self._crs.code:
            return f"{self._crs.authority}:{self._crs.code}"
        return str(self._crs)

    def add_layer(
        self,
        layer: Layer,
        prompted_crs: Optional[Union[CRS, str, int]] = None,
    ) -> Layer:
        """
        Add a layer, applying the missing-CRS setting when the layer has none.

        Args:
            layer: Layer to add.
            prompted_crs: CRS chosen when the missing-CRS setting is to prompt.

        Raises:
            CRSPromptRequired: When a CRS is required and ``prompted_crs`` was
                not given.
            ValueError: If this layer object is already in the project.
        """
        if any(existing is layer for existing in self._layers):
            raise ValueError(f"Layer {layer.name!r} is already in the project")
        if (
            self._crs is None
            and self._take_crs_from_first_layer
            and layer.crs is not None
        ):
            self._crs = layer.crs
            self._take_crs_from_first_layer = False
        if layer.crs is None:
            assigned = self.settings.resolve_missing_layer_crs(self._crs, prompted_crs)
            if assigned is not None:
                layer.assign_crs(assigned)
        self._layers.append(layer)
        return layer

    def crs_mismatch(self, layer: Layer) -> bool:
        """True when the Layers panel should flag this layer as a different CRS."""
        if self._crs is None or layer.crs is None:
            return False
        return not self._crs.equivalent(layer.crs)

    def on_the_fly(self, layer: Layer) -> bool:
        """True when the layer is reprojected to the project CRS for display."""
        return self.crs_mismatch(layer)

    def mismatched_layers(self) -> List[Layer]:
        """Layers flagged with the CRS-mismatch icon."""
        return [layer for layer in self._layers if self.crs_mismatch(layer)]

    def display_transform(self, layer: Layer) -> Optional[DatumTransformation]:
        """
        Transformation used to draw this layer in the project CRS.

        Returns:
            None when the layer is already in the project CRS.

        Raises:
            DatumTransformationRequired: When several paths exist and settings
                say to ask which one to use.
        """
        if not self.on_the_fly(layer) or layer.crs is None or self._crs is None:
            return None
        return DatumTransformationDialog(layer.crs, self._crs, self.settings).resolve()

    def __repr__(self) -> str:
        return f"Project(crs={self.status_bar_crs!r}, layers={len(self._layers)})"


class LayoutMapGrid:
    """
    A print-layout map grid.

    The grid can be drawn in a CRS other than the map's own CRS.
    """

    def __init__(
        self,
        map_crs: Union[CRS, str, int],
        grid_crs: Optional[Union[CRS, str, int]] = None,
    ) -> None:
        self.map_crs = _as_crs(map_crs)
        self._grid_crs = None if grid_crs is None else _as_crs(grid_crs)

    @property
    def grid_crs(self) -> CRS:
        """CRS used to draw the grid. Defaults to the map CRS."""
        if self._grid_crs is None:
            return self.map_crs
        return self._grid_crs

    @grid_crs.setter
    def grid_crs(self, value: Union[CRS, str, int]) -> None:
        self._grid_crs = _as_crs(value)

    @property
    def draws_in_map_crs(self) -> bool:
        """True when the grid uses the same CRS as the map."""
        return self.grid_crs.equivalent(self.map_crs)

    def __repr__(self) -> str:
        return (
            f"LayoutMapGrid(map_crs={self.map_crs.identifier!r}, "
            f"grid_crs={self.grid_crs.identifier!r})"
        )


__all__ = [
    "AreaOfUse",
    "CRS",
    "CRSCatalogEntry",
    "CRSPromptRequired",
    "CRSSelector",
    "CRSSettings",
    "CustomProjection",
    "CustomProjectionRegistry",
    "DatumTransformation",
    "DatumTransformationDialog",
    "DatumTransformationRequired",
    "Layer",
    "LayoutMapGrid",
    "MissingCRSBehavior",
    "PreferredDatumTransformation",
    "Project",
    "TransformationGrid",
    "crs_settings",
    "custom_projections",
]
