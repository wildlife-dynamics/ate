import os
import math
import logging
import ecoscope
import traceback
from enum import Enum
import geopandas as gpd
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname
from ecoscope_workflows_core.decorators import task
from pydantic import BaseModel, Field, field_validator
from ecoscope_workflows_core.annotations import AnyGeoDataFrame
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import ViewState
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import PointLayerStyle
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import LayerDefinition
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import LegendDefinition
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import PolygonLayerStyle
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import PolylineLayerStyle
from typing import Union, Dict, Optional, Literal, List, Annotated, TypedDict, Tuple
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import create_point_layer
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import create_polygon_layer
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import create_polyline_layer
from ecoscope_workflows_ext_ecoscope.tasks.analysis._time_density import CustomGridCellSize
from ecoscope_workflows_ext_ecoscope.tasks.analysis._create_meshgrid import create_meshgrid
from ecoscope_workflows_ext_ecoscope.tasks.analysis._calculate_feature_density import calculate_feature_density

logger = logging.getLogger(__name__)

class MapStyleConfig(BaseModel):
    styles: Dict[str, Dict] = Field(default_factory=dict)
    legend: Dict[str, List[str]] = Field(default_factory=dict)

class SupportedFormat(str, Enum):
    GPKG = ".gpkg"
    GEOJSON = ".geojson"
    SHP = ".shp"

SUPPORTED_FORMATS = [f.value for f in SupportedFormat]

class MapProcessingConfig(BaseModel):
    path: str = Field(..., description="Directory path to load geospatial files from")
    target_crs: Union[int, str] = Field(default=4326, description="Target CRS to convert maps to")
    recursive: bool = Field(default=False, description="Whether to walk folders recursively")

    @field_validator("path")
    @classmethod
    def validate_path_exists(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Invalid path: {v}")
        return v

class GeometrySummary(TypedDict):
    primary_type: Literal["Polygon", "Point", "LineString", "Other", "Mixed", "Line"]


@task
def remove_invalid_geometries(
    gdf: Annotated[AnyGeoDataFrame, Field(description="GeoDataFrame to filter for valid geometries.", exclude=True)],
) -> AnyGeoDataFrame:
    """
    Remove rows from the GeoDataFrame that contain null or empty geometries.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame that may contain invalid geometries.

    Returns:
        GeoDataFrame: Filtered GeoDataFrame containing only non-empty, non-null geometries.
    """
    return gdf.loc[(~gdf.geometry.isna()) & (~gdf.geometry.is_empty)]

@task
def detect_geometry_type(gdf: AnyGeoDataFrame) -> GeometrySummary:
    """
    Detect the dominant geometry type in a GeoDataFrame and count each type.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame whose geometries will be analyzed.

    Returns:
        GeometrySummary: Dict containing the detected geometry type and counts per geometry class.
    """
    geom_counts = gdf.geometry.geom_type.value_counts().to_dict()
    unique_types = list(geom_counts.keys())

    if len(unique_types) == 1:
        geom = unique_types[0]
        mapping = {
            "Polygon": "Polygon",
            "MultiPolygon": "Polygon",
            "Point": "Point",
            "MultiPoint": "Point",
            "LineString": "LineString",
            "MultiLineString": "LineString",
        }
        primary_type = mapping.get(geom, "Other")
    else:
        primary_type = "Mixed"

    return {"primary_type": primary_type, "counts": geom_counts}

@task
def load_geospatial_files(config: MapProcessingConfig) -> Dict[str, AnyGeoDataFrame]:
    """
    Load geospatial files from `config.path` and return a dict mapping
    relative file path -> cleaned GeoDataFrame (reprojected to target_crs).
    """
    base_path = Path(config.path)
    if not base_path.exists():
        raise FileNotFoundError(f"Path does not exist: {base_path!s}")

    target_crs = CRS.from_user_input(config.target_crs)

    loaded_files: Dict[str, AnyGeoDataFrame] = {}
    normalized_suffixes = {s.lower() if s.startswith(".") else f".{s.lower()}" for s in SUPPORTED_FORMATS}
    iterator = base_path.rglob("*") if config.recursive else base_path.iterdir()

    for p in iterator:
        try:
            if not p.is_file():
                continue

            if p.suffix.lower() not in normalized_suffixes:
                continue

            file_path = str(p)
            gdf = gpd.read_file(file_path)

            if gdf is None or gdf.empty:
                logger.info("Skipped empty or unreadable file: %s", file_path)
                continue

            if gdf.crs is None:
                logger.warning("File has no CRS, skipping reprojection: %s", file_path)
            else:
                try:
                    gdf_crs = CRS.from_user_input(gdf.crs)
                    if not gdf_crs == target_crs:
                        # geopandas accepts a CRS-like input string/object
                        gdf = gdf.to_crs(target_crs.to_string())
                except Exception as e:
                    logger.warning("Failed to normalize or compare CRS for %s: %s", file_path, e)

            cleaned = remove_invalid_geometries(gdf)
            key = str(p.relative_to(base_path))
            loaded_files[key] = cleaned

        except Exception:
            logger.error("Error processing %s", p, exc_info=True)

    logger.info("Loaded %d vector files from %s", len(loaded_files), base_path)
    return loaded_files


def clean_file_keys(file_dict: Dict[str, AnyGeoDataFrame]) -> Dict[str, AnyGeoDataFrame]:
    """
    Clean dictionary keys by removing file extensions and normalizing names.
    Args:
        file_dict: Dictionary mapping filenames to GeoDataFrames.
    Returns:
        A new dictionary with standardized, lowercase keys suitable for map layer identifiers.
    """
    def clean_key(key: str) -> str:
        for ext in SUPPORTED_FORMATS:
            if key.lower().endswith(ext):
                key = key[: -len(ext)]
                break

        key = re.sub(r'\band\b', '_', key, flags=re.IGNORECASE)
        key = re.sub(r'[^A-Za-z0-9_]+', '_', key)
        key = re.sub(r'_+', '_', key)  # collapse multiple underscores
        return key.strip('_').lower()
    return {clean_key(k): v for k, v in file_dict.items()}

@task
def create_layer_from_gdf(
    filename: str,
    gdf: AnyGeoDataFrame,
    style_config: MapStyleConfig,
    primary_type: str,
) -> Optional[LayerDefinition]:
    """
    Create an appropriately styled map layer from a GeoDataFrame.

    Args:
        filename: key matching an entry in style_config.styles (use cleaned keys).
        gdf: GeoDataFrame to render.
        style_config: MapStyleConfig containing style definitions and optional legend.
        primary_type: canonical geometry type (e.g. "Polygon", "Point", "LineString", "Other", "Mixed").

    Returns:
        A LayerDefinition (or subclass) or None if creation failed / unsupported.
    """
    canonical_map = {
        "MultiPolygon": "Polygon",
        "MultiPoint": "Point",
        "MultiLineString": "LineString",
    }
    primary = canonical_map.get(primary_type, primary_type)

    if filename not in style_config.styles:
        logger.warning("No style config for '%s'", filename)
        return None

    style_params: Dict[str, Any] = style_config.styles[filename]

    legend = None
    legend_cfg = getattr(style_config, "legend", None)
    if isinstance(legend_cfg, dict):
        labels = legend_cfg.get("labels")
        colors = legend_cfg.get("colors")
        if labels and colors and len(labels) == len(colors):
            legend = LegendDefinition(labels=labels, colors=colors)
        else:
            logger.debug("Skipping legend for '%s' due to missing or mismatched labels/colors", filename)

    try:
        gdf = remove_invalid_geometries(gdf)
    except Exception:
        pass

    try:
        if primary == "Polygon":
            logger.info("Creating polygon layer for '%s'", filename)
            return create_polygon_layer(gdf, layer_style=PolygonLayerStyle(**style_params), legend=legend)

        if primary == "Point":
            logger.info("Creating point layer for '%s'", filename)
            return create_point_layer(gdf, layer_style=PointLayerStyle(**style_params), legend=legend)

        if primary == "LineString":
            logger.info("Creating line layer for '%s'", filename)
            return create_polyline_layer(gdf, layer_style=PolylineLayerStyle(**style_params), legend=legend)

        logger.warning("Unsupported geometry type '%s' for file '%s'", primary_type, filename)
    except TypeError as te:
        logger.error("Invalid style params for '%s': %s", filename, te, exc_info=True)
    except Exception as e:
        logger.error("Error creating layer for '%s': %s", filename, e, exc_info=True)
    return None

@task
def create_map_layers(file_dict: Dict[str, AnyGeoDataFrame], style_config: MapStyleConfig) -> List[LayerDefinition]:
    """
    Create styled map layers from a dictionary of GeoDataFrames using the provided style config.

    Args:
        file_dict: Dictionary mapping filenames to GeoDataFrames.
        style_config: Object holding style definitions and legend config.
    Returns:
        A list of styled map layer objects.
    """
    layers: List[LayerDefinition] = []
    cleaned_files = clean_file_keys(file_dict)

    for filename, gdf in cleaned_files.items():
        try:
            try:
                gdf = remove_invalid_geometries(gdf)
            except Exception:
                gdf = gdf.loc[(~gdf.geometry.isna()) & (~gdf.geometry.is_empty)]

            geom_analysis = detect_geometry_type(gdf=gdf)
            gdf_geom_type = geom_analysis["primary_type"]
            gdf_counts = geom_analysis.get("counts", {})
            logger.info("%s geometry type: %s counts: %s", filename, gdf_geom_type, gdf_counts)
            layer = create_layer_from_gdf(filename, gdf, style_config, gdf_geom_type)

            if layer is not None:
                layers.append(layer)

        except Exception as e:
            logger.error("Error processing layer for '%s': %s", filename, e, exc_info=True)
    logger.info("Successfully created %d map layers", len(layers))
    return layers


# https://stackoverflow.com/questions/63787612/plotly-automatic-zooming-for-mapbox-maps
# Alternative approach using both dimensions for better fitting
def _zoom_from_bbox(minx, miny, maxx, maxy, map_width_px=800, map_height_px=600) -> float:
    """
    Calculate zoom level to fit bounding box in a given map size.
    This approach considers both dimensions for optimal fitting.

    Args:
        minx, miny, maxx, maxy (float): bounding box coordinates, must be in EPSG:4326
        map_width_px, map_height_px (int): target map dimensions in pixels

    Returns:
        float: zoom level that fits the bbox in the map
    """
    width_deg = abs(maxx - minx)
    height_deg = abs(maxy - miny)
    center_lat = (miny + maxy) / 2

    # Convert to km
    height_km = height_deg * 111.0
    width_km = width_deg * 111.0 * abs(math.cos(math.radians(center_lat)))

    world_width_km = 40075
    world_height_km = 40075

    zoom_for_width = math.log2(world_width_km * map_width_px / (512 * width_km))
    zoom_for_height = math.log2(world_height_km * map_height_px / (512 * height_km))

    zoom = min(zoom_for_width, zoom_for_height)
    zoom = round(max(0, min(20, zoom)), 2)
    return zoom

@task
def create_view_state_from_gdf(
    gdf: AnyGeoDataFrame, 
    pitch: int = 0, 
    bearing: int = 0,
) -> ViewState:

    if gdf.empty:
        raise ValueError("GeoDataFrame is empty. Cannot compute ViewState.")

    if gdf.crs is None or not gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = gdf.total_bounds
    center_lon = (minx + maxx) / 2.0
    center_lat = (miny + maxy) / 2.0
    zoom = _zoom_from_bbox(minx, miny, maxx, maxy)
    return ViewState(
        longitude=center_lon, 
        latitude=center_lat, 
        zoom=zoom, 
        pitch=pitch, 
        bearing=bearing
        )
@task
def download_land_dx(
    url: Annotated[str, Field(description="URL to retrieve the LandDx database")],
    path: Annotated[str, Field(description="Local path to save the LandDx database copy")],
    overwrite_existing: Annotated[bool, Field(default=False, description="Overwrite the existing file if it exists")],
    unzip: Annotated[bool, Field(default=True, description="Whether to unzip the downloaded file")],
) -> str:
    """
    Downloads the LandDx database from a given URL and saves it to the specified path.
    """
    # --- Normalize path if it starts with file:// ---
    if path.startswith("file://"):
        parsed = urlparse(path)
        path = url2pathname(parsed.path)

    # --- Ensure it's not a directory only ---
    if os.path.isdir(path) or path.endswith(os.sep):
        os.makedirs(path, exist_ok=True)
        filename = os.path.basename(urlparse(url).path) or "landdx.db"
        path = os.path.join(path, filename)

    ecoscope.io.utils.download_file(
        url=url,
        path=path,
        overwrite_existing=overwrite_existing,
        unzip=unzip,
    )
    return path
    
@task
def build_landdx_style_config(aoi_list: List[str], color_map: Dict[str, Tuple[int, int, int]]) -> MapStyleConfig:
    """
    Build a MapStyleConfig object with styles and legends for specified AOI types.

    Args:
        aoi_list (List[str]): List of area of interest (AOI) types.
        color_map (Dict[str, Tuple[int, int, int]]): Mapping from AOI type to RGB color.

    Returns:
        MapStyleConfig: A style configuration with defined styles and legends.
    """
    styles: Dict[str, Dict[str, object]] = {}
    legend: Dict[str, List[str]] = {"labels": [], "colors": []}

    for aoi in aoi_list:
        rgb = color_map.get(aoi, (128, 128, 128))  # fallback to gray
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)

        styles[aoi] = {
            "get_elevation": 1000,
            "get_fill_color": rgb,
            "opacity": 0.25,
            "get_line_width": 4.00,
            "get_width": 5,
            "get_line_color": rgb,
            "get_line_width": 2,
            "stroked":True
        }

        legend["labels"].append(aoi)
        legend["colors"].append(hex_color)
    return MapStyleConfig(styles=styles, legend=legend)

@task
def load_landdx_aoi(
    map_path: str,
    aoi: Optional[List[str]] = None,
) -> Optional[AnyGeoDataFrame]:

    # testing purposes
    print(f"Map path -->{map_path}")

    if map_path is None: 
        logger.error(f"Provided map path is empty")

    for root,_, files in os.walk(map_path):
        if "landDx.gpkg" in files:
            ldx_path = os.path.join(root, "landDx.gpkg")
            break
    
    print(f"landDx Path --> {ldx_path}")
    
    # Load and filter
    try:
        geodataframe = gpd.read_file(ldx_path, layer="landDx_polygons").set_index("globalid")
        
        if aoi is None or not aoi:
           print(f"Loaded landDx.gpkg — total features: {len(geodataframe)} (no filtering applied)")
        return geodataframe
        
        # Filter by AOI
        filtered = geodataframe[geodataframe["type"].isin(aoi)]
        print(
            f"Loaded landDx.gpkg — total: {len(geodataframe)}, "
            f"filtered by {aoi}: {len(filtered)}"
        )
        
        if filtered.empty:
            print(f"No features found matching AOI types: {aoi}")
        return filtered

    except FileNotFoundError as e:
        print(f"File not found: {ldx_path}")
        return None
    except KeyError as e:
        print(f"Required column missing: {e}")
        return None
    except Exception as e:
        print(f"Error loading or filtering landDx.gpkg: {e}")
        return None

@task
def annotate_gdf_dict_with_geometry_type(gdf_dict: Dict[str, AnyGeoDataFrame]) -> Dict[str, Dict[str, object]]:
    """
    Annotates each GeoDataFrame in the dictionary with its primary geometry type.

    Args:
        gdf_dict (Dict[str, gpd.GeoDataFrame]): Dictionary of GeoDataFrames.

    Returns:
        Dict[str, Dict]: Dictionary with keys preserved, and each value being a dict
                         containing the original GeoDataFrame and its geometry type.
    """
    result = {}

    # others to be added soon
    for name, gdf in gdf_dict.items():
        unique_geom_types = gdf.geometry.geom_type.unique()

        if len(unique_geom_types) == 1:
            geom_type = unique_geom_types[0]
            if "Polygon" in geom_type:
                primary_type = "Polygon"
            elif "Point" in geom_type:
                primary_type = "Point"
            elif "LineString" in geom_type:
                primary_type = "LineString"
            else:
                primary_type = "Other"
        else:
            primary_type = "Mixed"

        result[name] = {"gdf": gdf, "geometry_type": primary_type}
    return result

@task
def create_map_layers_from_annotated_dict(
    annotated_dict: Dict[str, Dict[str, object]], style_config: MapStyleConfig
) -> List[LayerDefinition]:
    """
    Create styled map layers from a dictionary of {name: {gdf, geometry_type}}.

    Args:
        annotated_dict: A dictionary where each value contains a GeoDataFrame and its geometry type.
        style_config: A MapStyleConfig object that defines styles for each name key.

    Returns:
        List of LayerDefinition objects for mapping.
    """
    # Convert dict to MapStyleConfig if needed
    if isinstance(style_config, dict):
        style_config = MapStyleConfig(**style_config)

    layers: List[LayerDefinition] = []

    for name, content in annotated_dict.items():
        print(f"Name: {name}")

        gdf = content.get("gdf")
        geometry_type = content.get("geometry_type")

        try:
            layer = create_layer_from_gdf(
                filename=name, 
                gdf=gdf, 
                style_config=style_config, 
                primary_type=geometry_type
                )

            if layer:
                layers.append(layer)
        except Exception as e:
            print(f"Error creating layer for '{name}': {e}")
            traceback.print_exc()

    print(f"Created {len(layers)} layers from annotated dict")
    return layers

@task
def combine_map_layers(
    static_layers: Annotated[
        Union[LayerDefinition, list[LayerDefinition]], Field(description="Static layers from local files or base maps.")
    ] = [],
    grouped_layers: Annotated[
        Union[LayerDefinition, list[LayerDefinition]],
        Field(description="Grouped layers generated from split/grouped data."),
    ] = [],
) -> list[LayerDefinition]:
    """
    Combine static and grouped map layers into a single list for rendering in `draw_ecomap`.
    """
    if not isinstance(static_layers, list):
        static_layers = [static_layers]
    if not isinstance(grouped_layers, list):
        grouped_layers = [grouped_layers]
    return static_layers + grouped_layers
