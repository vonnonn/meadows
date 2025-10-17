from pathlib import Path
from shapely import BufferCapStyle, BufferJoinStyle, buffer
from shapely.geometry import shape, Point, Polygon
from shapely.ops import transform
from scipy.spatial import cKDTree
from rasterio.features import geometry_mask
from rasterio.mask import mask
from rasterio.io import MemoryFile
from datetime import datetime
import rasterio as rio
import rasterio.mask
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import json
import requests
import os


def proj_to_3857(poly, orig_crs):
    """
    Function for reprojecting a polygon from a shapefile of any CRS to Web Mercator (EPSG: 3857).
    The original polygon must have a CRS assigned.
    
    :param poly: shapely polygon for user area of interest (AOI)
    :param orig_crs: the original CRS for the shapefile. It is stripped out during import_shapefile_to_shapely() method
    """
    wgs84 = pyproj.CRS("EPSG:4326")
    web_mercator = pyproj.CRS("EPSG:3857")
    project_gcs = pyproj.Transformer.from_crs(orig_crs, wgs84, always_xy=True).transform
    project_wm = pyproj.Transformer.from_crs(orig_crs, web_mercator, always_xy=True).transform
    user_poly_proj4326 = transform(project_gcs, poly)
    user_poly_proj3857 = transform(project_wm, poly)
    return(user_poly_proj4326, user_poly_proj3857)

def gcs_to_proj(poly):
    """
    Function for reprojecting polygon shapely object from geographic coordinates (EPSG:4326) 
    to Web Mercator (EPSG: 3857)). 
    
    :param poly: shapely polygon for user area of interest (AOI)
    """
    wgs84 = pyproj.CRS("EPSG:4326")
    web_mercator = pyproj.CRS("EPSG:3857")
    project = pyproj.Transformer.from_crs(wgs84, web_mercator, always_xy=True).transform
    user_poly_proj3857 = transform(project, poly)
    return(user_poly_proj3857)

def import_shapefile_to_shapely(path):
    """
    Conversion of shapefile to shapely object.
    
    :param path: location of shapefile on user's local file system
    """
    shapefile_path = path
    gdf = gpd.read_file(shapefile_path)
    orig_crs = gdf.crs                   # this is the original CRS of the imported shapefile
    user_shp = gdf.loc[0, 'geometry']
    user_shp_epsg4326, user_shp_epsg3857 = proj_to_3857(user_shp, orig_crs)
    user_AOI = [[user_shp_epsg4326, user_shp_epsg3857]]
    return user_AOI
    
def proj_to_3857(poly, orig_crs):
    """
    Function for reprojecting a polygon from a shapefile of any CRS to Web Mercator (EPSG: 3857).
    The original polygon must have a CRS assigned.
    
    :param poly: shapely polygon for user area of interest (AOI)
    :param orig_crs: the original CRS for the shapefile. It is stripped out during import_shapefile_to_shapely() method
    """
    wgs84 = pyproj.CRS("EPSG:4326")
    web_mercator = pyproj.CRS("EPSG:3857")
    project_gcs = pyproj.Transformer.from_crs(orig_crs, wgs84, always_xy=True).transform
    project_wm = pyproj.Transformer.from_crs(orig_crs, web_mercator, always_xy=True).transform
    user_poly_proj4326 = transform(project_gcs, poly)
    user_poly_proj3857 = transform(project_wm, poly)
    return(user_poly_proj4326, user_poly_proj3857)


def gcs_to_proj(poly):
    """
    Function for reprojecting polygon shapely object from geographic coordinates (EPSG:4326) 
    to Web Mercator (EPSG: 3857)). 
    
    :param poly: shapely polygon for user area of interest (AOI)
    """
    wgs84 = pyproj.CRS("EPSG:4326")
    web_mercator = pyproj.CRS("EPSG:3857")
    project = pyproj.Transformer.from_crs(wgs84, web_mercator, always_xy=True).transform
    user_poly_proj3857 = transform(project, poly)
    return(user_poly_proj3857)

def import_shapefile_to_shapely(path):
    """
    Conversion of shapefile to shapely object.
    
    :param path: location of shapefile on user's local file system
    """
    shapefile_path = path
    gdf = gpd.read_file(shapefile_path)
    orig_crs = gdf.crs                   # this is the original CRS of the imported shapefile
    user_shp = gdf.loc[0, 'geometry']
    user_shp_epsg4326, user_shp_epsg3857 = proj_to_3857(user_shp, orig_crs)
    user_AOI = [[user_shp_epsg4326, user_shp_epsg3857]]
    return user_AOI

def usgs_3dep_datasets(shapefile_path):

    if os.path.exists(shapefile_path):
        user_AOI = import_shapefile_to_shapely(shapefile_path)
        print('Vector file loaded.')
    else:
        print(f"Vector file not found at {shapefile_path}, please check the path and try again.")
        user_AOI = None    

    # Check if resources.geojson exists, if not download it
    if not Path('../resources.geojson').exists():
        print("resources.geojson not found.")
        # Get GeoJSON file for 3DEP outlines from URL 

        print("Requesting 3DEP dataset polygons...")

        #request the boundaries from the Github repo and save locally.
        url = 'https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson'
        r = requests.get(url)
        with open('../resources.geojson', 'w') as f:
            f.write(r.content.decode("utf-8"))
        print("resources.geojson downloaded and saved to data folder.")
    else:
        print("resources.geojson exists in the data folder.")
    
    #make pandas dataframe and create pandas.Series objects for the names, urls, and number of points for each boundary.
    with open('../resources.geojson', 'r') as f:
        geojsons_3DEP = json.load(f)

    with open('../resources.geojson', 'r') as f:
        df = gpd.read_file(f)
        names = df['name']
        urls = df['url']
        num_points = df['count']

    #project the boundaries to EPSG 3857 (necessary for API call to AWS for 3DEP data)
    projected_geoms = []
    for geometry in df['geometry']:
            projected_geoms.append(gcs_to_proj(geometry))

    geometries_GCS = df['geometry']
    geometries_EPSG3857 = gpd.GeoSeries(projected_geoms)

    print('3DEP polygons loaded and projected to Web Mercator (EPSG:3857)')

    #buffering AOI by 10meters to ensure uncomprimised DTM
    buffed = buffer(user_AOI, 10, quad_segs=8, cap_style='round', join_style='round')
    print('AOI buffered by 10 meters')

    AOI_GCS = buffed[-1][0]
    AOI_EPSG3857 = buffed[-1][1]

    intersecting_polys = []

    for i,geom in enumerate(geometries_EPSG3857):
        if geom.intersects(AOI_EPSG3857):
            intersecting_polys.append((names[i], geometries_GCS[i], geometries_EPSG3857[i], urls[i], num_points[i]))

    if len(intersecting_polys) == 0:
        print("No intersecting 3DEP polygons found for the provided AOI.")
    else:     
        print(intersecting_polys)

    return intersecting_polys, AOI_EPSG3857.wkt

def pdal_pipeline(extent, usgs_3dep_dataset_names):

    #this is the basic pipeline which only accesses the 3DEP data
    readers = []
    for name in usgs_3dep_dataset_names:
        url = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{}/ept.json".format(name)
        reader = {
            "type": "readers.ept",
            "filename": str(url),
            "polygon": str(extent),
            "requests": 3,
            "resolution": 1
        }
        readers.append(reader)
        
    pointcloud_pipeline = {
            "pipeline":
                readers
    }

    filter_low_outliers_stage = {
            "type":"filters.elm",
            "cell":20.0,
            "class":18
    }       

    filter_noise = {
            "type":"filters.outlier",
            "method":"radius",
            "radius":2,
            "min_k":3,
    }

    filter_stage = {
            "type":"filters.range",
            "limits":"Classification[1:6],Classification[8:17],Classification[20:20]"
    }

    filter_interquartile_stage = {
            "type":"filters.iqr",
            "dimension":"Z",
            "k":3.5 #3.5 may not be aggressive enough
    }

    reprojection_stage = {
        "type":"filters.reprojection",
        "out_srs":"EPSG:{}".format(6339)
    }

    pointcloud_pipeline['pipeline'].append(filter_low_outliers_stage)
    #pointcloud_pipeline['pipeline'].append(filter_noise)
    pointcloud_pipeline['pipeline'].append(filter_stage)
    #pointcloud_pipeline['pipeline'].append(filter_interquartile_stage)
    pointcloud_pipeline['pipeline'].append(reprojection_stage)

    dsm_stage = {
                "type":"writers.gdal",
                "filename": 'dsm_temp.tif',
                "gdaldriver":"GTiff",
                #"nodata":-9999, #leave commented out for gdal_fillnodata
                "output_type":'idw',
                "resolution":float(0.6),
                "gdalopts":"COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES"
    }

    pointcloud_pipeline['pipeline'].append(dsm_stage)
    #setting noise removal after DSM for fuller canopy
    pointcloud_pipeline['pipeline'].append(filter_noise)
    pointcloud_pipeline['pipeline'].append(filter_interquartile_stage)

    remove_classes_stage = {
            "type":"filters.assign",
            #added ReturnNumber and NumberOfReturns to prevent error
            "value": ["Classification = 0","ReturnNumber = 1","NumberOfReturns = 1"]
    }
        
        #aggressive parameters from pdal_for_lidar.ipynb for Lacey Meadow
    classify_ground_stage = {
            "type":"filters.pmf",
            "cell_size":1,
            "max_window_size":33,
            "slope": 0.05,
            "initial_distance":0.05,
            "max_distance": 2
    }
        
    reclass_stage = {
            "type":"filters.range",
            "limits":"Classification[2:2]"
    }

    pointcloud_pipeline['pipeline'].append(remove_classes_stage)
    pointcloud_pipeline['pipeline'].append(classify_ground_stage)
    pointcloud_pipeline['pipeline'].append(reclass_stage)

    groundfilter_stage = {
                "type":"filters.range",
                "limits":"Classification[2:2]"
    }

    pointcloud_pipeline['pipeline'].append(groundfilter_stage)

    dem_stage = {
                "type":"writers.gdal",
                "filename":'dtm_temp.tif',
                "gdaldriver":"GTiff",
                #"nodata":-9999, #leave commented out for gdal_fillnodata
                "output_type":'idw',
                "resolution":float(0.6),
                "gdalopts":"COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES"
    }

    pointcloud_pipeline['pipeline'].append(dem_stage)

    return pointcloud_pipeline

def fill_nodata(raster_path, boundary_path):

    gdf = gpd.read_file(boundary_path)

    # Step 1: Read the full raster
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        profile = src.profile
        transform = src.transform
        nodata = src.nodata

    mask = geometry_mask(
        geometries=gdf.geometry,
        transform=transform,
        invert=True,
        out_shape=data.shape
    )

    if gdf.crs.srs.split(':')[1] != str(src.crs.to_epsg()):
        gdf = gdf.to_crs(src.crs.to_epsg())
    
    assert gdf.crs.srs.split(':')[1] == str(src.crs.to_epsg())

    # Step 3: Identify valid and nodata pixels
    valid_mask = (data != nodata)
    nodata_mask = ~valid_mask

    # Step 4: Limit interpolation to nodata pixels **inside** the mask
    interpolate_mask = nodata_mask & mask

    # Step 5: Prepare coordinates
    rows, cols = np.indices(data.shape)
    coords_valid = np.column_stack((rows[valid_mask], cols[valid_mask]))
    values_valid = data[valid_mask]
    coords_interp = np.column_stack((rows[interpolate_mask], cols[interpolate_mask]))

    # Step 6: IDW interpolation
    tree = cKDTree(coords_valid)

    def idw(tree, known_coords, known_values, unknown_coords, k=8, power=2):
        dists, idxs = tree.query(unknown_coords, k=k)
        weights = 1 / (dists ** power + 1e-12)
        weighted_vals = np.sum(weights * known_values[idxs], axis=1)
        return weighted_vals / np.sum(weights, axis=1)

    interpolated_values = idw(tree, coords_valid, values_valid, coords_interp)


    # Step 7: Fill interpolated values into a copy of the original data
    data_filled = data.copy()
    data_filled[interpolate_mask] = interpolated_values

    # Step 8: Write the output raster
    profile.update(dtype=rasterio.float32)
    outfile = raster_path.split('.')[0] + '_filled.tif'

    with rasterio.open(outfile, "w", **profile) as dst:
        dst.write(data_filled.astype(rasterio.float32), 1)

    return f"Filled {raster_path} and saved to {outfile}"

def clip_and_rename_raster(input_raster, gdf):
    # Read and clip raster
    with rasterio.open(input_raster) as src:
        out_image, out_transform = mask(src, gdf.geometry, crop=True)
        out_meta = src.meta.copy()
    
    # Update metadata
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })
    
    # Create output filename with timestamp
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_filename = f"{input_raster[:3]}_clipped.tif"
    
    # Write clipped raster
    with rasterio.open(out_filename, "w", **out_meta) as dest:
        dest.write(out_image)

    return f"Saved as: {out_filename}"

# Canopy Height Model (CHM) calculation
def chm(dsm_path, dtm_path, gdf):
    
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1)
    with rasterio.open(dtm_path) as src:
        dtm = src.read(1)
        out_meta = src.meta.copy()
    chm = dsm - dtm
    chm[chm < 0] = 0

    # Write CHM to memory
    with MemoryFile() as memfile:
        with memfile.open(**out_meta) as dataset:
            dataset.write(chm.astype(rasterio.float32), 1)
            
            # Clip CHM with gdf.geometry
            out_image, out_transform = mask(dataset, gdf.geometry, crop=True, nodata=np.nan)
            out_meta_clipped = dataset.meta.copy()
            out_meta_clipped.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": np.nan
            })

    # Write clipped CHM to disk
    with rasterio.open("chm_clipped.tif", "w", **out_meta_clipped) as dest:
        dest.write(out_image)
    return f"Saved as: chm_clipped.tif"