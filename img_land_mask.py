import dataset_toolbox

from pathlib import Path
from PIL import Image
import rasterio.features
import rasterio.transform
from rasterio.control import GroundControlPoint
import numpy as np
import geopandas as gpd


####################
### User's Input ###
####################

# Directory - image patches
dir_dataset = "dataset"
# dir_dataset = "dataset_structured"

# If applied 'structure_dataset.py' for structured dataset, please set to True
structured_tf = False

# Data table downloaded from PANGAEA
data_tab = "DARTIS_2019.tab"

# Land polygons
landpolygons_file = "land_polygons.shp"

# Subsets applying land masking
select_subsets = ["nc", "oc"]

# Output directory for land masks or masked out image patches 
dir_output = "patches_masked"

# Create land masks: True / False
create_land_mask_tf = True
# Create masked out image patches 
create_masked_patch_tf = True

# Printout some messages 
verbose = True

#############
### Func. ###
#############

def land_masking(row, dir_output, create_mask=False, create_patch=False, verbose=False):
    if verbose: print(row.tag)

    # Get image patch information
    image_patch = row.path_jpg
    image_arr =np.array(Image.open(image_patch))
    height, width = image_arr.shape

    # Image patch corner coordinates
    patch_ul_lon = row.patch_ul_lon 
    patch_ul_lat = row.patch_ul_lat 
    patch_ur_lon = row.patch_ur_lon 
    patch_ur_lat = row.patch_ur_lat 
    patch_br_lon = row.patch_br_lon 
    patch_br_lat = row.patch_br_lat 
    patch_bl_lon = row.patch_bl_lon 
    patch_bl_lat = row.patch_bl_lat
    
    # Define GCPs for corner points
    gcps = [
        GroundControlPoint(-0.5, -0.5, patch_ul_lon, patch_ul_lat),
        GroundControlPoint(-0.5, width-0.5, patch_ur_lon, patch_ur_lat),
        GroundControlPoint(height-0.5, -0.5, patch_bl_lon, patch_bl_lat),
        GroundControlPoint(height-0.5, width-0.5, patch_br_lon, patch_br_lat),
    ]

    # Create bounding box for reading shapefile
    xmin, xmax = min([g.x for g in gcps]), max([g.x for g in gcps])
    ymin, ymax = min([g.y for g in gcps]), max([g.y for g in gcps])

    # Read relevant polygons from shapefile
    landpolygons = gpd.read_file(landpolygons_file,
                                    bbox=(xmin, ymin, xmax, ymax))

    # Transform land polygons to image geometry
    transformer = rasterio.transform.GCPTransformer(gcps, tps=True)

    def points_transformer(points):
        rows, cols = transformer.rowcol(
            points[:, 0], points[:, 1], op=lambda x: x)
        return np.column_stack((cols, rows))

    transformed = landpolygons.geometry.transform(points_transformer)

    landmask = rasterio.features.rasterize(transformed, (height, width), all_touched=True).astype(bool)

    # Output the land mask
    if create_land_mask_tf: 
        outfile = dir_output.joinpath("%s_mask.jpg"%row.path_jpg.stem)
        Image.fromarray(landmask).save(outfile)
        if verbose: print("    --> %s"%outfile)

    # Output the masked image
    if create_masked_patch_tf:
        outfile = dir_output.joinpath("%s_maskout.jpg"%row.path_jpg.stem)
        image_arr[landmask] = 0
        Image.fromarray(image_arr).save(outfile)
        if verbose: print("    --> %s"%outfile)

############
### Main ###
############

dir_dataset = Path(dir_dataset)
dir_output = Path(dir_output)
dir_output.mkdir(parents=True, exist_ok=True)
landpolygons_file = Path(landpolygons_file)

# Load dataset info
dataset_info = dataset_toolbox.DataTab(data_tab)

# Load only the subsets with land areas
df_dataset = dataset_info.extract_partial_dataset(select_subsets=select_subsets)

# For structured dataset
if structured_tf: 
    df_dataset = dataset_info.refresh2folder_structure(dir_dataset_new=dir_dataset, col_path_jpg_new="path_jpg", col_path_xml_new="path_xml")
else: 
    # Obtain the image paths
    df_dataset = dataset_info.load_jpg_path(dir_dataset=dir_dataset)

# Apply land masking 
df_dataset.apply(lambda row: land_masking(row, dir_output, create_mask=create_land_mask_tf, create_patch=create_masked_patch_tf, verbose=verbose), axis=1)
