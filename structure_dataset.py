import dataset_toolbox

from pathlib import Path
import shutil

####################
### User's Input ###
####################
# Directory - image patches
dir_dataset = "dataset/"

# Data table downloaded from PANGAEA
data_tab = "DARTIS_2019.tab"

# Output directory for new dataset folder
dir_output = "dataset_structured"

# Rename the file 
rename_tf = True 

# Printout some messages 
verbose = True

############
### Main ###
############
dict_subset_ref = {
    "oc": "oil/coast",
    "ow": "oil/water",
    "nc": "no_oil/coast",
    "nw": "no_oil/water"
}

dir_dataset = Path(dir_dataset)
dir_output = Path(dir_output)
dir_output.mkdir(parents=True, exist_ok=True)

# Load dataset info
dataset_info = dataset_toolbox.DataTab(data_tab)

# Load jpg file location
dataset_info.load_jpg_path(dir_dataset=dir_dataset)

# Load xml file location
dataset_info.load_xml_path(dir_dataset=dir_dataset)

# Refresh to folder structure
df_dataset, list_subfolders = dataset_info.refresh2folder_structure(dir_dataset_new=dir_output, rename_tf=rename_tf)

# Generate folder structure
if verbose: print("Image patches will be copied in to the following folders:")
for subfolder_i in list_subfolders:
    if verbose: print("    %s"%subfolder_i)
    subfolder_i.mkdir(parents=True, exist_ok=True)


# Copy to new folder
# - jpg files
df_dataset.apply(lambda row: shutil.copy(row.path_jpg, row.path_jpg_new), axis=1)

# - xml files
df_dataset.apply(lambda row: shutil.copy(row.path_xml, row.path_xml_new) if row.path_xml.is_file() else None, axis=1)

