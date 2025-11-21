import dataset_toolbox

from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

####################
### User's Input ###
####################

# Directory - image patches
dir_dataset = "dataset/"
# dir_dataset = "dataset_structured/"

# If applied 'structure_dataset.py' for structured dataset, please set to True
structured_tf = False

# Data table downloaded from PANGAEA
data_tab = "DARTIS_2019.tab"

# Subsets for annotation
select_subsets = ["ow", "oc"]

# Output directory for annotated image patches 
dir_output = "patches_annotated"

# Printout some messages 
verbose = True

#############
### Func. ###
#############

def load_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Save object positions
    list_bbox = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        list_bbox.append([
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text)
        ])
    return list_bbox
        
def plot_patch_label(img_file, img_out, xml_file, verbose=False):
    # Load annotations and image patch
    list_bbox = load_xml(xml_file)
    img = Image.open(img_file)
    # Plot annotations
    img_ann = ImageDraw.Draw(img)
    for bbox in list_bbox:
        img_ann.rectangle(bbox,
        fill=None, outline=255,
        width=2)
    
    # Resize image
    w, h = img.size
    if w > 640 or h > 640:
        img = img.resize((640,640), Image.LANCZOS)

    # Save image
    img.save(img_out)

    if verbose: print(img_out)

############
### Main ###
############

dir_dataset = Path(dir_dataset)
dir_output = Path(dir_output)
dir_output.mkdir(parents=True, exist_ok=True)

# Load dataset info
dataset_info = dataset_toolbox.DataTab(data_tab)

# Load only the subsets with oil slicks inside
df_dataset = dataset_info.extract_partial_dataset(select_subsets=select_subsets)

# For structured dataset
if structured_tf: 
    df_dataset = dataset_info.refresh2folder_structure(dir_dataset_new=dir_dataset, col_path_jpg_new="path_jpg", col_path_xml_new="path_xml")

else:
    # Obtain the paths of image patches and the xml files 
    df_dataset = dataset_info.load_jpg_path(dir_dataset=dir_dataset)
    df_dataset = dataset_info.load_xml_path(dir_dataset=dir_dataset)

# Plot annotation 
df_dataset.apply(lambda row: plot_patch_label(img_file=row.path_jpg, img_out=dir_output.joinpath("%s_label.jpg"%row.path_jpg.stem), xml_file=row.path_xml, verbose=verbose) if row.path_xml.is_file() else None, axis=1)
