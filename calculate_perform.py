import dataset_toolbox

import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from shapely.geometry import Polygon
import shapely.ops as so

####################
### User's Input ###
####################
# Thresholds for IoU and confidence score (0-1)
thres_IoU = 0.5
thres_score = 0.5

# Detection output json file (yolo format)
yolo_output_json = "results.json"

# Directory - image patches
# dir_dataset = "dataset/"
dir_dataset = "dataset_structured"

# Data table downloaded from PANGAEA
data_tab = "DARTIS_2019.tab"

# Output directory for saving the performance evaluation summary
dir_output = "performance"

# If image patches were renamed to patch_name with the program structure_dataset.py, then True; otherwise, False
renamed_tf = True

# Printout some messages 
verbose = True

#############
### Func. ###
#############

def refresh_df_dataset(df_dataset):
    df_dataset["object_tf"] = df_dataset.subset.apply(lambda x: True if x[0]=="o" else False)
    list_info_keep = ["subset", "jpg_file", "xml_file", "patch_name", "patch_width", "patch_height", "object_tf"]
    
    df_dataset_oil = df_dataset.loc[df_dataset["object_tf"]==True].copy()
    df_dataset_no_oil = df_dataset.loc[df_dataset["object_tf"]==False].copy()
    df_dataset_no_oil = df_dataset_no_oil[list_info_keep]
    
    # Work on oil set
    df_dataset_oil["object"] = df_dataset_oil.apply(lambda row: \
        [row["obj_patchloc_xmin"], row["obj_patchloc_ymin"],
         row["obj_patchloc_xmax"], row["obj_patchloc_ymax"]], axis=1
    )
    
    # 
    df_dataset_oil_new = pd.DataFrame()
    list_jpg_files = np.unique(df_dataset_oil["jpg_file"].to_list()).tolist()
    for i, jpg_file_i in enumerate(list_jpg_files): 
        df_dataset_oil_part = df_dataset_oil[df_dataset_oil["jpg_file"]==jpg_file_i].copy()
        
        dict_info = {}
        for info_i in list_info_keep: 
            dict_info[info_i] = df_dataset_oil_part.iloc[0][info_i]
        
        df_tmp = pd.DataFrame(dict_info, index=[i])
        df_tmp["objects"] = ""
        df_tmp.at[i, "objects"] = df_dataset_oil_part["object"].to_list()
        
        
        df_dataset_oil_new = pd.concat([df_dataset_oil_new, df_tmp], ignore_index=True)
    
    df_dataset_oil_new["objects_num"] = df_dataset_oil_new.objects.apply(lambda x: len(x))

    df_dataset = pd.concat([df_dataset_oil_new, df_dataset_no_oil], ignore_index=True)
    
    return df_dataset
       
def load_yolo_result(result_json):
    df_result = pd.read_json(result_json)

    df_result["detect_object_tf"] = df_result.objects.apply(lambda x: True if len(x)>0 else False )
    df_result = df_result.rename(columns={"objects": "detect_objects"})
    df_result["dataset"] = df_result.filename.apply(lambda x: "no_oil" if  x.split("/")[-4]=="no_oil" else "oil")

    return df_result

def rm_obj(list_obj_info, thres_score):
    list_obj_info_refreshed = []
    for obj_info in list_obj_info:
        if obj_info["confidence"] >= thres_score:
            list_obj_info_refreshed.append(obj_info)
    
    return list_obj_info_refreshed

def remove_detections_by_score(df_match, thres_score):
    df_match["detect_objects_refresh"] = \
    df_match.detect_objects.apply(lambda x: rm_obj(x, thres_score))

    df_match["detect_object_tf"] = \
        df_match.detect_objects_refresh.apply(lambda x: True if len(x)>0 else False)

    df_match["detect_objects_num"] = \
        df_match.detect_objects_refresh.apply(lambda x: len(x))
    
    return df_match

def match_yolo_results_w_labels(df_dataset, df_result, renamed_tf=True):
    # 
    if renamed_tf:
        df_result["patch_name"] = df_result.filename.apply(lambda x: x.split("/")[-1].split(".")[0])
        # After renamed, some files might have same file names; therefore, oil set and no-oil set are matching to the df_dataset separately, and concate the df later.
        # - yolo results
        df_result_oil = df_result.loc[df_result["dataset"]=="oil"].copy()
        df_result_no_oil = df_result.loc[df_result["dataset"]=="no_oil"].copy()
        # - given dataset 
        df_dataset_oil = df_dataset.loc[df_dataset["object_tf"]==True].copy()
        df_dataset_no_oil = df_dataset.loc[df_dataset["object_tf"]==False].copy()

        # Put the Manual inspection (labeled dataset) information into the same dataframe
        df_match_oil = pd.merge(df_dataset_oil, df_result_oil, on="patch_name")
        df_match_no_oil = pd.merge(df_dataset_no_oil, df_result_no_oil, on="patch_name")

        # Concate
        df_match = pd.concat([df_match_oil, df_match_no_oil], ignore_index=True)
        
    else: 
        df_result["jpg_file"] = df_result.filename.apply(lambda x: x.split("/")[-1])
        df_match = pd.merge(df_dataset, df_result, on="jpg_file")

    return df_match

def confusion_matrix(df, task, outname, thres_IoU, thres_score, verbose=True):
    g = open("%s.txt"%outname, "w")
    if verbose: 
        printto = [sys.stdout, g]
    else:
        printto = [g]

    df["TP_detc"] = 0
    df["TP_gt"] = 0
    
    # - FP
    df["FP"] = df.apply(lambda row: row["detect_objects_num"] if np.logical_and(row["detect_object_tf"]==True, row["object_tf"]==False) else 0, axis=1)
    num_FP = np.sum(df["FP"].to_list())
    
    if task == "no-oil":
        num_FP_patch = len(df.loc[df["FP"]>0])

        for t in printto: 
            print("FP (image): %i"%num_FP_patch, file=t)
            print("FP (obj): %i"%num_FP, file=t)
    
    # - FN
    if task == "oil":
        df["FN"] = df.apply(lambda row:
            row["objects_num"] if np.logical_and(row["detect_object_tf"]==False, row["object_tf"]==True) else 0, axis=1
        )
        # Check 
        num_FP = np.sum(df["FN"].to_list())
        

    # - 'TN'
    if task == "no-oil":
        df["TN"] = df.apply(lambda row:
            1 if np.logical_and(row["detect_object_tf"]==False, row["object_tf"]==False) else 0, axis=1
        )

        num_TN = np.sum(df["TN"].to_list())

        for t in printto:
            print("TN (image): %i"%num_TN, file=t)
            print("------------------", file=t)
            print("Score Threshold: %.2f"%thres_score, file=t)
        
        # Save results
        df.to_csv("%s.csv"%outname)

        return

    # 'oil' set
    # -> Check the rest 
    df_part = df.loc[np.logical_and(df["FP"]==0, df["FN"]==0)].copy()
    
    # Check one by one
    for inx, row in df_part.iterrows():
        # Load detection --> df_detc_case 
        # df_detc_case = pd.DataFrame(columns=["center_x", "center_y", "width", "height"])
        df_detc_case = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax"])
        list_detc_info = row["detect_objects_refresh"]
        img_width = row["patch_width"]
        img_height = row["patch_height"]
        
        # Load yolo results 
        for dect_info in list_detc_info:
            coord = dect_info["relative_coordinates"]
            # Convert yolo results to xmin, ymin, xmax, ymax
            info = {
                "xmin": (float(coord["center_x"])-float(coord["width"])/2)*img_width,
                "ymin": (float(coord["center_y"])-float(coord["height"])/2)*img_height,
                "xmax": (float(coord["center_x"])+float(coord["width"])/2)*img_width,
                "ymax": (float(coord["center_y"])+float(coord["height"])/2)*img_height
            }
            
            if len(df_detc_case)==0:
                df_detc_case = pd.DataFrame(info, index=[0])
            else:
                df_detc_case = pd.concat([df_detc_case, pd.DataFrame(info, index=[0])], ignore_index=True)

        # Load given objects from the dataset --> df_dataset_case
        list_obj = row["objects"]
        df_dataset_case = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax"])
        for obj_coord in list_obj:
            info = {
                "xmin": obj_coord[0],
                "ymin": obj_coord[1],
                "xmax": obj_coord[2],
                "ymax": obj_coord[3]
            }
            if len(df_dataset_case)==0:
                df_dataset_case =  pd.DataFrame(info, index=[0])
            else: 
                df_dataset_case = pd.concat([df_dataset_case, pd.DataFrame(info, index=[0])], ignore_index=True)
        
        # Define bounding boxes as Polygon 
        df_detc_case["poly"] = df_detc_case.apply(
            lambda row: Polygon([
                (row.xmin, row.ymin), (row.xmin, row.ymax),
                (row.xmax, row.ymax), (row.xmax, row.ymin)
            ]), axis=1 
        )
        
        df_dataset_case["poly"] = df_dataset_case.apply(
            lambda row: Polygon([
                (row.xmin, row.ymin), (row.xmin, row.ymax),
                (row.xmax, row.ymax), (row.xmax, row.ymin)
            ]), axis=1 
        )
        
        # Check the intersection
        # Use detections as base df, check if the manual inspections have intersection with them 
        # - Check if intersect
        def _check_intersect(dect_poly, list_obj_polys):
            list_intersect_area = []
            for obj_poly in list_obj_polys:
                if dect_poly.intersects(obj_poly):
                    list_intersect_area.append(dect_poly.intersection(obj_poly).area)
                else:
                    list_intersect_area.append(0)
                # list_intersect_tf.append(dect_poly.intersects(obj_poly))
            return list_intersect_area
        
        df_detc_case["list_intersect_area"] = df_detc_case.apply(
            lambda row: _check_intersect(row["poly"], df_dataset_case["poly"].to_list()),
            axis=1
        )

        # - Calculate the IoU and check if pass the threshold (for the intersected ones)
        def _check_IoU(dect_poly, list_obj_polys, list_intersect_area, thres_IoU):
            list_IoU_pass = []
            # If not pass, then save as 0 
            for obj_poly, intersect_area in zip(list_obj_polys, list_intersect_area): 
                IoU_pass = False
                if intersect_area > 0: 
                    union_area = so.unary_union([dect_poly, obj_poly]).area
                    IoU = intersect_area/union_area
                    if IoU >= thres_IoU:
                        IoU_pass = True
                list_IoU_pass.append(IoU_pass)
        
            return list_IoU_pass
                
        df_detc_case["list_IoU_pass"] = df_detc_case.apply(
            lambda row: _check_IoU(row["poly"], df_dataset_case["poly"].to_list(), row["list_intersect_area"], thres_IoU),
            axis=1
        )

        # - Count TP_dect and FP
        df_detc_case[["TP_detc", "FP"]] = df_detc_case.apply(
            lambda row: [1, 0] if True in row["list_IoU_pass"] else [0, 1],
            axis=1, result_type="expand"
        )

        # - Refering to the manual inspections --> df_dataset_case
        def _obtain_TP_inx(list_IoU):
            array_IoU_pass_TF = np.array(list_IoU)
            array_IoU_01 = np.ones(np.shape(list_IoU))
            array_IoU_01 = array_IoU_01 * array_IoU_pass_TF

            # sum up in column direction  --> get how many of the passed one in certain index in df_dataset_case 
            list_TP = list(np.sum(array_IoU_01, axis=0))
            list_FN = [0 if i>0 else 1 for i in list_TP]

            return list_TP, list_FN


        list_TP_gt, list_FN = _obtain_TP_inx(df_detc_case["list_IoU_pass"].to_list())
        df_dataset_case["TP_gt"] = list_TP_gt
        df_dataset_case["FN"] = list_FN
        
        # Get the numbers of confusion matrix
        num_TP_detc = np.sum(df_detc_case["TP_detc"].to_list())
        num_FP = np.sum(df_detc_case["FP"].to_list())
        num_TP_gt = np.sum(df_dataset_case["TP_gt"].to_list())
        num_FN = np.sum(df_dataset_case["FN"].to_list())
        
        # Update the dataframe 
        df.at[inx, "FP"] = num_FP
        df.at[inx, "TP_detc"] = num_TP_detc
        df.at[inx, "FN"] = num_FN
        df.at[inx, "TP_gt"] = num_TP_gt
    
    
    # Save results
    df.to_csv("%s.csv"%outname)
    num_TP_detc = int(np.sum(df["TP_detc"].to_list()))
    num_TP_gt = int(np.sum(df["TP_gt"].to_list()))
    num_FP = int(np.sum(df["FP"].to_list()))
    num_FN = int(np.sum(df["FN"].to_list()))

    num_objs = int(np.sum(df["objects_num"].to_list()))
    num_detc = int(np.sum(df["detect_objects_num"].to_list()))
    
    
    for t in printto: 
        print("TP (detc): %i"%num_TP_detc, file=t)
        print("TP (gt): %i"%num_TP_gt, file=t)
        print("FP: %i"%num_FP, file=t)
        print("FN: %i"%num_FN, file=t)
        print("Number of objects (dataset): %i"%num_objs, file=t)
        print("Number of detected objects: %i"%num_detc, file=t)
        print("------------------", file=t)
        print("Score Threshold: %.2f"%thres_score, file=t)
        print("IoU Threshold: %.2f"%thres_IoU, file=t)
    

############
### Main ###
############

dir_dataset = Path(dir_dataset)
dir_output = Path(dir_output)
dir_output.mkdir(parents=True, exist_ok=True)

# Load dataset info 
# (object information are included in the data table)
dataset_info = dataset_toolbox.DataTab(data_tab)

# Refresh df_dataset
# - merge rows where pointing to the same jpg file, and combine the object information into list
df_dataset = refresh_df_dataset(dataset_info.df_dataset)
if verbose:
    print("Load dataset information (%s):"%data_tab)
    print(df_dataset)

# Load yolo results 
df_result = load_yolo_result(result_json=yolo_output_json)

# Remove detected objetcts with confidence score lower than the given threshold
df_result = remove_detections_by_score(df_result, thres_score=thres_score)

if verbose: 
    print("Load user given detections in yolo json format (%s):"%yolo_output_json)
    print("- Detections whose confidence scores are lower than the given threshold (%.2f) were removed"%thres_score)
    print(df_result)

# Match yolo results with manual inspections 
df_match = match_yolo_results_w_labels(df_dataset, df_result, renamed_tf=renamed_tf)
if verbose: 
    print("Match manual inspections and detections:")
    print(df_match)

# no-oil set -> make each group a separate subgroup 
df_match["subgroup"] = df_match.apply(lambda row: "%s-%s"%(row["subset"], row["jpg_file"].split("-")[2]) if row["dataset"]=="no_oil" else row["subset"], axis=1)

subgroup = list(np.unique(df_match["subgroup"].to_list()))

# Calculate confusion matrix (each subgroup)
for subgroup_i in subgroup: 
    if verbose: 
        print("====================")
        
    if subgroup_i[0] == "o":
        if verbose: print("oil set (%s)"%subgroup_i)
        task = "oil"
    elif subgroup_i[0] == "n":
        if verbose: print("no-oil set (%s)"%subgroup_i)
        task = "no-oil"

    df_part = df_match.loc[df_match["subgroup"]==subgroup_i].copy()
    
    confusion_matrix(df_part, task, "%s/%s_s%03i_i%03i"%(dir_output, subgroup_i, thres_score*100, thres_IoU*100), thres_IoU, thres_score, verbose=verbose)
    