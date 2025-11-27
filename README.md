# Programs for working on SAR dataset with oil slicks, look-alikes and other remarkable signatures (DARTIS\_2019)

This repository provides programs for the dataset submitted to PANGAEA, which was submitted along with the data description paper.

* Yang, Yi-Jie; Singha, Suman: Oil slicks, look-alikes and other remarkable SAR signatures in Sentinel-1 imagery in the Eastern Mediterranean Sea in 2019 [dataset]. PANGAEA, https://doi.pangaea.de/10.1594/PANGAEA.980773 (dataset in review)
* Yang, Y.-J., Singha, S., Goldman, R., and Schütte, F.: Dataset of Oil Slicks, Look-Alikes and Remarkable SAR Signatures Obtained from Sentinel-1 Data in the Eastern Mediterranean Sea, Earth Syst. Sci. Data Discuss. [preprint], https://doi.org/10.5194/essd-2025-208, in review, 2025. 

## Get Started with the Dataset

The dataset includes two subsets, *oil* and *no-oil* sets. The dataset downloaded from PANGAEA is without folder structure. 

The filename of image patches are referring to unique tags. However, image patches can also be put into folder structure and using `patch_name` (as in the data table) for easily knowing the date and time of the acquisitions by using the program `structure_dataset.py`. The `patch_name` follows the naming convention `MM_YYYYMMDD_HHMMSS_HHMMSS_PP_i`, where

* `MM` refers to satellite mission, in this dataset, all the data are from Sentinel-1 mission, `S1`;
* `YYYYMMDD` shows the date of the product;
* `HHMMSS` shows the start and stop time of the product;
* `PP` indicates the polarization mode (e.g., `VV`);
* `i` is a series of numbers assigned when generated the image patches.

> [!NOTE]
>
> `patch_name` shown in the data table (`DARTIS_2019.tab`) was assigned during the generation of the image patches. As the image patches from *oil* and *no-oil* sets were generated separately, in some rare cases the image patches may have the same names. Therefore, `structure_dataset.py` also generate folder structure for the image patches. 



Folder structure is designed as: 

```
oil/
    coast/
    water/
no_oil/
    coast/
        c0/
        c1/
        ..
    water/
        c0/
        c1/
        ...
```

If users wish to only generate the folder structure, then they can change the `rename_tf` to `False`. 



## Summary of the Programs

| Programs                     | Descriptions                                                 |
| ---------------------------- | ------------------------------------------------------------ |
| `dataset_toolbox.py`         | Called by other programs                                     |
| `structure_dataset.py`       | Structure dataset and rename the file with image patch name  |
| `img_land_mask.py`           | Generate land masks corresponding to the image patches and/or the masked out image patches <br/>(not used in the study) |
| `img_annotation.py`          | Generate annotation of the image patches as quicklook images |
| `calculate_perform.py`       | Compare the detections with the manual inspections from given dataset, <br/> then assign all the detections into the confusion matrix and return the counts of TP, FP, and FN. |
| `plot_bar_results_no_oil.py` | Plot bar charts for showing how detector performs for the *no-oil* set |

* `img_land_mask.py`: Land masking of the image patches should not be necessary in author's opinions. If users wish to apply land masks to the published dataset, they can generate the corresponding mask out scenes by loading the data tab (provided along with the dataset). 

  Some land masks sources: 

  * Open Street Map: https://osmdata.openstreetmap.de/data/land-polygons.html
  * GSHHG: Wessel, P. and Smith, W. H. F.: A global, self-consistent, hierarchical, high-resolution shoreline database, Journal of Geophysical Research Solid Earth, 101, 8741–8743, https://doi.org/10.1029/96JB00104, 1996. 
  * toddkarin/global-land-mask: Karin, T.: toddkarin/global-land-mask: Release of version 1.0.0, https://doi.org/10.5281/zenodo.4066722, 2020.

> [!WARNING]
>
> Please note that after applying landmasks with `img_land_mask.py`, the files will be renamed, which is not considered in the programs for performance evaluation. Users have to revise the programs on their own. 

## Performance Evaluation

For users who already have their own trained object detectors, they can compare their model performance with this study using the following programs. (see also: Subsect. 5.1 Performance Evaluation of the article on ESSD)

| Programs                     | Descriptions                                                 |
| ---------------------------- | ------------------------------------------------------------ |
| `calculate_perform.py`       | Compare the detections with the manual inspections from given dataset, <br/> then assign all the detections into the confusion matrix and return the counts of TP, FP, and FN. |
| `plot_bar_results_no_oil.py` | Plot bar charts for showing how detector performs for the *no-oil* set |

The program takes JSON file from YOLO detector. 

If users used other detection formats, please check the following part of the program `calculate_perform.py`:

* `load_yolo_result()`: instead using this function, users should write their own function to load the results into the `pandas.DataFrame` format.  

 	An example of the detections from yolo: 

  ```json
  [
    {
    "frame_id":1,
    "filename":"oil/coast/S1_20190101_034235_034350_VV_0.jpg", 
    "objects": [ 
        {"class_id":0, "name":"oil", "relative_coordinates":{"center_x":0.431610, "center_y":0.384405, "width":0.120989, "height":0.085739}, "confidence":0.678388}, 
        {"class_id":0, "name":"oil", "relative_coordinates":{"center_x":0.381560, "center_y":0.521030, "width":0.147851, "height":0.152155}, "confidence":0.475489}, 
        {"class_id":0, "name":"oil", "relative_coordinates":{"center_x":0.286878, "center_y":0.646113, "width":0.116899, "height":0.074922}, "confidence":0.372496}, 
        {"class_id":0, "name":"oil", "relative_coordinates":{"center_x":0.391556, "center_y":0.462170, "width":0.084175, "height":0.051132}, "confidence":0.371184}
      ] 
    }
  ]
  ```
  The function renames the field `objects` to `detected_objects` (for better knowing `objects` are from dataset or from detector) and generated additional fields, `detected_object_tf` and `dataset`. A glimpse of the dataframe: 

  |      | `frame_id` | `filename`                                     | `detect_objects`                                             | `detected_object_tf` | `dataset` |
  | ---- | ---------- | ---------------------------------------------- | ------------------------------------------------------------ | -------------------- | --------- |
  | 0    | 1          | `oil/coast/S1_20190101_034235_034350_VV_0.jpg` | `[{'class_id': 0, 'name': 'oil', 'relative_coordinates': {'center_x': 0.43161, 'center_y': 0.384405, 'width': 0.120989, 'height': 0.085739}, 'confidence': 0.678388}, {'class_id': 0, 'name': 'oil', 'relative_coordinates': {'center_x': 0.38156, 'center_y': 0.52103, 'width': 0.14785099999999998, 'height': 0.15215499999999998}, 'confidence': 0.475489}, {'class_id': 0, 'name': 'oil', 'relative_coordinates': {'center_x': 0.28687799999999997, 'center_y': 0.6461129999999999, 'width': 0.11689899999999999, 'height': 0.074922}, 'confidence': 0.372496}, {'class_id': 0, 'name': 'oil', 'relative_coordinates': {'center_x': 0.39155599999999996, 'center_y': 0.46216999999999997, 'width': 0.084175, 'height': 0.051132}, 'confidence': 0.37118399999999996}]` | `True`               | `oil`     |

* `confusion_matrix()`: in this function, the returned coordinates from yolo are coverted to image cooridinates, which are later saved as polygon (`shapely.geometry`) for calculating the intersection over union (IoU) with the manual inpsections from the given dataset. Please check this part of the function and revise into the format fitting the user's own usage: 
  ```python
  coord = dect_info["relative_coordinates"]
  # Convert yolo results to xmin, ymin, xmax, ymax
  info = {
      "xmin": (float(coord["center_x"])-float(coord["width"])/2)*img_width,
      "ymin": (float(coord["center_y"])-float(coord["height"])/2)*img_height,
      "xmax": (float(coord["center_x"])+float(coord["width"])/2)*img_width,
      "ymax": (float(coord["center_y"])+float(coord["height"])/2)*img_height
  }
  ```





