import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

####################
### User's Input ###
####################

# Directory of the output for performance calculation (calculate_perform.py)
dir_output="performance"

# Thresholds for IoU and confidence score (0-1)
thres_IoU = 0.5
thres_score = 0.5

# Output figure format
figure_format = "eps"

#############
### Func. ###
#############

def read_result(txt):
    with open(txt, "r") as f:
        info = f.read().splitlines()
        for tmp in info:
            if "FP (image)" in tmp:
                FP = int(tmp.split(":")[-1])
            elif "TN (image)" in tmp:
                TN = int(tmp.split(":")[-1])
    
    return TN, FP

############
### Main ###
############

dir_output = Path(dir_output)
thres_keyword = "s%03i_i%03i"%(thres_score*100, thres_IoU*100)

dict_subgroup = {
    "nw-00":0, "nw-01":1, "nw-02":2, "nw-03":3, "nw-04":4,
    "nw-05":5, "nw-06":6, "nw-07":7, "nw-08":8, "nw-09":9,
    "nw-10":10, "nw-11":11, "nc-00":12, "nc-01":13, "nc-02":14,
    "nc-03":15, "nc-04":16
}


# Load performance summary txt file output from calculate_performance.py
list_group_ref = []
list_TN = []
list_FP = []
list_total = []

for subgroup_i, inx in dict_subgroup.items():
    performance_summary_txt = dir_output.joinpath("%s_%s.txt"%(subgroup_i, thres_keyword))
    if not performance_summary_txt.is_file():
        print("The file doesn't exist: %s."%str(performance_summary_txt))

    # Read the summary file
    TN, FP = read_result(performance_summary_txt)
    list_group_ref.append(inx)
    list_TN.append(TN)
    list_FP.append(FP)
    list_total.append(FP+TN)

# Plot 
fig, ax = plt.subplots(figsize=(6,5.5))
ax.set_yticks(list(dict_subgroup.values()))
ax.set_yticklabels(list(dict_subgroup.keys()))

ax.barh(list_group_ref, list_total, color="#e41a1c")    
ax.barh(list_group_ref, list_TN, color="#377eb8")

for TN, FP, count in zip(list_TN, list_FP, list_group_ref): 
    if FP>=50: 
        plt.text(FP/2+TN, count, FP, horizontalalignment="left",
    verticalalignment="center_baseline")
    else: 
        plt.text(FP+TN, count, FP, horizontalalignment="left",
        verticalalignment="center_baseline")

    if TN < 15:
        plt.text(TN/3, count, TN, horizontalalignment="left",
    verticalalignment="center_baseline")
    else:
        plt.text(TN/2, count, TN, horizontalalignment="left",
        verticalalignment="center_baseline")

# ax.set_ylabel()
ax.set_xlabel("Number of image patches")

# Save plot
plt.gca().invert_yaxis()
figure_output = "%s/performance_bar__no_oil__%s.%s"%(dir_output, thres_keyword, figure_format)
plt.savefig(figure_output, bbox_inches='tight', pad_inches=0.1)
plt.close()
print("Figure saved: %s"%figure_output)
