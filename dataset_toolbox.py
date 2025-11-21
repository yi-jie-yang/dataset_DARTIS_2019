from pathlib import Path
import re
import numpy as np
import pandas as pd 


class DataTab:
    def __init__(self, data_tab):
        self.data_tab = data_tab
        self.df_dataset = self._load_data_tab()

    def _read_data_tab(self):
        with open(self.data_tab, "r") as f:
            # lines = f.readlines()
            lines = f.read().splitlines()
            header_tf = True 
            ignore_tf = False
            comment_status = 0
            for line_i in lines:
                info = line_i.split()
                
                # Ignore the commented out information /* */ 
                if '/*' in info:
                    comment_status = 1
                    ignore_tf = True
                elif '*/' in info: 
                    comment_status = 0
                    ignore_tf = True 
                
                if ignore_tf:
                    if comment_status == 0: 
                        ignore_tf = False
                    continue
            
                # Save information to pandas dataframe
                info = line_i.split("\t")
                # Get the header
                if header_tf:
                    heading = []
                    for tmp in info: 
                        col_name = re.findall(r"\((.*?)\)", tmp)[0]
                        if ';' in col_name: 
                            col_name = col_name.split(';')[0]
                        heading.append(col_name)
                    df = pd.DataFrame(columns=heading)
                    header_tf = False
                # Save information into pandas dataframe
                else:
                    df_tmp = pd.DataFrame([info], columns=heading)
                    df = pd.concat([df, df_tmp], ignore_index=True)
        return df

    def _refresh_data_type(self, df):
        heading = df.columns
        for col_name in heading:
            if np.logical_or(
                np.logical_or('lon' in col_name, 'lat' in col_name), 
                np.logical_or('loc' in col_name, 'size' in col_name)): 
                df[col_name] = pd.to_numeric(df[col_name])
            elif col_name in ["patch_width", "patch_height"]:
                df[col_name] = pd.to_numeric(df[col_name])
            else:
                df[col_name] = df[col_name].astype(str)
        return df 
    
    def _load_data_tab(self):
        # Load data tab
        df = self._read_data_tab()
        # Refresh dtype 
        df = self._refresh_data_type(df)
        
        return df

    def extract_partial_dataset(self, select_subsets=[]):
        df = pd.DataFrame()
        list_subsets = np.unique(self.df_dataset["subset"].tolist())
        for subset_i in select_subsets:
            if subset_i not in list_subsets:
                print("Subset '%s' doesn't exist."%subset_i)
                continue
            df_part = self.df_dataset.loc[self.df_dataset["subset"]==subset_i].copy()
            df =  pd.concat([df, df_part], ignore_index=True)
        
        self.df_dataset = df
        return df 
    
    def load_jpg_path(self, dir_dataset): 
        self.df_dataset["path_jpg"] =  self.df_dataset.jpg_file.apply(lambda x: dir_dataset.joinpath(x) if dir_dataset.joinpath(x).is_file() else Path())

        df_no_file = self.df_dataset.loc[self.df_dataset["path_jpg"]==Path()].copy()
        df_no_file.jpg_file.apply(lambda x: print("File <%s> doesn't exist in the directory <%s>."%(x, dir_dataset)))

        if len(df_no_file) > 0:
            print(">> Please check again the dataset directory before further execution.")
            exit()

        return self.df_dataset

    def load_xml_path(self, dir_dataset): 
        self.df_dataset["path_xml"] =  self.df_dataset.xml_file.apply(lambda x: dir_dataset.joinpath(x) if dir_dataset.joinpath(x).is_file() else Path())

        return self.df_dataset

    def refresh2folder_structure(self, dir_dataset_new, col_path_jpg_new="path_jpg_new", col_path_xml_new="path_xml_new", rename_tf=True):
        dict_subset_ref = {
            "oc": "oil/coast",
            "ow": "oil/water",
            "nc": "no_oil/coast",
            "nw": "no_oil/water"
        }

        # Rename to patch_name or keep it to the current name
        if rename_tf:
            patch_name_new = "patch_name"
        else:
            patch_name_new = "jpg_name"
            self.df_dataset[patch_name_new] = self.df_dataset.jpg_file.apply(lambda x: x.split(".")[0])

        # Define new jpg file location
        self.df_dataset[col_path_jpg_new] = self.df_dataset.apply(lambda row: 
            dir_dataset_new.joinpath("%s/%s.jpg"%(dict_subset_ref[row["subset"]], row[patch_name_new])) \
                if row["subset"][0]=="o" \
                else dir_dataset_new.joinpath("%s/c%i/%s.jpg"%(
                        dict_subset_ref[row["subset"]], 
                        int(row["jpg_file"].split("-")[2]), 
                        row[patch_name_new]
                    )
            ), axis=1
        )

        # Define new xml file location
        self.df_dataset[col_path_xml_new] = self.df_dataset.apply(lambda row: 
            dir_dataset_new.joinpath("%s/%s.xml"%(dict_subset_ref[row["subset"]], row[patch_name_new])) \
            if row["xml_file"].split(".")[-1]=="xml" \
                else Path(), axis=1
        )
                
        # Save all the subfolder paths
        self.df_dataset["dir_jpg"] = self.df_dataset[col_path_jpg_new].apply(lambda x: Path(x).resolve().parent)
        list_subfolders = np.unique(self.df_dataset["dir_jpg"].to_list()).tolist()
        
        # Remove temperary column
        if not rename_tf: 
            self.df_dataset = self.df_dataset.drop(columns=[patch_name_new])
        self.df_dataset = self.df_dataset.drop(columns=["dir_jpg"])
            
        return self.df_dataset, list_subfolders