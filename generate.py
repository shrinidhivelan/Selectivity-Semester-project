import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score 
import os
import ast
from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget
from nwbwidgets.allen import AllenRasterWidget
from nwbwidgets.allen import AllenPSTHWidget
import pandas as pd
import numpy as np
from pathlib import Path
import ast
from helpers import *
from plot import *
from AUC import *


def generate_mice_data(folder_path):

    mouse_names = []
    mice_data = []

    for filepath in folder_path.glob("*.nwb"):
        
        if not filepath.name.startswith("._"):  

            passive_active_data = []
            print(f"Processing file: {filepath.name}")
    
            mouse_name = filepath.name[:-4]  
            mouse_names.append(mouse_name)


            io = NWBHDF5IO(str(filepath), 'r') 
            nwbfile = io.read()

            units = nwbfile.units.to_dataframe()
            trials = nwbfile.trials.to_dataframe()

            # Preprocessing 
            filtered_units = units[(units['bc_label'] == 'good') & (units['ccf_acronym'].str.contains('[A-Z]'))]
            data_total = spike_detect(nwbfile, filtered_units, trials, 0.2, 0.2)
        
            data_total['pre_time'] = 0.2 #pre_time
            data_total['post_time'] = 0.2 #post_time
            data_total['mouse_id'] = mouse_name #mouse_id

            ### This column causes problems (with formatting)
            data_total = data_total.drop(columns=['electrode_group'])


            folder = f'Data/{mouse_name}'
            os.makedirs(folder, exist_ok=True)
            data_total.to_parquet(f'{folder}/{mouse_name}_Selectivity_Dataframe2.parquet', index=False)

            mice_data.append(data_total)

    # Combine all event DataFrames into a single DataFrame
    df_combined = pd.concat(mice_data).reset_index(drop=True)
    os.makedirs('Data/Overall', exist_ok=True)
    df_combined.to_parquet(f'Data/Overall/Mice_Selectivity_Dataframe2.parquet', index=False)
    return mouse_names


def AUC_generate(mouse_names = [], save_files = False, visualize = False, nb_neurons = 100, pre_vs_post_visualization = False, start = 0.2, stop = 0.2):
    mice_data = []
    df_combined = []

    for i, mouse_name in enumerate(mouse_names):

        df = pd.read_parquet(f'Data/{mouse_name}/{mouse_name}_Selectivity_Dataframe2.parquet')
        #mice_dfs[i] #
        

        # Check whether we want to save or visualize the files:
        if (save_files): 
            print("Starting to save files!")
            save_overall_auc(df, mouse_name)
        if (visualize): visualize_auc(df, nb_neurons)

        print(f"Starting process for Mouse {i+1}/{len(mouse_names)} {mouse_name}")
        
        # AUC :
        for type in ['whisker', 'auditory', 'wh_vs_aud', 'spontaneous_licks']:
            if type == 'spontaneous_licks':
                df['spontaneous_licks_AUC'] = df.apply(lambda row: compute_AUC_2(row, type, context), axis=1)
                df['Transformed spontaneous_licks_AUC'] = df.apply(lambda row: 2*row[f'spontaneous_licks_AUC']-1, axis=1)

            else:
                contexts = ['passive', 'active']
                
            for context in contexts:
                df[f'{type}_{context}_AUC'] = df.apply(lambda row: compute_AUC_2(row, type, context), axis=1)
                df[f'Transformed {type}_{context}_AUC'] = df.apply(lambda row: 2*row[f'{type}_{context}_AUC']-1, axis=1)

        ## Bootstrapping process :
        print("Starting bootstrapping process...")
        new_df = bootstrapping(df)

        print('Pivotting table!')
        combined_df = create_combined_df_v6(new_df)
        combined_df['mouse_id'] = mouse_name
        combined_df['pre_time'] = start
        combined_df['post_time'] = stop


        ### save the parquet file :
        combined_df.to_parquet(f'Data/{mouse_name}/{mouse_name}_AUC_Selectivity2.parquet', index=False)
        print(f"Process finished for Mouse {i+1}/{len(mouse_names)}!")
        mice_data.append(combined_df)
    

     # Combine all event DataFrames into a single DataFrame
    df_combined = pd.concat(mice_data).reset_index(drop=True)
    os.makedirs('Data/Overall', exist_ok=True)
    df_combined.to_parquet(f'Data/Overall/Mice_AUC_Selectivity2.parquet', index=False)
    

    return df_combined

    

