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

import ast
from helpers import *
from plot import *
from AUC import *


def generate_mice_data(mouse_names):

    mice_data = []
    for mouse in mouse_names:
    
        # Load the file with the mice data
        io = NWBHDF5IO(f'/Users/shrinidhivelan/Desktop/Selectivity-Semester-project/{mouse}.nwb', mode='r')
        nwbfile = io.read()
        # To do : create a list of mouse names, to be used for the .py file.
        mouse_name = mouse

        units = nwbfile.units.to_dataframe()
        trials = nwbfile.trials.to_dataframe()

        # Preprocessing s
        filtered_units = units[(units['bc_label'] == 'good') & (units['ccf_acronym'].str.contains('[A-Z]'))]


        # Some chosen columns
        cons_columns  = ["cluster_id", "firing_rate", "ccf_acronym", "ccf_name", "ccf_parent_acronym", "ccf_parent_name", "spike_times"]#, "lick_flag", "lick_time"]
        filtered_units = filtered_units[cons_columns]

        data = spike_detect(filtered_units, trials, 0.2, 0.2, nwbfile)


        # Create a new dataframe 

        data['pre_time'] = 0.2 #pre_time
        data['post_time'] = 0.2 #post_time
        data['mouse_id'] = mouse_name #mouse_id


        folder = f'Data/{mouse_name}'
        os.makedirs(folder, exist_ok=True)
        data.to_parquet(f'{folder}/{mouse_name}_Selectivity_Dataframe.parquet', index=False)

        mice_data.append(data)

    # Combine all event DataFrames into a single DataFrame
    df_combined = pd.concat(mice_data).reset_index(drop=True)
    os.makedirs('Data/Overall', exist_ok=True)
    df_combined.to_parquet(f'Data/Overall/Mice_Selectivity_Dataframe.parquet', index=False)
    return df_combined, mice_data





def AUC_generate(mouse_names = [], save_files = False, visualize = False, nb_neurons = 100, pre_vs_post_visualization = False):
    mice_data = []
    for i, mouse_name in enumerate(mouse_names):

        df = pd.read_parquet(f'Data/{mouse_name}/{mouse_name}_Selectivity_Dataframe.parquet')
        #mice_dfs[i] #
        

        # Check whether we want to save or visualize the files:
        if (save_files): 
            print("Starting to save files!")
            save_overall_auc(df, mouse_name)
        if (visualize): visualize_auc(df, nb_neurons)

        print(f"Starting process for Mouse {i+1}/{len(mouse_names)}")
        # AUC :
        df['whisker_AUC'] = df.apply(lambda row: compute_AUC(row, "whisker"), axis=1)
        df['auditory_AUC'] = df.apply(lambda row: compute_AUC(row, "auditory"), axis=1)
        df['wh/aud_AUC'] = df.apply(lambda row: compute_AUC(row, "wh/aud"), axis=1)
        df['lick_stim_AUC'] = df.apply(lambda row: compute_AUC(row, "lick_stim"), axis=1)


        # transformed AUC :
        df['Transformed whisker_AUC'] = df.apply(lambda row: 2*row['whisker_AUC']-1, axis=1)
        df['Transformed auditory_AUC'] = df.apply(lambda row: 2*row['auditory_AUC']-1, axis=1)
        df['Transformed wh/aud AUC'] = df.apply(lambda row: 2*row['wh/aud_AUC']-1, axis=1)
        df['Transformed lick_stim AUC'] = df.apply(lambda row: 2*row['lick_stim_AUC']-1, axis=1)

        if pre_vs_post_visualization: plot_pre_post(df)

        ## Bootstrapping process :
        print("Starting bootstrapping process...")
        df = bootstrapping(df)

        # Pivot the table :
        print("Creating pivot table!")
        df_clean = create_combined_df(df)

        df_clean['mouse_id'] = mouse_name

        ### save the parquet file :
        df_clean.to_parquet(f'Data/{mouse_name}/{mouse_name}_AUC_Selectivity.parquet', index=False)
        print(f"Process finished for Mouse {i+1}/{len(mouse_names)}!")
        mice_data.append(df_clean)
    # Combine all event DataFrames into a single DataFrame
    df_combined = pd.concat(mice_data).reset_index(drop=True)
    os.makedirs('Data/Overall', exist_ok=True)
    df_combined.to_parquet(f'Data/Overall/Mice_AUC_Selectivity.parquet', index=False)
    
