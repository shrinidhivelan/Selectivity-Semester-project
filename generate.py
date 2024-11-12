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
from auc_analysis import *

def generate_mice_data(mouse_names):

    mice_data = []

    for mouse in mouse_names:
    
        # Load the file with the mice data
        io = NWBHDF5IO(f'/Users/shrinidhivelan/Desktop/LSENS - Semester Project/{mouse}.nwb', mode='r')
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

        # proc_data = spike_detection(unit_table, event_times)
        #data = spike_detection(filtered_units, trials, type = 'whisker')
        #data = spike_detection(data, trials, type = 'auditory')
        #data = spike_detection(data, trials, type = 'lick_stim', file=nwbfile)
        data = spike_detect(filtered_units, trials, 0.2, 0.2, nwbfile)
        # Create a new dataframe 

        data['pre'] = 0.2 #pre_time
        data['post'] = 0.2 #post_time
        data['Mouse name'] = mouse_name #mouse_id


        folder = f'Data/{mouse_name}'
        os.makedirs(folder, exist_ok=True)
        data.to_parquet(f'{folder}/{mouse_name}_Selectivity_Dataframe.parquet', index=False)

        mice_data.append(data)

    # Combine all event DataFrames into a single DataFrame
    df_combined = pd.concat(mice_data).reset_index(drop=True)
    os.makedirs('Data/Overall', exist_ok=True)
    df_combined.to_parquet(f'Data/Overall/Mice_Selectivity_Dataframe.parquet', index=False)
    return df_combined


def save_overall_auc(df, mouse_name):

    whisker_pre = df["whisker_pre_spikes"]
    whisker_post = df["whisker_post_spikes"]
    auditory_pre = df["auditory_pre_spikes"]
    auditory_post = df["auditory_post_spikes"]
    lick_stim_pre = df["lick_stim_pre_spikes"]
    lick_stim_post = df["lick_stim_post_spikes"]

    cluster_id = df["cluster_id"]

    save_roc_plots(whisker_pre, whisker_post, cluster_id, "whisker", mouse_name = mouse_name)
    save_roc_plots(auditory_pre, auditory_post, cluster_id, "auditory", mouse_name = mouse_name)
    save_roc_plots(whisker_post, auditory_post, cluster_id, "wh_aud", mouse_name = mouse_name)
    save_roc_plots(lick_stim_pre, lick_stim_post, cluster_id, "lick_stim", mouse_name = mouse_name)

def visualize_auc(df, nb_neurons=100):

    whisker_pre = df["whisker_pre_spikes"]
    whisker_post = df["whisker_post_spikes"]
    cluster_id = df["cluster_id"]
    auditory_pre = df["auditory_pre_spikes"]
    auditory_post = df["auditory_post_spikes"]
    lick_stim_pre = df["lick_stim_pre_spikes"]
    lick_stim_post = df["lick_stim_post_spikes"]


    # Whisker visualization plot
    plot_subplots(whisker_pre, whisker_post, cluster_id, "whisker", 4, nb_neurons)

    # Auditory visualization plot
    plot_subplots(auditory_pre, auditory_post, cluster_id, "auditory", 4, nb_neurons)#, 4, len(cluster_id))

    # Whisker vs Auditory visualization plot 
    plot_subplots(whisker_post, auditory_post, cluster_id, "wh/aud", 4, nb_neurons)

    # Lick stim pre vs post visualization
    plot_subplots(lick_stim_pre, lick_stim_post, cluster_id, "wh/aud", 4, nb_neurons)


def plot_pre_post(df):
    interesting_whisker_AUC = df[abs(df["Transformed whisker_AUC"])>0.5]
    interesting_auditory_AUC = df[abs(df["Transformed auditory_AUC"])>0.5]
    interesting_wa_AUC = df[abs(df["Transformed wh/aud AUC"])>0.5]
    interesting_lick_stim_AUC = df[abs(df["Transformed lick_stim AUC"])>0.5]

    for i in range(len(interesting_whisker_AUC)):
        plot_single_roc(
            interesting_whisker_AUC["whisker_pre_spikes"].iloc[i], 
            interesting_whisker_AUC["whisker_post_spikes"].iloc[i], 
            interesting_whisker_AUC["cluster_id"].iloc[i], 
            index=i, type = 'whisker'
        )    

    for i in range(len(interesting_auditory_AUC)):
        plot_single_roc(
            interesting_auditory_AUC["auditory_pre_spikes"].iloc[i], 
            interesting_auditory_AUC["auditory_post_spikes"].iloc[i], 
            interesting_auditory_AUC["cluster_id"].iloc[i], 
            index=i, type = "auditory"
        )

    for i in range(len(interesting_wa_AUC)):
        plot_single_roc(
            interesting_wa_AUC["whisker_post_spikes"].iloc[i], 
            interesting_wa_AUC["auditory_post_spikes"].iloc[i], 
            interesting_wa_AUC["cluster_id"].iloc[i], 
            index=i, type = "wh/aud"
        )
    for i in range(len(interesting_lick_stim_AUC)):
        plot_single_roc(
            interesting_lick_stim_AUC["lick_stim_post_spikes"].iloc[i], 
            interesting_lick_stim_AUC["lick_stim_post_spikes"].iloc[i], 
            interesting_lick_stim_AUC["cluster_id"].iloc[i], 
            index=i, type = "lick_stim"
        )

def bootstrap_column(df, p_values_pos, p_values_neg, bootstrap_aucs, type = "whisker"):

    df[f"p-values positive {type}"] = p_values_pos
    df[f"p-values negative {type}"] = p_values_neg

    # Create a new column in the table to show the selectiveness : True if it is of good importance, else false 
    df[f'selective {type}'] = (df[f'p-values positive {type}'] < 0.05) | (df[f"p-values negative {type}"] < 0.05)

    # Add "direction" column for positive or negative correlation
    df[f'direction {type}'] = np.where(df[f'p-values positive {type}'] < 0.05, 'positive', 
                            np.where(df[f'p-values negative {type}'] < 0.05, 'negative', 'none'))
    
    return df


def bootstrapping(df):

    # bootstrapping for whisker
    print("Starting bootstrapping for Whisker stimulation...")
    p_values_pos_whisker, p_values_neg_whisker, bootstrap_aucs_whisker, original_aucs_whisker = perform_bootstrap_roc_analysis(df["whisker_pre_spikes"], df["whisker_post_spikes"], df["cluster_id"], 1000, "whisker")
    
    print("Starting bootstrapping for Auditory stimulation...")
    p_values_pos_auditory, p_values_neg_auditory, bootstrap_aucs_auditory, original_aucs_auditory = perform_bootstrap_roc_analysis(df["auditory_pre_spikes"], df["auditory_post_spikes"], df["cluster_id"], 1000, "auditory")

    print("Starting bootstrapping for Aud vs Whisker stimulation...")
    p_values_pos_wh_aud, p_values_neg_wh_aud, bootstrap_aucs_wh_aud, original_aucs_wh_aud = perform_bootstrap_roc_analysis(df["whisker_post_spikes"], df["auditory_post_spikes"], df["cluster_id"], 1000, "wh/aud")
    
    print("Starting bootstrapping for Lick stim pre vs post stimulation...")
    p_values_pos_lick_stim, p_values_neg_lick_stim, bootstrap_aucs_lick_stim, original_aucs_lick_stim = perform_bootstrap_roc_analysis(df["lick_stim_post_spikes"], df["lick_stim_post_spikes"], df["cluster_id"], 1000, "lick_stim")
    
    print("Creating all bootstrapping columns...")
    df = bootstrap_column(df, p_values_pos_whisker, p_values_neg_whisker, bootstrap_aucs_whisker, "whisker")
    df = bootstrap_column(df, p_values_pos_auditory, p_values_neg_auditory, bootstrap_aucs_auditory, "auditory")
    df = bootstrap_column(df, p_values_pos_wh_aud, p_values_neg_wh_aud, bootstrap_aucs_wh_aud, "wh/aud")
    df = bootstrap_column(df, p_values_pos_lick_stim, p_values_neg_lick_stim, bootstrap_aucs_lick_stim, "lick_stim")

    return df

def pivot_table(df):

    # Whisker columns
    df_whisker = df[['cluster_id', 'firing_rate', 'ccf_acronym', 'ccf_name', 
                    'ccf_parent_acronym', 'ccf_parent_name', 'spike_times', 
                    'whisker_pre_spikes', 'whisker_post_spikes', 'pre', 'post', 
                    'Whisker_AUC', 'Transformed Whisker_AUC', 
                    'p-values positive Whisker', 'p-values negative Whisker', 
                    'selective Whisker', 'direction Whisker']].copy()

    # Add a category column for Whisker
    df_whisker['category'] = 'Whisker'

    # Rename spikes and AUC columns for consistency
    df_whisker.rename(columns={'whisker_pre_spikes': 'pre_spikes', 
                            'whisker_post_spikes': 'post_spikes',
                            'Whisker_AUC': 'AUC',
                            'Transformed Whisker_AUC': 'Transformed AUC',
                            'p-values positive Whisker': 'p-values positive',
                            'p-values negative Whisker': 'p-values negative',
                            'selective Whisker': 'selective',
                            'direction Whisker': 'direction'}, inplace=True)

    # Repeat similar steps for Auditory
    df_auditory = df[['cluster_id', 'firing_rate', 'ccf_acronym', 'ccf_name', 
                    'ccf_parent_acronym', 'ccf_parent_name', 'spike_times', 
                    'auditory_pre_spikes', 'auditory_post_spikes', 'pre', 'post', 
                    'Auditory_AUC', 'Transformed Auditory_AUC', 
                    'p-values positive Auditory', 'p-values negative Auditory', 
                    'selective Auditory', 'direction Auditory']].copy()

    df_auditory['category'] = 'Auditory'

    df_auditory.rename(columns={'auditory_pre_spikes': 'pre_spikes', 
                                'auditory_post_spikes': 'post_spikes',
                                'Auditory_AUC': 'AUC',
                                'Transformed Auditory_AUC': 'Transformed AUC',
                                'p-values positive Auditory': 'p-values positive',
                                'p-values negative Auditory': 'p-values negative',
                                'selective Auditory': 'selective',
                                'direction Auditory': 'direction'}, inplace=True)
    # Similarly for Aud/Wh
    df_aud_wh = df[['cluster_id', 'firing_rate', 'ccf_acronym', 'ccf_name', 
                    'ccf_parent_acronym', 'ccf_parent_name', 'spike_times', 
                    'pre', 'post', 'Wh/Aud AUC', 'Transformed Wh/Aud AUC', 
                    'p-values positive Aud/Wh', 'p-values negative Aud/Wh', 
                    'selective Aud/Wh', 'direction Aud/Wh']].copy()

    df_aud_wh['category'] = 'Aud/Wh'

    df_aud_wh.rename(columns={'Wh/Aud AUC': 'AUC',
                            'Transformed Wh/Aud AUC': 'Transformed AUC',
                            'p-values positive Aud/Wh': 'p-values positive',
                            'p-values negative Aud/Wh': 'p-values negative',
                            'selective Aud/Wh': 'selective',
                            'direction Aud/Wh': 'direction'}, inplace=True)

    # Combine all the DataFrames together
    df_clean = pd.concat([df_whisker, df_auditory, df_aud_wh])

    # Optionally reset the index for cleaner view
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean





def AUC_generate(mouse_names = [], save_files = False, visualize = False, nb_neurons = 100, pre_vs_post_visualization = False):
    mice_data = []
    for i, mouse_name in enumerate(mouse_names):

        df = pd.read_parquet(f'Data/{mouse_name}/{mouse_name}_Selectivity_Dataframe.parquet')

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

        df_clean['Mouse name'] = mouse_name

        ### save the parquet file :
        df_clean.to_parquet(f'Data/{mouse_name}/{mouse_name}_AUC_Selectivity.parquet', index=False)
        print(f"Process finished for Mouse {i+1}/{len(mouse_names)}!")
        mice_data.append(df_clean)
    # Combine all event DataFrames into a single DataFrame
    df_combined = pd.concat(mice_data).reset_index(drop=True)
    os.makedirs('Data/Overall', exist_ok=True)
    df_combined.to_parquet(f'Data/Overall/Mice_AUC_Selectivity.parquet', index=False)
    
def AUC_plots(mouse_names = []):
    mice_data = []
    for i, mouse_name in enumerate(mouse_names):
        df = pd.read_parquet(f'Data/{mouse_name}/{mouse_name}_Selectivity_Dataframe.parquet')
        # Check whether we want to save or visualize the files:
        print("Starting to save files!")
        save_overall_auc(df, mouse_name)