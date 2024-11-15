import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score 
import os
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
from auc_analysis import *



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
                    'whisker_pre_spikes', 'whisker_post_spikes', 'pre_time', 'pre_time', 
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
                    'auditory_pre_spikes', 'auditory_post_spikes', 'pre_time', 'pre_time', 
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
                    'pre_time', 'post_time', 'Wh/Aud AUC', 'Transformed Wh/Aud AUC', 
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
