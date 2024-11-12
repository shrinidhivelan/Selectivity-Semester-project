import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score 
import os
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_bootstrap_auc_distribution(df, bootstrap_aucs, original_auc, cluster_id, cluster_nb):
    """
    Plots the bootstrap AUC distribution for a specific cluster, with the original AUC shown as a dashed line.

    Parameters:
    - bootstrap_aucs: List or array of bootstrap AUC values for the specific cluster.
    - original_auc: The original AUC value for the specific cluster.
    - cluster_id: Identifier of the cluster for labeling the plot.
    """

    i = df["cluster_id"].tolist().index(str(cluster_nb))

    plt.figure(figsize=(8, 6))

    # Plot the histogram of bootstrap AUCs
    plt.hist(bootstrap_aucs, bins=30, color='skyblue', edgecolor='black', alpha=0.6, label='Bootstrap AUCs')

    # Plot the original AUC as a vertical line
    plt.axvline(original_auc[i], color='red', linestyle='--', linewidth=2, label=f'Original AUC = {original_auc[i]:.2f}')

    # Adding labels and title
    plt.xlabel('AUC')
    plt.ylabel('Frequency')
    plt.title(f'Bootstrap AUC Distribution for Cluster ID: {cluster_id[i]}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def calculate_ROC2(whisker_pre, whisker_post, index, cluster_ID, type="whisker"):
    # Combine data and labels
    whisker_spike_counts = np.concatenate([whisker_pre[index], whisker_post[index]])
    labels = np.concatenate([np.ones(len(whisker_pre[index])), np.zeros(len(whisker_post[index]))])
    clu_id = cluster_ID[index]
    
    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, whisker_spike_counts)
    #roc_auc = auc(fpr, tpr)
    roc_auc = roc_auc_score(labels, whisker_spike_counts)
    return fpr, tpr, thresholds, labels, roc_auc, whisker_spike_counts


# nommer plus generalement
def perform_bootstrap_roc_analysis(whisker_pre, whisker_post, cluster_id, n_iterations=1000, type = "whisker"):
    p_values_pos = []
    p_values_neg = []
    original_aucs = []  # List to store original AUCs for each cluster

    
    # Iterate over each cluster
    #for i in range(len(cluster_id)):
    for i in tqdm(range(len(cluster_id)), desc="Processing Clusters"):

        # Calculate ROC for current cluster
        fpr, tpr, thresholds, labels, roc_auc, whisker_spike_counts = calculate_ROC2(
            whisker_pre, whisker_post, i, cluster_id, type=type
        )

        # Store the original AUC for this cluster
        original_aucs.append(roc_auc)
        
        # Bootstrap iterations
        bootstrap_aucs = []
        for _ in range(n_iterations):
            # Shuffle labels and compute ROC
            shuffled_labels = np.random.permutation(labels)
            fpr, tpr, _ = roc_curve(shuffled_labels, whisker_spike_counts)
            bootstrap_auc = auc(fpr, tpr)
            bootstrap_aucs.append(bootstrap_auc)

        # Convert to numpy array for easier calculations
        bootstrap_aucs = np.array(bootstrap_aucs)

        # Calculate p-values
        p_value_pos = np.sum(bootstrap_aucs >= roc_auc) / n_iterations
        p_value_neg = np.sum(bootstrap_aucs <= roc_auc) / n_iterations

        # Append the p-values
        p_values_pos.append(p_value_pos)
        p_values_neg.append(p_value_neg)

        # Debugging print statements (can be removed or replaced with logging)
        # print(f"{i}/{len(cluster_id)} Original AUC: {roc_auc} for Cluster: {cluster_id.values[i]}")
        # print(f"Positive p-value: {p_value_pos}")
        # print(f"Negative p-value: {p_value_neg}")
    
    return p_values_pos, p_values_neg, bootstrap_aucs, original_aucs



def calculate_ROC_individual(whisker_pre, whisker_post, cluster_ID, type = "whisker"):
    # Combine data and labels
    whisker_spike_counts = np.concatenate([whisker_pre, whisker_post])
    len_per_element_pre = len(whisker_pre)
    len_per_element_post = len(whisker_post)
    
    labels = np.concatenate([np.zeros(len_per_element_pre), np.ones(len_per_element_post)])
    clu_id = cluster_ID
    
    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, whisker_spike_counts)
    #roc_auc = auc(fpr, tpr)
    #longueur du labels -> nombre d'events
    roc_auc = roc_auc_score(labels, whisker_spike_counts)
    return fpr, tpr, thresholds, roc_auc

def compute_AUC(row, type="whisker"):
    if (type == "whisker"):
        _, _, _, roc_auc = calculate_ROC_individual(row["whisker_pre_spikes"], row["whisker_post_spikes"], row["cluster_id"], type)
    elif (type == "auditory"):
        _, _, _, roc_auc = calculate_ROC_individual(row["auditory_pre_spikes"], row["auditory_post_spikes"], row["cluster_id"], type)
    elif (type == "lick_stim"):
        _, _, _, roc_auc = calculate_ROC_individual(row["lick_stim_pre_spikes"], row["lick_stim_post_spikes"], row["cluster_id"], type)

    else:
        _, _, _, roc_auc = calculate_ROC_individual(row["whisker_post_spikes"], row["auditory_post_spikes"], row["cluster_id"], type)
    return roc_auc

def calculate_ROC(whisker_pre, whisker_post, index, cluster_ID, type = "whisker"):
    # Combine data and labels of the two elements we want to test agaib
    whisker_spike_counts = np.concatenate([whisker_pre[index], whisker_post[index]])
    len_per_element_pre = len(whisker_pre[index])
    len_per_element_post = len(whisker_post[index])
    # note invert
    labels = np.concatenate([np.zeros(len_per_element_pre), np.ones(len_per_element_post)])
    clu_id = cluster_ID[index]
    
    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, whisker_spike_counts)
    #roc_auc = auc(fpr, tpr)
    roc_auc = roc_auc_score(labels, whisker_spike_counts)

    return fpr, tpr, thresholds, labels, roc_auc


def plot_roc_curve(ax, whisker_pre, whisker_post, index, cluster_ID, type="whisker"):

    # Calculate ROC
    fpr, tpr, thresholds, labels, roc_auc = calculate_ROC(whisker_pre, whisker_post, index, cluster_ID, type)
    transformed_auc = 2 * roc_auc - 1
    
    # Plot the ROC Curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {transformed_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random guessing line
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Neuron {index}, Cluster ID: {cluster_ID[index]}')
    ax.legend(loc="lower right")


def plot_subplots(whisker_pre, whisker_post, cluster_id, type="whisker", plots_per_row=4, total_plots=667, indices = [-9999]):
    # Define how many plots per figure
    plots_per_figure = 100  # Adjust this to control how many plots per figure
    num_figures = (total_plots + plots_per_figure - 1) // plots_per_figure  # Calculate number of figures needed
    
    for fig_index in range(num_figures):
        # Calculate range for this figure
        start_index = fig_index * plots_per_figure
        end_index = min(start_index + plots_per_figure, total_plots)
            

        # Calculate number of rows needed
        num_plots = end_index - start_index
        rows = (num_plots + plots_per_row - 1) // plots_per_row  # Ensures enough rows

        # Create subplots
        fig, axes = plt.subplots(rows, plots_per_row, figsize=(plots_per_row * 4, rows * 4))

        # Flatten axes array for easier indexing
        axes = axes.flatten()

        # Plot each ROC curve in its respective subplot

        if indices[0]==-9999:
            for i in range(start_index, end_index):
                plot_roc_curve(axes[i - start_index], whisker_pre, whisker_post, i, cluster_id, type)
        else:
            for i in range(indices):
                plot_roc_curve(axes[i], whisker_pre, whisker_post, i, cluster_id, type)

        # Hide any extra subplots (if total_plots is not a perfect multiple of plots_per_row)
        for i in range(num_plots, len(axes)):
            axes[i].axis('off')  # Turn off unused subplots

        # Adjust layout
        plt.tight_layout()
        plt.show()

def save_roc_plots(arr1, arr2, cluster_id, type="whisker", indices=[-9999], mouse_name = ''):
    # Ensure indices list covers the desired range
    total_plots = len(arr1)
    if indices[0] == -9999:
        indices = range(total_plots)
    
    # Create folder if it doesn't exist
    if mouse_name == '':
        folder_path = f'plots/other/AUC_plots/{type}'
    else:
        folder_path = f'plots/{mouse_name}/AUC_plots/{type}'
    os.makedirs(folder_path, exist_ok=True)
    
    # Generate and save each ROC plot
    for i in tqdm(indices):
        fig, ax = plt.subplots(figsize=(6, 6))  # Create a new figure for each plot
        plot_roc_curve(ax, arr1, arr2, i, cluster_id, type)
        
        # Define file path and save the figure
        file_path = f"{folder_path}/roc_neuron_{i}_cluster_{cluster_id[i]}.png"
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory

def plot_single_roc(whisker_pre, whisker_post, cluster_id, index, type="whisker"):
    # Convert strings to lists if needed
    if isinstance(whisker_pre, str):
        whisker_pre = ast.literal_eval(whisker_pre)
    if isinstance(whisker_post, str):
        whisker_post = ast.literal_eval(whisker_post)

    # Ensure whisker_pre and whisker_post are arrays
    whisker_pre = np.array(whisker_pre)
    whisker_post = np.array(whisker_post)

    # Combine data for the specific index
    whisker_spike_counts = np.concatenate([whisker_pre, whisker_post])
    len_per_element = len(whisker_pre)
    len_per_element2 = len(whisker_post)
    labels = np.concatenate([np.ones(len_per_element), np.zeros(len_per_element2)])
    
    # Compute the ROC curve
    fpr, tpr, _ = roc_curve(labels, whisker_spike_counts)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f}), transformed : {2*roc_auc-1:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
    plt.title(f'{type} ROC Curve for Cluster ID: {cluster_id}, Index: {index}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()


def plot_analysis(whisker_pre, whisker_post, i, cluster_id, bootstrap_aucs_whisker, original_auc, p_values_pos_whisker):
    """
    Generate and display plots for ROC analysis, histogram of bootstrap AUCs, 
    and p-values across all indices.

    Parameters:
    - whisker_pre: Pre-stimulation data for whisker.
    - whisker_post: Post-stimulation data for whisker.
    - i: Index for cluster ID to plot.
    - cluster_id: Array of cluster IDs.
    - bootstrap_aucs_whisker: Array of bootstrap AUC values for whisker.
    - original_auc: Original AUC value.
    - p_values_pos_whisker: Array of p-values for whisker.
    """

    # Assuming calculate_ROC2 is defined elsewhere and returns the required values
    fpr, tpr, thresholds, labels, original_auc, whisker_spike_counts = calculate_ROC2(
        whisker_pre, whisker_post, i, cluster_id, type="Whisker"
    )

    # Plot 2: Histogram of Bootstrap AUCs
    plt.figure(figsize=(8, 6))
    plt.hist(bootstrap_aucs_whisker, bins=30, color='skyblue', alpha=0.7, label='Bootstrap AUCs')
    plt.axvline(original_auc, color='red', linestyle='dashed', linewidth=2, label=f'Original AUC = {original_auc:.2f}')
    plt.title(f'Bootstrap AUC Distribution for Cluster ID: {cluster_id[i]}')
    plt.xlabel('AUC')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

    # Plot 3: Original AUC vs. Bootstrap AUC (overlay on histogram)
    plt.figure(figsize=(8, 6))
    plt.hist(bootstrap_aucs_whisker, bins=30, alpha=0.7, label='Bootstrap AUCs', color='lightgreen')
    plt.axvline(original_auc, color='darkred', linestyle='--', linewidth=2, label='Original AUC')
    plt.title(f'Original AUC Overlay on Bootstrap Distribution for Cluster ID: {cluster_id[i]}')
    plt.xlabel('AUC')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

    # Plot 4: p-values across all indices
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(p_values_pos_whisker)), p_values_pos_whisker, color='purple', label='P-values', alpha=0.6)
    plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold (p=0.05)')

    plt.title('P-values Across All Cluster Indices')
    plt.xlabel('Cluster Index')
    plt.ylabel('p-value')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def create_columns_p_values(df, type, p_values_pos, p_values_neg):
    df[f"p-values positive {type}"] = p_values_pos
    df[f"p-values negative {type}"] = p_values_neg

    # Create a new column in the table to show the selectiveness : True if it is of good importance, else false 
    # (meaning that there are no correlations between the quantities in question)
    # this column contains more false than the selective 2, meaning that it is more selective.
    df[f'selective {type}'] = (df[f'p-values positive {type}'] < 0.05) | (df[f"p-values negative {type}"] < 0.05)

    #df['selective_2'] = (df['p-values positive Whisker'] < 0.05) | (df["p-values positive Whisker"] > 0.95)

    # Add "direction" column for positive or negative correlation
    df[f'direction {type}'] = np.where(df[f'p-values positive {type}'] < 0.05, 'positive', 
                            np.where(df[f'p-values negative {type}'] < 0.05, 'negative', 'none'))
    # ou >0.95
    ## Ajouter pour negatifs 
    # nouvelle colonne
    #  "direction"
    return df



def create_combined_df(df):
    # Define the configuration for each event
    events_config = {
        'whisker': {
            'columns': ['cluster_id', 'firing_rate', 'ccf_acronym', 'ccf_name', 
                        'ccf_parent_acronym', 'ccf_parent_name', 'spike_times', 
                        'whisker_pre_spikes', 'whisker_post_spikes', 'pre', 'post', 
                        'whisker_AUC', 'Transformed whisker_AUC', 
                        'p-values positive whisker', 'p-values negative whisker', 
                        'selective whisker', 'direction whisker'],
            'new_columns': {
                'whisker_pre_spikes': 'pre_spikes',
                'whisker_post_spikes': 'post_spikes',
                'whisker_AUC': 'AUC',
                'Transformed whisker_AUC': 'Transformed AUC',
                'p-values positive whisker': 'p-values positive',
                'p-values negative whisker': 'p-values negative',
                'selective whisker': 'selective',
                'direction whisker': 'direction'
            },
            'event_name': 'whisker'
        },
        'auditory': {
            'columns': ['cluster_id', 'firing_rate', 'ccf_acronym', 'ccf_name', 
                        'ccf_parent_acronym', 'ccf_parent_name', 'spike_times', 
                        'auditory_pre_spikes', 'auditory_post_spikes', 'pre', 'post', 
                        'auditory_AUC', 'Transformed auditory_AUC', 
                        'p-values positive auditory', 'p-values negative auditory', 
                        'selective auditory', 'direction auditory'],
            'new_columns': {
                'auditory_pre_spikes': 'pre_spikes',
                'auditory_post_spikes': 'post_spikes',
                'auditory_AUC': 'AUC',
                'Transformed auditory_AUC': 'Transformed AUC',
                'p-values positive auditory': 'p-values positive',
                'p-values negative auditory': 'p-values negative',
                'selective auditory': 'selective',
                'direction auditory': 'direction'
            },
            'event_name': 'auditory'
        },
        'wh/aud': {
            'columns': ['cluster_id', 'firing_rate', 'ccf_acronym', 'ccf_name', 
                        'ccf_parent_acronym', 'ccf_parent_name', 'spike_times', 
                        'whisker_post_spikes', 'auditory_post_spikes', 'pre', 'post', 
                        'wh/aud_AUC', 'Transformed wh/aud AUC', 
                        'p-values positive wh/aud', 'p-values negative wh/aud', 
                        'selective wh/aud', 'direction wh/aud'],
            'new_columns': {
                'whisker_post_spikes': 'pre_spikes',
                'auditory_post_spikes': 'post_spikes',
                'wh/aud_AUC': 'AUC',
                'Transformed wh/aud AUC': 'Transformed AUC',
                'p-values positive wh/aud': 'p-values positive',
                'p-values negative wh/aud': 'p-values negative',
                'selective wh/aud': 'selective',
                'direction wh/aud': 'direction'
            },
            'event_name': 'wh/aud'
        },
        'lick_stim': {
            'columns': ['cluster_id', 'firing_rate', 'ccf_acronym', 'ccf_name', 
                        'ccf_parent_acronym', 'ccf_parent_name', 'spike_times', 
                        'lick_stim_pre_spikes', 'lick_stim_post_spikes', 'pre', 'post', 
                        'lick_stim_AUC', 'Transformed lick_stim AUC', 
                        'p-values positive lick_stim', 'p-values negative lick_stim', 
                        'selective lick_stim', 'direction lick_stim'],
            'new_columns': {
                'lick_stim_pre_spikes': 'pre_spikes',
                'lick_stim_post_spikes': 'post_spikes',
                'lick_stim_AUC': 'AUC',
                'Transformed lick_stim AUC': 'Transformed AUC',
                'p-values positive lick_stim': 'p-values positive',
                'p-values negative lick_stim': 'p-values negative',
                'selective lick_stim': 'selective',
                'direction lick_stim': 'direction'
            },
            'event_name': 'lick_stim'
        }
    }

    # List to hold the DataFrames
    dfs = []

    for event, config in events_config.items():
        event_df = df[config['columns']].copy()
        event_df['event'] = config['event_name']
        event_df.rename(columns=config['new_columns'], inplace=True)
        dfs.append(event_df)

    # Combine all event DataFrames into a single DataFrame
    df_combined = pd.concat(dfs).reset_index(drop=True)
    
    return df_combined