import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score 
import os
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
#from auc_analysis import *


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


def calculate_ROC(whisker_pre, whisker_post, index, cluster_ID, type="whisker"):
    # Combine data and labels
    whisker_spike_counts = np.concatenate([whisker_pre[index], whisker_post[index]])
    labels = np.concatenate([np.ones(len(whisker_pre[index])), np.zeros(len(whisker_post[index]))])
    clu_id = cluster_ID[index]


    if len(np.unique(labels)) < 2:
        #print(f"Cluster {clu_id} has only one class in labels. Skipping AUC computation.")
        return None, None, None, labels, np.nan, whisker_spike_counts  # Return NaN for ROC AUC or another default


    
    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, whisker_spike_counts)
    #roc_auc = auc(fpr, tpr)
    roc_auc = roc_auc_score(labels, whisker_spike_counts)

    
    return fpr, tpr, thresholds, labels, roc_auc, whisker_spike_counts



def perform_bootstrap_roc_analysis(whisker_pre, whisker_post, cluster_id, n_iterations=1000, type="whisker"):
    p_values_pos = []
    p_values_neg = []
    original_aucs = []  
    
    # Iterate over each cluster
    for i in tqdm(range(len(cluster_id)), desc="Processing Clusters"):
        
        # Calculate ROC for current cluster
        fpr, tpr, thresholds, labels, roc_auc, whisker_spike_counts = calculate_ROC(
            whisker_pre, whisker_post, i, cluster_id, type=type
        )
        
        if roc_auc is np.nan or len(np.unique(labels)) < 2:
            # Skip the current cluster if ROC AUC is NaN or there is not enough class variability
            p_values_pos.append(np.nan)
            p_values_neg.append(np.nan)
            original_aucs.append(np.nan)
            continue
        
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

    return p_values_pos, p_values_neg, original_aucs


def calculate_ROC_individual(pre, post, cluster_ID):
    # Combine data and labels
    whisker_spike_counts = np.concatenate([pre, post])
    len_per_element_pre = len(pre)
    len_per_element_post = len(post)
    
    labels = np.concatenate([np.zeros(len_per_element_pre), np.ones(len_per_element_post)])
    clu_id = cluster_ID

    # Check if labels have at least two classes
    if len(np.unique(labels)) < 2:
        #print(f"Cluster {clu_id} has only one class in labels. Skipping AUC computation.")
        return None, None, None, np.nan  # Return NaN for ROC AUC or another default

    
    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, whisker_spike_counts)
    #roc_auc = auc(fpr, tpr)
    #longueur du labels -> nombre d'events
    roc_auc = roc_auc_score(labels, whisker_spike_counts)
    return fpr, tpr, thresholds, roc_auc


def compute_AUC_2(row, type, context):

    if (type == 'whisker') or (type == 'auditory'):
        pre = f"{type}_{context}_pre_spikes"
        post = f"{type}_{context}_post_spikes"
    if (type == 'wh_vs_aud'):
        pre = f"whisker_{context}_post_spikes"
        post = f"auditory_{context}_post_spikes"
    else:
        pre = "spontaneous_licks_pre_spikes"
        post = "spontaneous_licks_post_spikes"

    _, _, _, roc_auc = calculate_ROC_individual(row[pre], row[post], row["cluster_id"])
    return roc_auc



def plot_roc_curve(ax, whisker_pre, whisker_post, index, cluster_ID, type="whisker"):

    # Calculate ROC
    fpr, tpr, thresholds, labels, roc_auc, _ = calculate_ROC(whisker_pre, whisker_post, index, cluster_ID, type)
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

def save_roc_plots(arr1, arr2, cluster_id, type="whisker", indices=[-9999], mouse_name = '', context = 'passive'):
    # Ensure indices list covers the desired range
    total_plots = len(arr1)
    if indices[0] == -9999:
        indices = range(total_plots)
    
    # Create folder if it doesn't exist
    if mouse_name == '':
        folder_path = f'plots/other/AUC_plots/{type}/{context}/'
    else:
        folder_path = f'plots/{mouse_name}/AUC_plots/{type}/{context}'
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
    fpr, tpr, thresholds, labels, original_auc, whisker_spike_counts = calculate_ROC(
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

def bootstrap_columns(old_df, p_values_pos, p_values_neg, bootstrap_aucs, type = "whisker", context = ''):
    if context != '':
        context = " " + context

    df = old_df.copy()

    df[f"p-values positive {type+context}"] = p_values_pos
    df[f"p-values negative {type+context}"] = p_values_neg

    # Create a new column in the table to show the selectiveness : True if it is of good importance, else false 
    df[f'selective {type+context}'] = (df[f'p-values positive {type+context}'] < 0.05) | (df[f"p-values negative {type+context}"] < 0.05)

    # Add "direction" column for positive or negative correlation
    df[f'direction {type+context}'] = np.where(df[f'p-values positive {type+context}'] < 0.05, 'positive', 
                            np.where(df[f'p-values negative {type+context}'] < 0.05, 'negative', 'none'))
    
    return df



def bootstrapping(old_df, nb_iterations = 1000):
    df = old_df.copy()

    types = ['whisker', 'auditory', 'wh_vs_aud', 'spontaneous_licks']

    for type in types:
        if type == "spontaneous_licks":            
            contexts=''
            print("Starting bootstrapping for spontaneous_licks...")
            p_values_pos, p_values_neg, bootstrap_aucs, original_aucs = perform_bootstrap_roc_analysis(df['spontaneous_licks_pre_spikes'], df['spontaneous_licks_post_spikes'], df["cluster_id"], nb_iterations, type)
            df[f"p-values positive {type}"] = p_values_pos
            df[f"p-values negative {type}"] = p_values_neg

            # Create a new column in the table to show the selectiveness : True if it is of good importance, else false 
            df[f'selective {type}'] = (df[f'p-values positive {type}'] < 0.05) | (df[f"p-values negative {type}"] < 0.05)

            # Add "direction" column for positive or negative correlation
            df[f'direction {type}'] = np.where(df[f'p-values positive {type}'] < 0.05, 'positive', 
                                    np.where(df[f'p-values negative {type}'] < 0.05, 'negative', 'none'))
            
            
            
        else:
            contexts = ['passive', 'active']
            for context in contexts:
                #context = " "+context

                if type == 'wh_vs_aud':
                    pre = f'whisker_{context}_post_spikes'
                    post = f'auditory_{context}_post_spikes'
                else:
                    pre = f'{type}_{context}_pre_spikes'
                    post = f'{type}_{context}_post_spikes'

                print(f"Starting bootstrapping for {context} {type} stimulation...")
                
                p_values_pos, p_values_neg, bootstrap_aucs, original_aucs = perform_bootstrap_roc_analysis(df[pre], df[post], df["cluster_id"], nb_iterations, type)
                
                df[f"p-values positive {type+context}"] = p_values_pos
                df[f"p-values negative {type+context}"] = p_values_neg
                # Create a new column in the table to show the selectiveness : True if it is of good importance, else false 
                df[f'selective {type+context}'] = (df[f'p-values positive {type+context}'] < 0.05) | (df[f"p-values negative {type+context}"] < 0.05)
                # Add "direction" column for positive or negative correlation
                df[f'direction {type+context}'] = np.where(df[f'p-values positive {type+context}'] < 0.05, 'positive', 
                                        np.where(df[f'p-values negative {type+context}'] < 0.05, 'negative', 'none'))


    """ 
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
    """

    return df


def create_combined_df_v6(df):
    # Define the configuration for each event and its context
    events_config = {
        'whisker': {
            'columns': {
                'pre_spikes': 'whisker_active_pre_spikes',
                'post_spikes': 'whisker_active_post_spikes',
                'AUC': 'whisker_active_AUC',
                'Transformed AUC': 'Transformed whisker_active_AUC',
                'p-values positive': 'p-values positive whiskeractive',
                'p-values negative': 'p-values negative whiskeractive',
                'selective': 'selective whiskeractive',
                'direction': 'direction whiskeractive'
            },
            'contexts': ['active', 'passive'],
            'event': 'whisker'
        },
        'auditory': {
            'columns': {
                'pre_spikes': 'auditory_active_pre_spikes',
                'post_spikes': 'auditory_active_post_spikes',
                'AUC': 'auditory_active_AUC',
                'Transformed AUC': 'Transformed auditory_active_AUC',
                'p-values positive': 'p-values positive auditoryactive',
                'p-values negative': 'p-values negative auditoryactive',
                'selective': 'selective auditoryactive',
                'direction': 'direction auditoryactive'
            },
            'contexts': ['active', 'passive'],
            'event': 'auditory'
        },
        'wh_vs_aud': {
            'columns': {
                'pre_spikes': 'whisker_passive_post_spikes',  # This comes from whisker post_spikes
                'post_spikes': 'auditory_passive_post_spikes',  # This comes from auditory post_spikes
                'AUC': 'wh_vs_aud_active_AUC',
                'Transformed AUC': 'Transformed wh_vs_aud_active_AUC',
                'p-values positive': 'p-values positive wh_vs_audactive',
                'p-values negative': 'p-values negative wh_vs_audactive',
                'selective': 'selective wh_vs_audactive',
                'direction': 'direction wh_vs_audactive'
            },
            'contexts': ['active', 'passive'],
            'event': 'wh_vs_aud'
        },
        'spontaneous_licks': {
            'columns': {
                'pre_spikes': 'spontaneous_licks_pre_spikes',
                'post_spikes': 'spontaneous_licks_post_spikes',
                'AUC': 'spontaneous_licks_AUC',
                'Transformed AUC': 'Transformed spontaneous_licks_AUC',
                'p-values positive': 'p-values positive spontaneous_licks',
                'p-values negative': 'p-values negative spontaneous_licks',
                'selective': 'selective spontaneous_licks',
                'direction': 'direction spontaneous_licks'
            },
            'contexts': ['active'],  # Only one context for spontaneous
            'event': 'spontaneous_licks'
        }
    }
    
    # Create a list to hold all the processed rows
    combined_rows = []
    
    # Iterate through each row in the original dataframe
    for idx, row in df.iterrows():
        # For each event, iterate over its contexts and create new rows
        for event, config in events_config.items():
            for context in config['contexts']:
                # Create a new row with the selected columns
                new_row = {
                    'cluster_id': row['cluster_id'],
                    'ccf_acronym': row['ccf_acronym'],
                    'ccf_name': row['ccf_name'],
                    'ccf_parent_id': row['ccf_parent_id'],
                    'ccf_parent_acronym': row['ccf_parent_acronym'],
                    'ccf_parent_name': row['ccf_parent_name'],
                    'spike_times': row['spike_times'],
                    'mouse_id': row['mouse_id'],
                    'context': context,
                    'event': config['event']
                }
                
                # Assign the appropriate columns based on the event/context
                for col, column_name in config['columns'].items():
                    new_row[col] = row[column_name]
                
                # Append the new row to the list
                combined_rows.append(new_row)
    
    # Convert the list of rows to a DataFrame
    combined_df = pd.DataFrame(combined_rows)
    
    return combined_df

