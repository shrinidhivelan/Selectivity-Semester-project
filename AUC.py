import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score 
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
            return None, None, None, None
        
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

    return p_values_pos, p_values_neg, bootstrap_aucs, original_aucs


def calculate_ROC_individual(pre, post, cluster_ID):
    # Combine data and labels
    whisker_spike_counts = np.concatenate([pre, post])
    len_per_element_pre = len(pre)
    len_per_element_post = len(post)
    
    labels = np.concatenate([np.zeros(len_per_element_pre), np.ones(len_per_element_post)])

    # Check if labels have at least two classes
    if len(np.unique(labels)) < 2:
        return None, None, None, np.nan  # Return NaN for ROC AUC or another default

    
    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, whisker_spike_counts)

    roc_auc = roc_auc_score(labels, whisker_spike_counts)
    return fpr, tpr, thresholds, roc_auc


def compute_AUC(row, type, context):

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



def bootstrapping(old_df, nb_iterations = 1000, has_context = True):
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
            contexts = ["active"] if not has_context else ["passive", "active"]
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


def create_combined_df_v6(df, has_context=True):
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
            'contexts': ['active'] if not has_context else ['active', 'passive'],
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
            'contexts': ['active'] if not has_context else ['active', 'passive'],
            'event': 'auditory'
        },
        'wh_vs_aud': {
            'columns': {
                'pre_spikes': 'whisker_passive_post_spikes',
                'post_spikes': 'auditory_passive_post_spikes',
                'AUC': 'wh_vs_aud_active_AUC',
                'Transformed AUC': 'Transformed wh_vs_aud_active_AUC',
                'p-values positive': 'p-values positive wh_vs_audactive',
                'p-values negative': 'p-values negative wh_vs_audactive',
                'selective': 'selective wh_vs_audactive',
                'direction': 'direction wh_vs_audactive'
            },
            'contexts': ['active'] if not has_context else ['active', 'passive'],
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
        for event, config in events_config.items():
            for context in config['contexts']:
                # Create a new row with shared information
                new_row = {
                    'cluster_id': row['cluster_id'],
                    'ccf_acronym': row['ccf_acronym'],
                    'ccf_name': row['ccf_name'],
                    'ccf_parent_id': row['ccf_parent_id'],
                    'ccf_parent_acronym': row['ccf_parent_acronym'],
                    'ccf_parent_name': row['ccf_parent_name'],
                    'spike_times': row['spike_times'],
                    'mouse_id': row['mouse_id'],
                    'context': context if has_context else 'active',
                    'event': config['event']
                }

                # Assign event-specific columns
                for col, column_name in config['columns'].items():
                    new_row[col] = row[column_name]

                # Append the new row
                combined_rows.append(new_row)
    
    # Convert to DataFrame
    combined_df = pd.DataFrame(combined_rows)
    
    return combined_df


def create_combined_df_v7(df, has_context=True):
    # Define the configuration for each event and its context
    events_config = {
        'whisker': {
            'columns': {
                'pre_spikes': 'pre_spikes',
                'post_spikes': 'post_spikes',
                'AUC': 'AUC',
                'Transformed AUC': 'Transformed AUC',
                'p-values positive': 'p-values positive',
                'p-values negative': 'p-values negative',
                'selective': 'selective',
                'direction': 'direction'
            },
            'contexts': ['active', 'passive'] if has_context else ['active'],
            'event': 'whisker'
        },
        'auditory': {
            'columns': {
                'pre_spikes': 'pre_spikes',
                'post_spikes': 'post_spikes',
                'AUC': 'AUC',
                'Transformed AUC': 'Transformed AUC',
                'p-values positive': 'p-values positive',
                'p-values negative': 'p-values negative',
                'selective': 'selective',
                'direction': 'direction'
            },
            'contexts': ['active', 'passive'] if has_context else ['active'],
            'event': 'auditory'
        },
        'wh_vs_aud': {
            'columns': {
                'pre_spikes': 'pre_spikes',
                'post_spikes': 'post_spikes',
                'AUC': 'AUC',
                'Transformed AUC': 'Transformed AUC',
                'p-values positive': 'p-values positive',
                'p-values negative': 'p-values negative',
                'selective': 'selective',
                'direction': 'direction'
            },
            'contexts': ['active', 'passive'] if has_context else ['active'],
            'event': 'wh_vs_aud'
        },
        'spontaneous_licks': {
            'columns': {
                'pre_spikes': 'pre_spikes',
                'post_spikes': 'post_spikes',
                'AUC': 'AUC',
                'Transformed AUC': 'Transformed AUC',
                'p-values positive': 'p-values positive',
                'p-values negative': 'p-values negative',
                'selective': 'selective',
                'direction': 'direction'
            },
            'contexts': ['active'],  # Only one context for spontaneous
            'event': 'spontaneous_licks'
        }
    }

    # Create a list to hold all the processed rows
    combined_rows = []
    
    # Iterate through each row in the original dataframe
    for idx, row in df.iterrows():
        for event, config in events_config.items():
            for context in config['contexts']:
                # Build the suffix for active/passive column handling
                context_suffix = f"{context}" if context in ['active', 'passive'] else 'active'

                # Create a new row with shared information
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

                # Assign event-specific columns based on context
                for col, column_name in config['columns'].items():
                    full_column_name = f"{event}_{context_suffix}_{column_name}"  # Match context-specific column names
                    new_row[col] = row[full_column_name]

                # Append the new row
                combined_rows.append(new_row)
    
    # Convert to DataFrame
    combined_df = pd.DataFrame(combined_rows)
    
    return combined_df



import pandas as pd

# Assuming your dataframe is named df
def pivot_table(df, has_context = True):
    # Define the lists of columns to pivot
    events = ['whisker', 'auditory', 'wh_vs_aud', 'spontaneous_licks']
    contexts = ['active', 'passive'] if has_context else ['active']    
    # Create an empty list to store the reshaped rows
    reshaped_rows = []
    
    for event in events:
        for context in contexts:
            # Determine the column prefix
            pre_col = f'{event}_active_pre_spikes' if context == 'active' else f'{event}_passive_pre_spikes'
            post_col = f'{event}_active_post_spikes' if context == 'active' else f'{event}_passive_post_spikes'
            auc_col = f'{event}_active_AUC' if context == 'active' else f'{event}_passive_AUC'
            transformed_auc_col = f'{event}_active_Transformed AUC' if context == 'active' else f'{event}_passive_Transformed AUC'
            p_value_pos_col = f'p-values positive {event}active' if context == 'active' else f'p-values positive {event}passive'
            p_value_neg_col = f'p-values negative {event}active' if context == 'active' else f'p-values negative {event}passive'
            selective_col = f'selective {event}active' if context == 'active' else f'selective {event}passive'
            direction_col = f'direction {event}active' if context == 'active' else f'direction {event}passive'
            
            # Special mapping for wh_vs_aud
            if event == 'wh_vs_aud':
                pre_col = f'whisker_active_post_spikes' if context == 'active' else f'auditory_active_post_spikes'
                post_col = f'auditory_active_post_spikes' if context == 'active' else f'whisker_active_post_spikes'
            
            # Special mapping for spontaneous_licks (no change in pre/post logic)
            if event == 'spontaneous_licks':
                pre_col = f'spontaneous_licks_pre_spikes'
                post_col = f'spontaneous_licks_post_spikes'
                # Adjust p-values and selective columns for spontaneous_licks
                p_value_pos_col = f'p-values positive spontaneous_licks'
                p_value_neg_col = f'p-values negative spontaneous_licks'
                selective_col = f'selective spontaneous_licks'
                direction_col = f'direction spontaneous_licks'

            # Extract the necessary information from each row and reshape
            for _, row in df.iterrows():
                reshaped_row = {
                    'cluster_id': row['cluster_id'],
                    'ccf_acronym': row['ccf_acronym'],
                    'ccf_name': row['ccf_name'],
                    'ccf_parent_id': row['ccf_parent_id'],
                    'ccf_parent_acronym': row['ccf_parent_acronym'],
                    'ccf_parent_name': row['ccf_parent_name'],
                    'spike_times': row['spike_times'],
                    'mouse_id': row['mouse_id'],
                    'context': context,
                    'event': event,
                    'pre_spikes': row[pre_col],
                    'post_spikes': row[post_col],
                    'AUC': row[auc_col],
                    'Transformed AUC': row[transformed_auc_col],
                    'p-values positive': row[p_value_pos_col],
                    'p-values negative': row[p_value_neg_col],
                    'selective': row[selective_col],
                    'direction': row[direction_col],
                    'pre_time': row['pre_time'],
                    'post_time': row['post_time']
                }
                reshaped_rows.append(reshaped_row)
    
    # Convert the list of reshaped rows into a new DataFrame
    reshaped_df = pd.DataFrame(reshaped_rows)
    
    return reshaped_df