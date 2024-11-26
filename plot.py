import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score 
import os
import ast
from generate import *
from AUC import *
from helpers import *

################################ **** Functions used for plotting **** ################################

########## The following functions are there to visualize the neurons in terms of categories ###########

def barplots_specific(df, column_name = "ccf_acronym", category_name = "Aud/Wh", show = False):

    safe_category = category_name.replace("/", "_")

    # Create the directory structure if it doesn't exist
    save_dir = os.path.join("Plots", "Barplots", safe_category, column_name)
    os.makedirs(save_dir, exist_ok=True)

    # Get unique brain regions from the 'CCF region' column
    unique_regions = df[column_name].unique()

    # Loop over each unique brain region and create a separate plot
    for region in unique_regions:
        # Filter the DataFrame for the current region
        region_data = df[(df[column_name] == region) & (df['category'] == category_name)]
        safe_region = region.replace("/", "_")

        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.hist(region_data['Transformed AUC'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        
        # Add vertical lines for selectivity thresholds
        plt.axvline(x=0.6, color='red', linestyle='--', label='Selectivity Threshold')
        plt.axvline(x=-0.6, color='red', linestyle='--')
        
        # Add labels and title
        plt.xlabel("Transformed AUC")
        plt.ylabel("Number of Neurons")
        plt.title(f"Distribution of Selectivity (Transformed AUC) - {region}")
        
        # Add legend
        plt.legend()

        # Save the plot in the appropriate folder
        save_path = os.path.join(save_dir, f"barplot_{safe_region}.png")
        plt.savefig(save_path, bbox_inches='tight')
        
        if show:
            plt.show()

        # Close the plot to free up memory
        plt.close()


def plot_neuron_counts_with_percentages(df, offset=2, category='Whisker'):
    """
    Plots a stacked bar chart of selective and non-selective neurons by brain region with percentage annotations.

    Parameters:
    - df: DataFrame containing columns 'ccf_parent_acronym' (brain region) and 'selective' (boolean for neuron selectivity).
    - offset: Minimum vertical distance for percentage annotations (default=2).
    - category: Either 'Whisker', 'Auditory', 'Aud/Wh', or 'all'.
    """

    if category != 'all':
        df = df[df['category'] == category]

    # Count selective and non-selective neurons
    count_df = df.groupby(['ccf_parent_acronym', 'selective']).size().reset_index(name='Count')

    # Expand DataFrame to calculate percentages
    expanded_df = count_df.copy()
    total_counts = expanded_df.groupby('ccf_parent_acronym')['Count'].sum().reset_index()
    expanded_df = expanded_df.merge(total_counts, on='ccf_parent_acronym', suffixes=('', '_Total'))
    expanded_df['Percentage'] = (expanded_df['Count'] / expanded_df['Count_Total']) * 100

    # Create the pivot DataFrame for plotting
    pivot_df = expanded_df.pivot(index='ccf_parent_acronym', columns='selective', values='Count').fillna(0)

    # Plot the stacked bar chart
    ax = pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=['salmon', 'lightblue'])

    # Annotate the bar segments with percentages for only selective (True) neurons
    for n, region in enumerate(pivot_df.index):
        cumulative_height = 0  # Start from 0 for each region
        if True in pivot_df.columns:  # Ensure 'True' column exists
            value = pivot_df.loc[region, True]  # Get count of selective neurons
            if value > 0:
                # Fetch percentage for selective neurons in the specific region
                percentage_data = expanded_df[
                    (expanded_df['ccf_parent_acronym'] == region) &
                    (expanded_df['selective'] == True)
                ]

                # Only annotate if there is a valid percentage available
                if not percentage_data.empty:
                    percentage = percentage_data['Percentage'].iloc[0]

                    # Place text annotation at midpoint of the selective segment
                    text_y_position = cumulative_height + value / 2
                    ax.text(
                        n, text_y_position + offset,  # Position text above the bar
                        f"{percentage:.2f}%", ha='center', color='black'
                    )
                cumulative_height += value  # Update cumulative height with selective neuron count

    # Customize the plot
    plt.ylabel('Neuron Count')
    plt.title(f'Counts and Percentages of Selective Neurons by Brain Region for {category} Simulation')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()



def plot_neuron_percentages(df, offset=2, category='Whisker'):
    """
    Plots a grouped bar chart of the percentage of selective and non-selective neurons by brain region.

    Parameters:
    - df: DataFrame containing columns 'ccf_parent_acronym' (brain region) and 'selective' (boolean for neuron selectivity).
    - offset: Minimum vertical distance for percentage annotations (default=2).
    - category: Either 'Whisker', 'Auditory', 'Aud/Wh', or 'all'.
    """

    if category != 'all':
        df = df[df['category'] == category]

    # Count selective and non-selective neurons and calculate percentages
    count_df = df.groupby(['ccf_parent_acronym', 'selective']).size().reset_index(name='Count')
    total_counts = count_df.groupby('ccf_parent_acronym')['Count'].sum().reset_index(name='Total')
    percentage_df = count_df.merge(total_counts, on='ccf_parent_acronym')
    percentage_df['Percentage'] = (percentage_df['Count'] / percentage_df['Total']) * 100

    # Pivot DataFrame for plotting
    pivot_df = percentage_df.pivot(index='ccf_parent_acronym', columns='selective', values='Percentage').fillna(0)

    # Plot the grouped bar chart with percentages
    ax = pivot_df.plot(kind='bar', stacked=False, figsize=(10, 6), color=['salmon', 'lightblue'])

    # Annotate each bar with its percentage value
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', label_type='center', color='black', padding=offset)

    # Customize the plot
    plt.ylabel('Percentage of Neurons')
    plt.title(f'Percentage of Selective and Non-Selective Neurons by Brain Region for {category} Simulation')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.legend(['Non-Selective', 'Selective'])
    plt.show()



def plot_selective_neuron_percentage(df, offset=2, category='Whisker'):
    """
    Plots a bar chart of the percentage of selective neurons in each brain region relative to the total neurons,
    with percentage labels positioned above each bar.

    Parameters:
    - df: DataFrame containing columns 'ccf_parent_acronym' (brain region) and 'selective' (boolean for neuron selectivity).
    - offset: Minimum vertical distance for percentage annotations (default=2).
    - category: Either 'Whisker', 'Auditory', 'Aud/Wh', or 'all'.
    """

    if category != 'all':
        df = df[df['category'] == category]

    # Calculate the count of selective neurons per brain region
    selective_counts = df[df['selective'] == True].groupby('ccf_parent_acronym').size().reset_index(name='Selective_Count')

    # Calculate the total count of neurons per brain region
    total_counts = df.groupby('ccf_parent_acronym').size().reset_index(name='Total_Count')

    # Merge selective and total counts, then calculate the percentage of selective neurons
    percentage_df = selective_counts.merge(total_counts, on='ccf_parent_acronym')
    percentage_df['Percentage'] = (percentage_df['Selective_Count'] / percentage_df['Total_Count']) * 100

    # Generate unique colors for each brain region
    unique_regions = percentage_df['ccf_parent_acronym'].nunique()
    colors = plt.cm.viridis(np.linspace(0, 1, unique_regions))  # Using the 'viridis' colormap

    # Plot the bar chart with unique colors
    ax = percentage_df.plot(
        x='ccf_parent_acronym', y='Percentage', kind='bar', figsize=(10, 6),
        color=colors, legend=False
    )

    # Annotate each bar with its percentage value positioned outside the bar
    for i, (index, row) in enumerate(percentage_df.iterrows()):
        ax.text(
            i, row['Percentage'] + offset,  # Position text above the bar
            f"{row['Percentage']:.2f}%", ha='center', va='bottom', color='black'
        )

    # Customize the plot
    plt.ylabel('Percentage of Selective Neurons')
    plt.title(f'Percentage of Selective Neurons by Brain Region for {category} Simulation')
    plt.xticks(rotation=45, ha='right')  # Rotate x-ticks for better readability
    plt.tight_layout()
    plt.show()


########## Visualize lick times via raster plots : ###########
def visualize_lick_times(array, filtered, start = 0, end = 0):


    # Event plot setup
    fig, ax = plt.subplots(figsize=(15, 4))

    # Plot all lick times on the first line
    ax.eventplot([array], lineoffsets=1, colors='blue', linelengths=0.5, label='All Lick Times')

    # Plot filtered lick times on the second line
    ax.eventplot([filtered], lineoffsets=0, colors='orange', linelengths=0.5, label='Filtered Lick Times')

    # Customize the plot
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Filtered Lick Times', 'All Lick Times'])
    ax.set_xlabel('Time')
    if start!=end: ax.set_xlim(start, end)
    ax.legend()
    plt.title("Lick Times Event Plot")
    plt.show()


################### AUC ANALYSIS ###################
########## The following functions are used to plot the different AUC plots : ###########


def plot_roc_curve(ax, pre, post, index, cluster_ID, type="whisker", context = "passive"):


    # Calculate ROC
    fpr, tpr, thresholds, labels, roc_auc, _ = calculate_ROC(pre, post, index, cluster_ID, type)
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


def plot_subplots(pre, post, cluster_id, type="whisker", plots_per_row=4, total_plots=667, indices = [-9999]):
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
                plot_roc_curve(axes[i - start_index], pre, post, i, cluster_id, type)
        else:
            for i in range(indices):
                plot_roc_curve(axes[i], pre, post, i, cluster_id, type)

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
    if type != 'spontaneous_licks':
        if mouse_name == '':
            folder_path = f'plots/other/AUC_plots/{type}/{context}/'
        else:
            folder_path = f'plots/{mouse_name}/AUC_plots/{type}/{context}'
    else:
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

def plot_single_roc(pre, post, cluster_id, index, type="whisker", context = 'passive'):
    # Convert strings to lists if needed
    if isinstance(pre, str):
        pre = ast.literal_eval(pre)
    if isinstance(post, str):
        post = ast.literal_eval(post)

    # Ensure whisker_pre and whisker_post are arrays
    pre = np.array(pre)
    post = np.array(post)

    # Combine data for the specific index
    whisker_spike_counts = np.concatenate([pre, post])
    len_per_element = len(pre)
    len_per_element2 = len(post)
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





########### The following function is used to visualize pre-post behaviour ###########

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


############ The following functions are used to make the raster plots: ############

def Final_spikes(units, trials, start=0.5, stop=1):

    whisker_trials = trials[trials["trial_type"]=="whisker_trial"]
    auditory_trials = trials[trials["trial_type"]=="auditory_trial"]
    nostim_trials = trials[trials["trial_type"]=="no_stim_trial"]
    whisker_array, clusters = create_array_spikes(units, whisker_trials, type='whisker', start=start, stop=stop)
    auditory_array, _ = create_array_spikes(units, auditory_trials, type='auditory', start=start, stop=stop)
    nostim_array, _ = create_array_spikes(units, nostim_trials, type='no_stim', start=start, stop=stop)
    
    return whisker_array, auditory_array, nostim_array, clusters


def plot_raster_final(whisker_test, auditory_test, nostim_test, cluster, mouse_name = ''):

    filename = f"Raster_Neuron_{cluster}.png"

    folder = f'plots/{mouse_name}/raster plots'
    os.makedirs(folder, exist_ok=True)

    # Create a figure and axis
    plt.figure(figsize=(10, 6))

    # Plot no stimulation test spikes (y indices start from 0)
    for idx, spikes in enumerate(nostim_test):
        plt.scatter(spikes, np.full_like(spikes, idx), color='k', marker='|', s=50, label='No Stim' if idx == 0 else "")

    # Plot auditory test spikes (y indices start from 0)
    for idx, spikes in enumerate(auditory_test):
        plt.scatter(spikes, np.full_like(spikes, idx + len(nostim_test)), color='mediumblue', marker='|', s=50, label='Auditory' if idx == 0 else "")

    # Plot whisker test spikes (y indices start from 0)
    for idx, spikes in enumerate(whisker_test):
        plt.scatter(spikes, np.full_like(spikes, idx + len(nostim_test) + len(auditory_test)), color='forestgreen', marker='|', s=50, label='Whisker' if idx == 0 else "")

    # Set y-ticks to show a limited number of labels
    plt.yticks([0, len(nostim_test), len(nostim_test) + len(auditory_test)], ['No Stim', 'Auditory', 'Whisker'])

    # Add vertical shaded lines at -0.2 and 0.2
    plt.axvline(x=-0.2, color='grey', linestyle='--', linewidth=5, label='-0.2')  # Thicker line
    plt.axvline(x=0.2, color='grey', linestyle='--', linewidth=5, label='0.2')    # Thicker line

    plt.axvline(x=0.0, color='red', linestyle='-', linewidth=5, label='0.0')  # Thicker line


    # Add labels, title and save the plots
    plt.xlabel('Time (s)')
    plt.ylabel('Trial Index')
    plt.title(f'Raster Plot of Spike Times of cluster {cluster}')
    plt.savefig(os.path.join(folder, filename), bbox_inches='tight')
    plt.close()  

def Raster_total(units, trials, start = 0.5, stop = 1, mouse_name = ''):
    whisker_array, auditory_array, nostim_array, whisker_clusters = Final_spikes(units, trials, start, stop)
    Nb_neurons = len(whisker_array)
    for i in range(Nb_neurons):
        plot_raster_final(whisker_array[i], auditory_array[i], nostim_array[i], whisker_clusters[i], mouse_name)

""" 
def Raster_passive_active(units, trials, start = 0.5, stop = 1, mouse_name = '')

"""

########## Generate plots per category

def Final_spikes_context(units, trials, start=0.5, stop=1):
    """
    Separates spike arrays for passive and active contexts.

    Parameters:
    - units: DataFrame containing unit information, including spike times.
    - trials: DataFrame containing trial information, including trial type and context.
    - start: Time (in seconds) before trial start to include spikes.
    - stop: Time (in seconds) after trial start to include spikes.

    Returns:
    - Dictionary of spike arrays for passive and active contexts.
    - clusters: List of cluster IDs corresponding to each unit.
    """
    contexts = ['passive', 'active']
    spike_data = {}
    clusters = None

    for context in contexts:
        context_trials = trials[trials["context"] == context]
        whisker_trials = context_trials[context_trials["trial_type"] == "whisker_trial"]
        auditory_trials = context_trials[context_trials["trial_type"] == "auditory_trial"]
        nostim_trials = context_trials[context_trials["trial_type"] == "no_stim_trial"]
        
        whisker_array, clusters = create_array_spikes(units, whisker_trials, type='whisker', start=start, stop=stop)
        auditory_array, _ = create_array_spikes(units, auditory_trials, type='auditory', start=start, stop=stop)
        nostim_array, _ = create_array_spikes(units, nostim_trials, type='no_stim', start=start, stop=stop)
        
        spike_data[context] = {
            "whisker": whisker_array,
            "auditory": auditory_array,
            "nostim": nostim_array
        }
    
    return spike_data, clusters


def plot_raster_final_context(spike_data, cluster, mouse_name=''):
    """
    Plots raster plots for passive and active contexts side by side for a given cluster.
    Includes a legend to indicate the color mapping for trial types.
    """
    filename = f"Raster_Neuron_{cluster}.png"
    folder = f'plots/{mouse_name}/raster_plots'
    os.makedirs(folder, exist_ok=True)

    # Create a figure with two subplots (one for passive, one for active)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    contexts = ['passive', 'active']

    for ax, context in zip(axes, contexts):
        nostim_test = spike_data[context]["nostim"]
        auditory_test = spike_data[context]["auditory"]
        whisker_test = spike_data[context]["whisker"]

        # Plot no stimulation test spikes
        for idx, spikes in enumerate(nostim_test):
            ax.scatter(spikes, np.full_like(spikes, idx), color='k', marker='|', s=15, label='No Stim' if idx == 0 else "")

        # Plot auditory test spikes
        for idx, spikes in enumerate(auditory_test):
            ax.scatter(spikes, np.full_like(spikes, idx + len(nostim_test)), color='mediumblue', marker='|', s=15, label='Auditory' if idx == 0 else "")

        # Plot whisker test spikes
        for idx, spikes in enumerate(whisker_test):
            ax.scatter(spikes, np.full_like(spikes, idx + len(nostim_test) + len(auditory_test)), color='forestgreen', marker='|', s=15, label='Whisker' if idx == 0 else "")

        # Add vertical lines and labels
        ax.axvline(x=-0.2, color='grey', linestyle='--', linewidth=2, label='-0.2 s' if idx == 0 else "")
        ax.axvline(x=0.2, color='grey', linestyle='--', linewidth=2, label='0.2 s' if idx == 0 else "")
        ax.axvline(x=0.0, color='red', linestyle='-', linewidth=2, label='Trial Start' if idx == 0 else "")

        ax.set_title(f'{context.capitalize()} Context')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Trial Index')
        ax.legend(loc='upper right')  # Add legend to each subplot

    plt.suptitle(f'Raster Plot of Spike Times for Cluster {cluster}')
    plt.savefig(os.path.join(folder, filename), bbox_inches='tight')
    plt.close()



def Raster_total_context(units, trials, start=0.5, stop=1, mouse_name=''):
    """
    Generates raster plots for all clusters, split by passive and active contexts.
    """
    spike_data, clusters = Final_spikes_context(units, trials, start, stop)
    Nb_neurons = len(spike_data['passive']["whisker"])
    for i in range(Nb_neurons):
        plot_raster_final_context(
            {
                "passive": {
                    "whisker": spike_data["passive"]["whisker"][i],
                    "auditory": spike_data["passive"]["auditory"][i],
                    "nostim": spike_data["passive"]["nostim"][i],
                },
                "active": {
                    "whisker": spike_data["active"]["whisker"][i],
                    "auditory": spike_data["active"]["auditory"][i],
                    "nostim": spike_data["active"]["nostim"][i],
                }
            },
            clusters[i],
            mouse_name
        )

