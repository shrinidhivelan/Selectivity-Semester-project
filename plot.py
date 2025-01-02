import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score 
import os
import ast
from generate import *
from AUC import *
from helpers import *
from tqdm import tqdm
import seaborn as sns


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



def plot_neuron_percentages(df, offset=2, category='whisker'):
    """
    Plots a grouped bar chart of the percentage of selective and non-selective neurons by brain region.

    Parameters:
    - df: DataFrame containing columns 'ccf_parent_acronym' (brain region) and 'selective' (boolean for neuron selectivity).
    - offset: Minimum vertical distance for percentage annotations (default=2).
    - category: Either 'whisker', 'auditory', 'wh_vs_aud', or 'spontaneous_licks'.
    """

    if category != 'all':
        df = df[df['event'] == category]

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
    plt.xticks(rotation=45)
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


def save_roc_plots_context(arr1, arr2, cluster_id, type="whisker", indices=[-9999], mouse_name = '', context = 'passive', auc_path = ''):
    # Ensure indices list covers the desired range
    total_plots = len(arr1)
    if indices[0] == -9999:
        indices = range(total_plots)
    
    # Create folder if it doesn't exist
    if type == 'spontaneous_licks':
        if mouse_name == '':
            folder_path = f'{auc_path}/Plots/AUC_plots/other/{type}/'
        else:
            folder_path = f'{auc_path}/Plots/AUC_plots/{mouse_name}/{type}/'
    else:
        if mouse_name == '':
            folder_path = f'{auc_path}/Plots/AUC_plots/other/{type}/{context}/'
        else:
            folder_path = f'{auc_path}/Plots/AUC_plots/{mouse_name}/{type}/{context}'

    os.makedirs(folder_path, exist_ok=True)
    
    # Generate and save each ROC plot
    for i in tqdm(indices):
        fig, ax = plt.subplots(figsize=(6, 6))  # Create a new figure for each plot
        plot_roc_curve(ax, arr1, arr2, i, cluster_id, type)
        
        # Define file path and save the figure
        file_path = f"{folder_path}/roc_neuron_{i}_cluster_{cluster_id[i]}_{mouse_name}.png"
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory

def process_and_save_roc(mouse_name, main_path, has_context = True, types = ['whisker', 'auditory', 'wh_vs_aud', 'spontaneous_licks'], contexts = ['active', 'passive_pre', 'passive_post']):
    """
    Process types and contexts, generating and saving ROC plots.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        types (list): List of types to process (e.g., 'whisker', 'auditory').
        contexts (list): List of contexts to process (e.g., 'active', 'passive').
        cluster_id (pd.Series): Series of cluster IDs.
        mouse_name (str): Identifier for the mouse.
        main_path (str): Base path to save the ROC plots.
    """

    if not has_context:
        contexts = ['active']

    df = pd.read_parquet(main_path+"/"+mouse_name+"/"+mouse_name+'_Selectivity_Dataframe.parquet')
    cluster_id = df['cluster_id']

    
    for type in tqdm(types, desc="Processing types", dynamic_ncols=True):
        # Determine the appropriate contexts for the current type
        if type == 'spontaneous_licks':
            context_list = ['']  # Override contexts for this type
        else:
            context_list = contexts

        # Ensure the inner progress bar is visible
        for context in tqdm(context_list, desc=f"Processing {type} contexts", dynamic_ncols=True):
            if type == 'wh_vs_aud':
                save_roc_plots_context(
                    df[f"whisker_{context}_post_spikes"], df[f"auditory_{context}_post_spikes"], cluster_id,
                    type, mouse_name=mouse_name, context=context, auc_path=main_path
                )
            elif type == 'spontaneous_licks':
                save_roc_plots_context(
                    df[f"{type}_post_spikes"], df[f"{type}_post_spikes"], cluster_id,
                    type, mouse_name=mouse_name, context=context, auc_path=main_path
                )
            else:
                save_roc_plots_context(
                    df[f"{type}_{context}_pre_spikes"], df[f"{type}_{context}_post_spikes"], cluster_id,
                    type, mouse_name=mouse_name, context=context, auc_path=main_path
                )



def plot_single_roc(pre, post, cluster_id, index, type="whisker", context='passive'):
    # Convert strings to lists if needed
    if isinstance(pre, str):
        pre = ast.literal_eval(pre)
    if isinstance(post, str):
        post = ast.literal_eval(post)

    # Ensure whisker_pre and whisker_post are arrays
    pre = np.array(pre)
    post = np.array(post)

    # Combine data for the specific index
    spike_counts = np.concatenate([pre, post])
    len_per_element = len(pre)
    len_per_element2 = len(post)
    labels = np.concatenate([np.ones(len_per_element), np.zeros(len_per_element2)])
    
    # Compute the ROC curve
    fpr, tpr, _ = roc_curve(labels, spike_counts)
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

def compute_trial_baseline_from_peth(peri_stim_hist, bas_start, bas_stop):
    """
    Computes the baseline firing rate for each trial based on the peri-stimulus time histogram.
    :param peri_stim_hist: Array of shape (N_trials, N_bins) with spike counts.
    :param bas_start: Start index of the baseline period.
    :param bas_stop: Stop index of the baseline period.
    :return: Vector of size N_trials with mean baseline firing rates for each trial.
    """
    # Extract the baseline period and compute the mean for each trial
    baseline_period = peri_stim_hist[:, bas_start:bas_stop]
    trial_baselines = np.mean(baseline_period, axis=1)
    return trial_baselines


def compute_unit_peri_event_histogram(spike_times, event_times, bin_size, time_start, time_stop, artifact_correction=False):
    """
    Computes peri-stimulus time histogram for a single unit.
    :param spike_times:  Spike times in seconds.
    :param event_times:  Stimulus times in seconds.
    :param bin_size: Bin size in seconds.
    :param time_start: Start of peri-stimulus time window.
    :param time_stop: End of peri-stimulus time window.
    :param artifact_correction: Boolean to apply artifact correction.
    :return: Peri-stimulus time histogram of spike counts.
    """

    # Initialize histogram
    if artifact_correction:
        bin_size_hist = 0.001
        n_bins = int((time_stop - time_start) / 0.001)
    else:
        bin_size_hist = bin_size
        n_bins = int((time_stop - time_start) / bin_size)
        
    peri_stim_hist = np.zeros((len(event_times), n_bins))
        
    # Compute histogram
    for i, stim_time in enumerate(event_times):
        spike_times_in_window = spike_times[(spike_times >= stim_time + time_start) & (spike_times < stim_time + time_stop)]
        spike_times_in_window -= stim_time # align
        spike_counts = np.histogram(spike_times_in_window, bins=np.arange(time_start, time_stop + bin_size_hist, bin_size_hist), density=False)[0]
        peri_stim_hist[i,:] = spike_counts # add counts

    if artifact_correction:
        # Bin to correct for artifact
        if bin_size_hist==0.001:
            stim_dur = 3 # stimulus duration in msec
            art_start = -1  # ms before stim
            art_stop = stim_dur + 1    # ms after stim
            art_start_bin = int(abs(time_start) / bin_size_hist) + art_start
            art_stop_bin = int(abs(time_start) / bin_size_hist) + art_stop

        # Get baseline firing rate from PETH
        bas_stop = int(abs(time_start) / bin_size_hist) - 5 # 5 time bins before stim
        trial_baselines = compute_trial_baseline_from_peth(peri_stim_hist,
                                                           bas_start=0,
                                                           bas_stop=bas_stop)

        # Make Poisson noise based on baseline firing rate
        rng = np.random.default_rng(seed=None)  # no seed for variability
        poisson_noise = [rng.poisson(lam=trial_baselines[i], size=n_bins) for i in range(len(event_times))]
        poisson_noise = np.array(poisson_noise)

        # Replace spike counts with Poisson noise in artifact window
        try:
            peri_stim_hist[:, art_start_bin:art_stop_bin] = poisson_noise[:, art_start_bin:art_stop_bin]
        except IndexError:
            print('Index error in artifact correction. Skipping correction.')
            print('art_start_bin:', art_start_bin)
            return peri_stim_hist

        # Rebin to desired bin size if there was artifact correction
        if artifact_correction and bin_size != 0.001:
            # Aggregate spike counts in bin_size in ms time bins using the sum over the bins
            current_bin_size_ms = int(bin_size_hist * 1000)
            new_bin_size_ms = int(bin_size * 1000)
            n_trials = peri_stim_hist.shape[0]

            peri_stim_hist_original = peri_stim_hist.copy()
            peri_stim_hist = peri_stim_hist.reshape(n_trials, -1, new_bin_size_ms // current_bin_size_ms).sum(axis=2)

            debug=False
            if debug:
                fig, ax = plt.subplots(1,1)
                time = np.linspace(time_start, time_stop, peri_stim_hist.shape[1])
                ax.plot(time,np.nanmean(peri_stim_hist_original, axis=0), c='k')
                ax.plot(time,np.nanmean(peri_stim_hist, axis=0), c='r')
                ax.axvline(0, c='k', linestyle='--')
                plt.show()
            
    return peri_stim_hist

########## Generate plots per category

#### Reformuler ceci par rapport a df final !!!!!!!!!! ####
def Final_spikes_context(units, trials, start=0.5, stop=1, has_context = True):
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
    contexts = ['passive', 'active'] if has_context else ['active'] 
    spike_data = {}
    clusters = None

    for context in contexts:
        context_trials = trials[trials["context"] == context] if has_context else trials
        whisker_trials = context_trials[context_trials["trial_type"] == "whisker_trial"]
        auditory_trials = context_trials[context_trials["trial_type"] == "auditory_trial"]
        nostim_trials = context_trials[context_trials["trial_type"] == "no_stim_trial"]
        
        whisker_array, clusters = create_array_spikes(units, whisker_trials, type='whisker', start=start, stop=stop)
        auditory_array, _ = create_array_spikes(units, auditory_trials, type='auditory', start=start, stop=stop)
        nostim_array, _ = create_array_spikes(units, nostim_trials, type='no_stim', start=start, stop=stop)
        
        """ 
        print(f"Shape whisker_array for {context} : {len(whisker_array)} ")
        print(f"Shape auditory_array for {context} : {len(auditory_array)} ")
        print(f"Shape nostim_array for {context} : {len(nostim_array)} ")
        """
        
        spike_data[context] = {
            "whisker": whisker_array,
            "auditory": auditory_array,
            "nostim": nostim_array
        }
    
    return spike_data, clusters


def plot_raster_final_context(spike_data, cluster, mouse_name='', raster_path='', has_context = True):
    """
    Plots raster plots for passive and active contexts side by side for a given cluster.
    Includes a legend to indicate the color mapping for trial types.
    """
    filename = f"Raster_Neuron_{cluster}_{mouse_name}.png"
    #folder = f'plots/{mouse_name}/raster_plots'
    os.makedirs(raster_path, exist_ok=True)

    # Create a figure with two subplots (one for passive, one for active)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    contexts = ['passive', 'active'] if has_context else ['active']

    for ax, context in zip(axes, contexts):
        if context == 'passive':
            # Only plot whisker and auditory trials in passive context (no nostim)
            whisker_test = spike_data[context]["whisker"]
            auditory_test = spike_data[context]["auditory"]
            
            # Plot whisker test spikes
            for idx, spikes in enumerate(whisker_test):
                ax.scatter(spikes, np.full_like(spikes, idx), color='forestgreen', marker='|', s=10, label='Whisker' if idx == 0 else "")
                
            # Plot auditory test spikes
            for idx, spikes in enumerate(auditory_test):
                ax.scatter(spikes, np.full_like(spikes, idx + len(whisker_test)), color='mediumblue', marker='|', s=10, label='Auditory' if idx == 0 else "")
                
        else:  # active context, plot all trials including nostim
            nostim_test = spike_data[context]["nostim"]
            auditory_test = spike_data[context]["auditory"]
            whisker_test = spike_data[context]["whisker"]

            # Plot no stimulation test spikes
            for idx, spikes in enumerate(nostim_test):
                ax.scatter(spikes, np.full_like(spikes, idx), color='k', marker='|', s=10, label='No Stim' if idx == 0 else "")

            # Plot auditory test spikes
            for idx, spikes in enumerate(auditory_test):
                ax.scatter(spikes, np.full_like(spikes, idx + len(nostim_test)), color='mediumblue', marker='|', s=10, label='Auditory' if idx == 0 else "")

            # Plot whisker test spikes
            for idx, spikes in enumerate(whisker_test):
                ax.scatter(spikes, np.full_like(spikes, idx + len(nostim_test) + len(auditory_test)), color='forestgreen', marker='|', s=10, label='Whisker' if idx == 0 else "")

        # Add vertical lines and labels
        ax.axvline(x=-0.2, color='grey', linestyle='--', linewidth=2, label='-0.2 s' if idx == 0 else "")
        ax.axvline(x=0.2, color='grey', linestyle='--', linewidth=2, label='0.2 s' if idx == 0 else "")
        ax.axvline(x=0.0, color='red', linestyle='-', linewidth=2, label='Trial Start' if idx == 0 else "")

        ax.set_title(f'{context.capitalize()} Context')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Trial Index')
        ax.legend(loc='upper right')  # Add legend to each subplot

    plt.suptitle(f'Raster Plot of Spike Times for Cluster {cluster} for mouse {mouse_name}')
    plt.savefig(os.path.join(raster_path, filename), bbox_inches='tight')
    plt.close()




def Raster_total_context(nwbfile, start=0.5, stop=1, mouse_name='', main_folder = '/Volumes/LaCie/EPFL/Mastersem3/Semester Project Lsens', has_context = True):
    """
    Generates raster plots for all clusters, split by passive and active contexts.
    """
    units, trials = preprocessing(nwbfile)
    
    raster_path = main_folder+"/Plots/Rasters/"+mouse_name
    os.makedirs(raster_path, exist_ok=True)

    spike_data, clusters = Final_spikes_context(units, trials, start, stop, has_context)
    if has_context:
        Nb_neurons = len(spike_data['passive']["whisker"])
    else:
        Nb_neurons = len(spike_data['active']["whisker"])
    for i in range(Nb_neurons):
        if has_context:
            plot_raster_final_context(
                {
                    "passive": {
                        "whisker": spike_data["passive"]["whisker"][i],
                        "auditory": spike_data["passive"]["auditory"][i],
                        "nostim": [],
                    },
                    "active": {
                        "whisker": spike_data["active"]["whisker"][i],
                        "auditory": spike_data["active"]["auditory"][i],
                        "nostim": spike_data["active"]["nostim"][i],
                    }
                },
                clusters[i],
                mouse_name,
                raster_path
            )
        else:
            plot_raster_final_context(
                {
                    "active": {
                        "whisker": spike_data["active"]["whisker"][i],
                        "auditory": spike_data["active"]["auditory"][i],
                        "nostim": spike_data["active"]["nostim"][i],
                    }
                },
                clusters[i],
                mouse_name,
                raster_path
            )

    import os
import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO


def generate_psth_plots(
    mouse_name='AB122_20240804_134554',
    main_path='/Volumes/LaCie/EPFL/Mastersem3/Semester Project Lsens/',
    has_context=True,
    df=pd.DataFrame()
):
    """
    Generates PSTH plots for passive and active contexts based on the spike and event data from NWB files.
    Args:
        mouse_name (str): Name of the mouse data file
        main_path (str): Path to the main directory containing data
        has_context (bool): Boolean to determine if context-based plots should be created
        df (DataFrame): DataFrame with behavioral or spike metadata
    """
    # Preprocess context column into 'active' or 'passive'
    df['context_new'] = df['context'].apply(
        lambda x: 'active' if x == 'active' else 'passive' if x in ['passive_pre', 'passive_post', 'passive'] else None
    )

    # Define the directories for saving plots
    plots_path = os.path.join(main_path, 'Plots', 'PSTH')
    passive_path = os.path.join(plots_path, mouse_name, 'passive')
    active_path = os.path.join(plots_path, mouse_name, 'active')
    context = 'context' if has_context else 'nocontext'
    
    # Locate the NWB file
    nwbfile_path = os.path.join(main_path, 'Mice_data', context, mouse_name + '.nwb')
    
    # Open NWB file
    io = NWBHDF5IO(nwbfile_path, 'r')
    nwbfile = io.read()

    # Create directories
    if has_context:
        os.makedirs(passive_path, exist_ok=True)
    os.makedirs(active_path, exist_ok=True)

    # Preprocess the data
    units, trials = preprocessing(nwbfile)
    check = True  # Boolean flag to toggle behavior for visualization
    
    # Loop over each cluster
    for index, row in units.iterrows():
        spike_times = row['spike_times']
        cluster_nb = row['cluster_id']

        # Define save paths for plots
        passive_path_save = os.path.join(passive_path, f'{cluster_nb}_passive_cluster.png')
        active_path_save = os.path.join(active_path, f'{cluster_nb}_active_cluster.png')

        # PSTH Parameters
        bin_size = 0.01  # Bin size in seconds
        time_start = -0.5
        time_stop = 2
        artifact_correction = False

        # Dictionary to store PSTH data
        active_psths = {}
        passive_psths = {}
        active_directions = []
        passive_directions = []

        # Extract data by context and event type
        for context_str in ['active', 'passive'] if has_context else ['active']:
            for event in ['whisker', 'auditory', 'spontaneous_licks']:
                event_times = extract_event_times(nwbfile, type=event, context=context_str)
                psth = compute_unit_peri_event_histogram(
                    spike_times, event_times, bin_size, time_start, time_stop, artifact_correction
                )

                mean_psth = np.mean(psth, axis=0)
                if event == 'spontaneous_licks':
                    filtered_df = df[
                    (df['event'] == event) &
                    (df['mouse_id'] == mouse_name) &
                    (df['cluster_id'] == cluster_nb)
                    ]
                else:
                    filtered_df = df[
                        (df['context_new'] == context_str) &
                        (df['event'] == event) &
                        (df['mouse_id'] == mouse_name) &
                        (df['cluster_id'] == cluster_nb)
                    ]

                print(cluster_nb)

                # Handle case with no matching DataFrame entries
                if not filtered_df.empty:
                    direction_value = filtered_df['direction'].iloc[0]
                else:
                    direction_value = None
                    #print("No matching rows found for:", event, context_str)

                # Determine if direction is negative
                direction = (direction_value == 'negative') if direction_value else False

                # Store data based on context
                if context_str == 'active':
                    active_psths[event] = mean_psth
                    active_directions.append(direction)
                elif context_str == 'passive':
                    passive_psths[event] = mean_psth
                    passive_directions.append(direction)

        # Create time bins
        time_bins = np.arange(time_start, time_stop, bin_size)[:len(mean_psth)]

        # Plotting Active PSTH
        plt.figure(figsize=(10, 5))
        for i, (event, mean_psth) in enumerate(active_psths.items()):
            if active_directions[i]:
                mean_psth = -mean_psth  # Negate PSTH if the condition is True
            plt.plot(time_bins, mean_psth, linewidth=1, alpha=0.7, label=event.capitalize())
        plt.axvline(0, linestyle='--', color='gray', linewidth=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel('Mean Spike Count')
        title = 'Active PSTHs' if check else 'Active PSTHs (-)'
        plt.title(title)
        plt.legend(title="Event Type")
        plt.tight_layout()
        plt.savefig(active_path_save, format="png")
        plt.close()

        # Plotting Passive PSTH only if has_context is True
        if has_context:
            plt.figure(figsize=(10, 5))
            for j, (event, mean_psth) in enumerate(passive_psths.items()):
                if passive_directions[j]:
                    mean_psth = -mean_psth
                plt.plot(time_bins, mean_psth, linewidth=1, alpha=0.7, label=event.capitalize())
            plt.axvline(0, linestyle='--', color='gray', linewidth=0.8)
            plt.xlabel('Time (s)')
            plt.ylabel('Mean Spike Count')
            title2 = 'Passive PSTHs' if check else 'Passive PSTHs (-)'
            plt.legend(title=title2)
            plt.tight_layout()
            plt.savefig(passive_path_save, format="png")
            plt.close()


def plot_selectivity(df, offset=2, category='whisker', context='active', has_context=0, over_mouse=False):
    """
    Plots the average percentage of selective neurons per brain region across all mice.
    
    Parameters:
    - df: DataFrame with 'mouse_id', 'area_acronym' (brain region), and 'selective'.
    - offset: Distance for percentage annotation above the bars (default=2).
    - category: Filter based on category (default='whisker').
    """
    if category == 'spontaneous_licks':
        # Filter by category
        df_filtered = df[(df['event'] == category)]
    else:
        # Filter by category and context
        df_filtered = df[(df['event'] == category) & (df['context'] == context)]

    if (has_context == True) or (has_context == False):
        df_filtered = df_filtered[df_filtered['has context'] == has_context]
    
    # Drop NaN values in 'selective' column
    df_filtered = df_filtered.dropna(subset=['selective'])

    # Ensure 'selective' is a binary column (True/False)
    df_filtered['selective'] = df_filtered['selective'].astype(bool)

    # Calculate the percentage of selective neurons per mouse and brain region
    if over_mouse:
        percentages = df_filtered.groupby(['area_acronym', 'mouse_id'])['selective'].mean().reset_index()
    else:
        percentages = df_filtered.groupby(['area_acronym'])['selective'].mean().reset_index()
    percentages['selective'] *= 100  # Convert to percentage

    # Average percentages across mice for each brain region
    avg_percentages = percentages.groupby('area_acronym')['selective'].mean().reset_index()

    # Get the unique order of brain regions (area_acronym)
    category_order = avg_percentages['area_acronym'].tolist()

    # Create the bar plot with explicit order
    plt.figure(figsize=(10, 5))  # Set figure size
    ax = sns.barplot(x='area_acronym', y='selective', data=avg_percentages, palette='viridis', order=category_order)

    # Add annotations to the bars
    for i, row in avg_percentages.iterrows():
        ax.text(i, row['selective'] + offset, f"{row['selective']:.2f}%", ha='center', va='bottom', color='black')

    # Title and labels
    title = (f"Percentage of Selective Neurons by Brain Region for {category}"
             if category == 'spontaneous_licks' else
             f"Percentage of Selective Neurons by Brain Region for {category} with {context}")
    plt.title(title)
    plt.xlabel('Brain Region')
    plt.ylabel('Percentage of Selective Neurons')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90, ha='center')

    # Tight layout
    plt.tight_layout()

    # Show the plot
    plt.show()




import matplotlib.pyplot as plt
import seaborn as sns

def plot_selectivity_direction_mice(df, event=''):

    # If filtering by event, only take rows corresponding to the given event
    if event != '':
        df = df[df['event'] == event]

    # Group by mouse_id and direction to calculate counts
    selective_counts = df.groupby(['mouse_id', 'direction']).size().reset_index(name='count')

    # Total counts for normalization purposes
    total_counts = selective_counts.groupby('mouse_id')['count'].sum().reset_index(name='total')

    # Merge to compute percentage of selectivity per mouse
    selective_percentages = selective_counts.merge(total_counts, on='mouse_id')
    selective_percentages['percentage'] = (selective_percentages['count'] / selective_percentages['total']) * 100

    # Adjust visualization values:
    # - Keep positive responses as-is
    # - For negative responses, map their magnitude to "below the baseline"
    selective_percentages['visual_percentage'] = selective_percentages.apply(
        lambda row: -row['percentage'] if row['direction'] == 'negative' else row['percentage'],
        axis=1
    )

    # Plotting
    plt.figure(figsize=(15, 8))

    # Create the bar plot with seaborn
    ax = sns.barplot(
        data=selective_percentages,
        x='mouse_id',
        y='visual_percentage',
        hue='direction',
        dodge=True,
        palette='viridis'
    )

    # Annotate the bars
    for p in ax.patches:
        percentage = f"{abs(p.get_height()):.1f}%"  # Always annotate using magnitude
        ax.annotate(
            percentage,
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='bottom',
            fontsize=9,
            color='black',
            xytext=(0, 5),
            textcoords='offset points'
        )

    # Adjust legends to avoid duplicates
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), title='Direction')

    # Set axis labels and titles
    if event != '':
        title = f'Percentage of Selective Neurons per Mouse for {event}'
    else:
        title = 'Percentage of Selective Neurons per Mouse'

    plt.axhline(0, color='black', linewidth=0.8)  # Reference line at baseline
    plt.title(title)
    plt.ylabel('Percentage (%)')
    plt.xlabel('Mouse ID')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()




def plot_selectivity_direction_event_direction(df, event=''):


    # Group by mouse_id and direction to counts
    selective_counts = df.groupby(['event', 'direction']).size().reset_index(name='count')

    # Total counts for normalization purposes
    total_counts = selective_counts.groupby('event')['count'].sum().reset_index(name='total')

    # Merge to compute percentage of selectivity per mouse
    selective_percentages = selective_counts.merge(total_counts, on='event')
    selective_percentages['percentage'] = (selective_percentages['count'] / selective_percentages['total']) * 100

    # Adjust visualization values:
    # - Keep positive responses as-is
    # - For negative responses, map their magnitude to "below the baseline"
    selective_percentages['visual_percentage'] = selective_percentages.apply(
        lambda row: -row['percentage'] if row['direction'] == 'negative' else row['percentage'],
        axis=1
    )

    # Plotting
    plt.figure(figsize=(15, 8))

    # Create the bar plot with seaborn
    ax = sns.barplot(
        data=selective_percentages,
        x='event',
        y='visual_percentage',
        hue='direction',
        dodge=True,
        palette='viridis'
    )

    # Annotate the bars
    for p in ax.patches:
        percentage = f"{abs(p.get_height()):.1f}%"  # Always annotate using magnitude
        ax.annotate(
            percentage,
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='bottom',
            fontsize=9,
            color='black',
            xytext=(0, 5),
            textcoords='offset points'
        )

    # Adjust legends to avoid duplicates
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), title='Direction')

    # Set axis labels and titles
    if event != '':
        title = f'Percentage of Selective Neurons per Mouse for {event}'
    else:
        title = 'Percentage of Selective Neurons per Mouse'

    plt.axhline(0, color='black', linewidth=0.8)  # Reference line at baseline
    plt.title(title)
    plt.ylabel('Percentage (%)')
    plt.xlabel('Event')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


