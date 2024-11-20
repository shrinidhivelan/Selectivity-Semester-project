import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score 
import os
from generate import *
from AUC import *



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


def AUC_plots(mouse_names = []):
    mice_data = []
    for i, mouse_name in enumerate(mouse_names):
        df = pd.read_parquet(f'Data/{mouse_name}/{mouse_name}_Selectivity_Dataframe.parquet')
        # Check whether we want to save or visualize the files:
        print("Starting to save files!")
        save_overall_auc(df, mouse_name)


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