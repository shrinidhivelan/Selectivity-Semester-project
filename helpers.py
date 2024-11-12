import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score 
import os





        
def spike_detect(unit_table, trials, start, stop, nwbfile):
    
    def extract_event_times():
        trials = nwbfile.trials.to_dataframe()
        #if type == 'spontaneous_licks':
        if type == 'lick_stim':
            _, event_time = filtered_lick_times(nwbfile, 1)
        else:
            event_time =  trials[trials[type + '_stim'] == 1]['start_time'].values
        return event_time

    # Helper function to count spikes within a given time window
    def count_spikes_in_window(spike_times, start_time, end_time):
        return len(spike_times[(spike_times >= start_time) & (spike_times <= end_time)])
    
    table = unit_table.copy()
    types = ["whisker", "auditory", "lick_stim"]

    for type in types:
        if type + '_pre_spikes' not in table.columns:
            table[type + '_pre_spikes'] = [[] for _ in range(len(table))]
        if type + '_post_spikes' not in table.columns:
            table[type + '_post_spikes'] = [[] for _ in range(len(table))]
        

        
        event_times = extract_event_times()

        for unit_id, row in table.iterrows():
            spike_times = row['spike_times']
            pre_spikes, post_spikes = [], []

            # Calculate pre- and post-spike counts for each lick time
            for event in event_times:
                # Pre-stimulus window
                pre_spikes.append(count_spikes_in_window(spike_times, event - start, event))
                # Post-stimulus window
                post_spikes.append(count_spikes_in_window(spike_times, event, event + stop))

            # Assign lists to the DataFrame columns
            table.at[unit_id, type + '_pre_spikes'] = pre_spikes
            table.at[unit_id, type + '_post_spikes'] = post_spikes
    return table


def spike_detection(table, trials, type='whisker', start=0.2, stop=0.2, file=''):
    # use either trials table 
    # Ensure columns for pre- and post-spikes are initialized as lists
    if type + '_pre_spikes' not in table.columns:
        table[type + '_pre_spikes'] = [[] for _ in range(len(table))]
    if type + '_post_spikes' not in table.columns:
        table[type + '_post_spikes'] = [[] for _ in range(len(table))]

    # Helper function to count spikes within a given time window
    def count_spikes_in_window(spike_times, start_time, end_time):
        return len(spike_times[(spike_times >= start_time) & (spike_times <= end_time)])

    # If type is 'lick_stim', use the filtered lick times
    if type == 'lick_stim':
        assert file, "File is empty, give a nwbfile!"
        _, filtered_lick = filtered_lick_times(file, 1)
        
        for unit_id, row in table.iterrows():
            spike_times = row['spike_times']
            pre_spikes, post_spikes = [], []

            # Calculate pre- and post-spike counts for each lick time
            for lick_time in filtered_lick:
                # Pre-stimulus window
                pre_spikes.append(count_spikes_in_window(spike_times, lick_time - start, lick_time))
                # Post-stimulus window
                post_spikes.append(count_spikes_in_window(spike_times, lick_time, lick_time + stop))

            # Assign lists to the DataFrame columns
            table.at[unit_id, type + '_pre_spikes'] = pre_spikes
            table.at[unit_id, type + '_post_spikes'] = post_spikes

    # For other stimulus types
    else:
        for unit_id, row in table.iterrows():
            spike_times = row['spike_times']
            pre_spikes, post_spikes = [], []

            for _, trial in trials.iterrows():
                if trial[type + '_stim'] == 1:
                    # Pre-stimulus window
                    pre_spikes.append(count_spikes_in_window(spike_times, trial['start_time'] - start, trial['start_time']))
                    # Post-stimulus window
                    post_spikes.append(count_spikes_in_window(spike_times, trial['start_time'], trial['start_time'] + stop))

            # Assign lists to the DataFrame columns
            table.at[unit_id, type + '_pre_spikes'] = pre_spikes
            table.at[unit_id, type + '_post_spikes'] = post_spikes

    return table
    # return proc_data



def spike_raster(table, trials, type='whisker', start=0.5, stop=1):
    relative_unit_pre = []
    relative_unit_post = []
    trials_pre = []
    trials_post = []

    for unit_id, row in table.iterrows():
        spike_times = row['spike_times']
        relative_time1 = []
        relative_time2 = []
        trials_per_neuron_pre = []
        trials_per_neuron_post = []
        
        for trial_idx, trial in trials.iterrows():
            if trial[type + '_stim'] == 1:
                # Pre-stimulus
                trial_start_1 = trial['start_time'] - start
                trial_stop_1 = trial['start_time']
                spikes_during_trial_1 = spike_times[(spike_times >= trial_start_1) & (spike_times <= trial_stop_1)]

                #if len(spikes_during_trial_1)!=0:
                trials_per_neuron_pre.append(trial_idx)

                relative_time1.append(spikes_during_trial_1 - trial['start_time'])

                # Post-stimulus
                trial_start_2 = trial['start_time']
                trial_stop_2 = trial['start_time'] + stop
                spikes_during_trial_2 = spike_times[(spike_times >= trial_start_2) & (spike_times <= trial_stop_2)]

                #if len(spikes_during_trial_2)!=0:
                trials_per_neuron_post.append(trial_idx)

                relative_time2.append(spikes_during_trial_2 - trial['start_time'])
        
        relative_unit_pre.append(relative_time1)
        relative_unit_post.append(relative_time2)
        trials_pre.append(trials_per_neuron_pre)
        trials_post.append(trials_per_neuron_post)

    return relative_unit_pre, relative_unit_post, trials_pre, trials_post

def Create_array_spikes(table, trials, type='whisker', start=0.5, stop=1):
    relative_unit = []
    clusters = []
    # Going through each row of the units table:
    for _, row in table.iterrows():  
        spike_times = row['spike_times']
        relative_time = []
        for _, trial in trials.iterrows():  
            # Pre-stimulus
            trial_start = trial['start_time'] - start
            trial_stop = trial['start_time'] + stop
            spikes_during_trial = spike_times[(spike_times >= trial_start) & (spike_times <= trial_stop)]
            if spikes_during_trial.size != 0:
                relative_time.append(spikes_during_trial - trial['start_time'])

        # Check if relative_time is not empty before adding it to relative_unit
        if relative_time:  # This checks if relative_time has any elements
            relative_unit.append(relative_time)
            clusters.append(row['cluster_id'])
    
    return relative_unit, clusters

def Final_spikes(units, trials, start=0.5, stop=1):

    whisker_trials = trials[trials["trial_type"]=="whisker_trial"]
    auditory_trials = trials[trials["trial_type"]=="auditory_trial"]
    nostim_trials = trials[trials["trial_type"]=="no_stim_trial"]
    whisker_array, whisker_clusters = Create_array_spikes(units, whisker_trials, type='whisker', start=0.5, stop=1)
    auditory_array, auditory_clusters = Create_array_spikes(units, auditory_trials, type='auditory', start=0.5, stop=1)
    nostim_array, no_stim_clusters = Create_array_spikes(units, nostim_trials, type='no_stim', start=0.5, stop=1)
    
    return whisker_array, auditory_array, nostim_array, whisker_clusters


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


    # Add labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('Trial Index')
    plt.title(f'Raster Plot of Spike Times of cluster {cluster}')

    # Add legend
    #plt.legend()

    # Save the plot to the "rasters" folder
    plt.savefig(os.path.join(folder, filename), bbox_inches='tight')
    plt.close()  # Close the figure to free up memory


def Raster_total(units, trials, start = 0.5, stop = 1, mouse_name = ''):

    whisker_array, auditory_array, nostim_array, whisker_clusters = Final_spikes(units, trials, start, stop)
    Nb_neurons = len(whisker_array)
    for i in range(Nb_neurons):
        plot_raster_final(whisker_array[i], auditory_array[i], nostim_array[i], whisker_clusters[i], mouse_name)

def plot_raster(data, trials, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get spike times for whisker and auditory
    whisker_spike_times_1, whisker_spike_times_2, w_trials_pre, w_trials_post = spike_raster(data, trials)
    auditory_spike_times_1, auditory_spike_times_2, a_trials_pre, a_trials_post = spike_raster(data, trials, "auditory")
    cluster_values = data["cluster_id"].values

    for unit_id in range(len(data)):
        fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        colors = {'whisker_pre': 'forestgreen', 'whisker_post': 'green', 'auditory_pre':'mediumblue', 'auditory_post': 'blue'}

        # Adjust the space between plots to zero
        plt.subplots_adjust(hspace=0)

        # Define the two subplots
        for ax, (stim_times_1, stim_times_2, trials_pre, trials_post, stim_type) in zip(
            axes, 
            [(whisker_spike_times_1, whisker_spike_times_2, w_trials_pre, w_trials_post, 'Whisker'),
             (auditory_spike_times_1, auditory_spike_times_2, a_trials_pre, a_trials_post, 'Auditory')]
        ):
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
            # Plot pre-stimulus spikes
            for trial_idx, spikes in enumerate(stim_times_1[unit_id]):
                trial_pre = trials_pre[unit_id][trial_idx]
                ax.scatter(spikes, np.ones_like(spikes) * trial_pre, color=colors[f'{stim_type.lower()}_pre'], marker='|', label=f'{stim_type} Pre' if trial_idx == 0 else "")
            # Plot post-stimulus spikes
            for trial_idx, spikes in enumerate(stim_times_2[unit_id]):
                trial_post = trials_post[unit_id][trial_idx]
                ax.scatter(spikes, np.ones_like(spikes) * trial_post, color=colors[f'{stim_type.lower()}_post'], marker='|', label=f'{stim_type} Post' if trial_idx == 0 else "")

            ax.set_xlim(-0.5, 1.0)
            ax.set_ylabel('Trials')
            ax.legend(title=f'{stim_type} Stimulus')
            
            # Invert the y-axis
            ax.invert_yaxis()

        # Set labels and title
        axes[1].set_xlabel('Time (s) relative to stimulus onset')
        fig.suptitle(f'Raster Plots for Neuron {unit_id} & Cluster {cluster_values[unit_id]}')

        # Save plot to file
        plt.savefig(os.path.join(output_folder, f'raster_plot_cluster_{cluster_values[unit_id]}.png'))
        plt.close(fig)  # Close the figure to save memory


def event_plot(data, trials, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get spike times for whisker and auditory
    whisker_spike_times_1, whisker_spike_times_2, w_trials_pre, w_trials_post = spike_raster(data, trials)
    auditory_spike_times_1, auditory_spike_times_2, a_trials_pre, a_trials_post = spike_raster(data, trials, "auditory")
    cluster_values = data["cluster_id"].values
    index_values = trials.index.values

    # Loop over each neuron
    for unit_id in range(len(data)):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid for pre and post plots

        # Plot whisker spikes (pre)
        whisker_pre = whisker_spike_times_1[unit_id]
        trial_whisker_pre = w_trials_pre[unit_id]
        colors1 = ['C{}'.format(i) for i in range(len(whisker_pre))]
        axs[0, 0].eventplot(whisker_pre, colors=colors1, lineoffsets=trial_whisker_pre, linelengths=5)
        axs[0, 0].set_title("Whisker Stim - Pre")
        axs[0, 0].set_xlabel("Time")
        axs[0, 0].set_ylabel("Trials")

        # Plot whisker spikes (post)
        whisker_post = whisker_spike_times_2[unit_id]
        trial_whisker_post = w_trials_post[unit_id]
        colors2 = ['C{}'.format(i) for i in range(len(whisker_post))]
        axs[0, 1].eventplot(whisker_post, colors=colors2, lineoffsets=trial_whisker_post, linelengths=5)
        axs[0, 1].set_title("Whisker Stim - Post")
        axs[0, 1].set_xlabel("Time")
        axs[0, 1].set_ylabel("Trials")

        # Plot auditory spikes (pre)
        auditory_pre = auditory_spike_times_1[unit_id]
        trial_auditory_pre = a_trials_pre[unit_id]
        colors3 = ['C{}'.format(i) for i in range(len(auditory_pre))]
        axs[1, 0].eventplot(auditory_pre, colors=colors3, lineoffsets=trial_auditory_pre, linelengths=5)
        axs[1, 0].set_title("Auditory Stim - Pre")
        axs[1, 0].set_xlabel("Time")
        axs[1, 0].set_ylabel("Trials")

        # Plot auditory spikes (post)
        auditory_post = auditory_spike_times_2[unit_id]
        trial_auditory_post = a_trials_post[unit_id]
        colors4 = ['C{}'.format(i) for i in range(len(auditory_post))]
        axs[1, 1].eventplot(auditory_post, colors=colors4, lineoffsets=trial_auditory_post, linelengths=5)
        axs[1, 1].set_title("Auditory Stim - Post")
        axs[1, 1].set_xlabel("Time")
        axs[1, 1].set_ylabel("Trials")

        # Adjust layout
        plt.tight_layout()

        output_path = os.path.join(output_folder, f'event_plot_{cluster_values[unit_id]}.png')
        plt.savefig(output_path)
        plt.close(fig)  


def plot_event(data, trials, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get spike times for whisker and auditory
    whisker_spike_times_1, whisker_spike_times_2, w_trials_pre, w_trials_post = spike_raster(data, trials)
    auditory_spike_times_1, auditory_spike_times_2, a_trials_pre, a_trials_post = spike_raster(data, trials, "auditory")
    cluster_values = data["cluster_id"].values

    for unit_id in range(len(data)):
        plt.figure(figsize=(10, 6))
        colors = {'whisker_pre': 'forestgreen', 'whisker_post': 'yellow', 
                  'auditory_pre': 'mediumblue', 'auditory_post': 'red'}

        # Prepare lists to hold spike times for event plot
        event_times = []

        # Collect whisker spikes
        for trial_idx, spikes in enumerate(whisker_spike_times_1[unit_id]):
            trial_whisker_pre = w_trials_pre[unit_id][trial_idx]
            if len(spikes) > 0:  # Check for empty spikes
                event_times.append((spikes, [trial_whisker_pre] * len(spikes), colors['whisker_pre']))

        for trial_idx, spikes in enumerate(whisker_spike_times_2[unit_id]):
            trial_whisker_post = w_trials_post[unit_id][trial_idx]
            if len(spikes) > 0:  # Check for empty spikes
                event_times.append((spikes, [trial_whisker_post] * len(spikes), colors['whisker_post']))

        # Collect auditory spikes
        for trial_idx, spikes in enumerate(auditory_spike_times_1[unit_id]):
            trial_auditory_pre = a_trials_pre[unit_id][trial_idx]
            if len(spikes) > 0:  # Check for empty spikes
                event_times.append((spikes, [trial_auditory_pre] * len(spikes), colors['auditory_pre']))

        for trial_idx, spikes in enumerate(auditory_spike_times_2[unit_id]):
            trial_auditory_post = a_trials_post[unit_id][trial_idx]
            if len(spikes) > 0:  # Check for empty spikes
                event_times.append((spikes, [trial_auditory_post] * len(spikes), colors['auditory_post']))

        # Flatten event times for plotting
        spikes_list = np.concatenate([spikes for spikes, _, _ in event_times])
        trials_list = np.concatenate([trials for _, trials, _ in event_times])
        colors_list = [color for _, _, color in event_times for _ in range(len(spikes))]

        # Create the event plot
        plt.eventplot(spikes_list, orientation='horizontal', colors=colors_list)

        plt.xlim(-0.5, 1.0)
        plt.xlabel('Time (s) relative to stimulus onset')
        plt.ylabel('Trials')
        plt.title(f'Event Plot for Neuron {unit_id} & Cluster {cluster_values[unit_id]}')

        # Add legend with correct labels
        handles = [
            plt.Line2D([0], [0], color=colors['whisker_pre'], lw=4, label='Whisker Pre'),
            plt.Line2D([0], [0], color=colors['whisker_post'], lw=4, label='Whisker Post'),
            plt.Line2D([0], [0], color=colors['auditory_pre'], lw=4, label='Auditory Pre'),
            plt.Line2D([0], [0], color=colors['auditory_post'], lw=4, label='Auditory Post'),
        ]
        plt.legend(handles=handles, title='Stimulus Type')

        # Save plot to file
        plt.savefig(os.path.join(output_folder, f'event_plot_cluster_{cluster_values[unit_id]}.png'))
        plt.close()  # Close the figure to save memory


def check_trials_overlap(array, trial_type_array):
    ## This function does a filter according to whether a lick time preceeds a certain type of trial or not to prevent bias
    # Filter out lick times that occur within 1 second after any whisker hit
    array_new = []
    for time in array:
        # Check if the time is within 1 second of any whisker hit
        if not any((time - hit_time <= 1) and (time - hit_time > 0) for hit_time in trial_type_array):
            array_new.append(time)
    return array_new



def lick_times(arr, interval = 1, auditory_hit = [], auditory_miss = [], whisker_hit = [], whisker_miss = []):

    arr_new = []

    # Check all other elements 
    for i in range(1, len(arr) - 1):
        if (arr[i] - arr[i - 1] > interval):
            arr_new.append(arr[i])

    # Check the final element of the table
    if len(arr) > 1 and (arr[-1] - arr[-2] > interval):
        arr_new.append(arr[-1])  
    
    ### check that each element is far enough from stim times:
    arr_new = check_trials_overlap(arr_new, auditory_hit)
    arr_new = check_trials_overlap(arr_new, auditory_miss)
    arr_new = check_trials_overlap(arr_new, whisker_hit)
    arr_new = check_trials_overlap(arr_new, whisker_miss)

    return arr_new


def filtered_lick_times(nwbfile, interval = 1):
    # Access the processing section
    processing = nwbfile.processing
    behavior = processing['behavior']

    # Access data interfaces
    data_interfaces = behavior.data_interfaces
    # Extract behavioral events and time series
    behavioral_events = data_interfaces['BehavioralEvents']  # Assuming 'BehavioralEvents' is a defined interface
    #time_series = behavior.time_series  # Access the time series data
    piezo_lick_times_series = behavioral_events.time_series['piezo_lick_times']
    piezo_lick_time = piezo_lick_times_series.data[:]  # Extract data as an array

    auditory_hit = behavioral_events.time_series['auditory_hit_trial'].data[:]
    auditory_miss = behavioral_events.time_series['auditory_miss_trial'].data[:]
    whisker_hit = behavioral_events.time_series['whisker_hit_trial'].data[:]
    whisker_miss = behavioral_events.time_series['whisker_miss_trial'].data[:]

    filtered_piezo = lick_times(piezo_lick_time, interval, auditory_hit = auditory_hit, auditory_miss = auditory_miss, whisker_hit = whisker_hit, whisker_miss = whisker_miss)
    return piezo_lick_time, filtered_piezo
# utiliser iter tools -> iterer pour tous les cas differents au lieu de considérer : pairwise 