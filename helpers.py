import pandas as pd

########### Find functions to generate the data ###########



def add_context_new_column(df):
    # Get the middle index of the dataframe
    mid_index = len(df) // 2
    
    # Create the new column 'context_new'
    df['context_new'] = df.apply(lambda row: 
                                'active' if row['context'] == 'active' else
                                ('passive_pre' if row['context'] == 'passive' and row.name < mid_index else
                                 'passive_post'), axis=1)
    
    return df


def preprocessing(nwbfile):
    """
    Preprocesses the NWB file to filter units and prepare trials data for further analysis.

    Parameters:
    - nwbfile (NWBFile): The NWB file containing the units and trials data.

    Returns:
    - filtered_units (DataFrame): Filtered DataFrame of units based on specific criteria, 
                                  containing only selected columns.
    - trials (DataFrame): DataFrame containing the trials data.
    """

    # Convert NWB units and trials tables into Pandas DataFrames
    units = nwbfile.units.to_dataframe()
    trials = nwbfile.trials.to_dataframe()

    trials = add_context_new_column(trials)

    # Preprocessing 
    filtered_units = units[(units['bc_label'] == 'good') & (units['ccf_acronym'].str.contains('[A-Z]'))]

    # Some chosen columns
    cons_columns  = ["cluster_id", "firing_rate", "ccf_acronym", "ccf_name", "ccf_parent_acronym", "ccf_parent_id","ccf_parent_name", "spike_times"]
    filtered_units = filtered_units[cons_columns]

    return filtered_units, trials





def extract_event_times(nwbfile, type = 'whisker', context = 'passive', has_context = True):
    """
    Extract event times from an NWB file based on stimulus type and context.

    Args:
        nwbfile: NWB file object.
        type (str): The type of event to extract ('whisker', 'auditory', or 'spontaneous_licks').
        context (str): The behavioral context ('passive' or 'active').

    Returns:
        np.ndarray: An array of event times matching the specified type and context.
    """
    # Convert trials table to a DataFrame
    #trials = nwbfile.trials.to_dataframe()
    _, trials = preprocessing(nwbfile)

    if type == 'spontaneous_licks':
        # For spontaneous licks, get event times using a helper function
        _, event_time = filtered_lick_times(nwbfile, 1)
    else:
        if not has_context:
            return trials[(trials[type + '_stim'] == 1)]['start_time'].values
        
        # Extract event times for specified type, context, and lick_flag
        if context == 'active':
            event_time =  trials[
                (trials[type + '_stim'] == 1) &  # Stimulus type must match
                (trials['lick_flag'] == 1) &    # Lick flag must be true
                (trials['context_new'] == context)  # Context must match
            ]['start_time'].values
        else: 
            event_time = trials[
                (trials[type + '_stim'] == 1) &  # Stimulus type must match
                (trials['context_new'] == context)  # Context must match
            ]['start_time'].values


    return event_time


def spike_detect(nwbfile, start=0.2, stop=0.2, has_context=True):
    """
    Detect spikes within pre- and post-stimulus windows for various event types and contexts.

    Args:
        nwbfile: NWB file object.
        start (float): Pre-stimulus window duration (seconds).
        stop (float): Post-stimulus window duration (seconds).

    Returns:
        pd.DataFrame: Updated unit_table with pre- and post-spike counts for each event type and context.
    """

    #### Preprocessing ####
    table, _ = preprocessing(nwbfile)
    
    # Helper function to count spikes within a given time window
    def count_spikes_in_window(spike_times, start_time, end_time):
        return len(spike_times[(spike_times >= start_time) & (spike_times <= end_time)])
    
    # Define event types and contexts
    types = ["whisker", "auditory", "spontaneous_licks"]
    #context = ["passive", "active"]

    for type in types:
        # Handle column initialization for spontaneous licks separately
        if type == 'spontaneous_licks':
            if type + '_pre_spikes' not in table.columns:
                table[type + '_pre_spikes'] = [[] for _ in range(len(table))]
            if type + '_post_spikes' not in table.columns:
                table[type + '_post_spikes'] = [[] for _ in range(len(table))]
            contexts = [""]
        
        # Initialize columns for other event types and contexts
        else:
            contexts = ["active", "passive_pre", "passive_post"] if has_context else ["active"]
            for context in contexts:
                if type + "_" + context + '_pre_spikes' not in table.columns:
                    table[type + "_" + context + '_pre_spikes'] = [[] for _ in range(len(table))]
                if type + "_" + context + '_post_spikes' not in table.columns:
                    table[type + "_" + context + '_post_spikes'] = [[] for _ in range(len(table))]

        for context in contexts:        
            event_times = extract_event_times(nwbfile, type, context, has_context)

            for unit_id, row in table.iterrows():
                spike_times = row['spike_times']
                pre_spikes, post_spikes = [], []

                # Calculate pre- and post-spike counts for each lick time
                for event in event_times:
                    # Pre-stimulus window
                    pre_spikes.append(count_spikes_in_window(spike_times, event - start, event))
                    # Post-stimulus window
                    post_spikes.append(count_spikes_in_window(spike_times, event, event + stop))

                # Assign calculated spike counts to the corresponding DataFrame columns
                if type == 'spontaneous_licks':
                    table.at[unit_id, type + '_pre_spikes'] = pre_spikes
                    table.at[unit_id, type + '_post_spikes'] = post_spikes
                else:
                    table.at[unit_id, type + "_" + context + '_pre_spikes'] = pre_spikes
                    table.at[unit_id, type + "_" + context + '_post_spikes'] = post_spikes
    return table

 

########### function to generate spikes for different types ########### 

def create_array_spikes(table, trials, type='whisker', start=0.5, stop=1):
    """
    Creates an array of relative spike times for a specified trial type.

    Parameters:
    - table (DataFrame): DataFrame containing unit information, including spike times.
    - trials (DataFrame): DataFrame containing trial information, including start times.
    - type (str): Type of trial to consider (default is 'whisker').
    - start (float): Time (in seconds) before the trial start to include spikes.
    - stop (float): Time (in seconds) after the trial start to include spikes.

    Returns:
    - relative_unit (list): A list of lists containing relative spike times for each unit.
    - clusters (list): A list of cluster IDs corresponding to each unit with relative spikes.
    """

    relative_unit = []
    clusters = []
    
    # Iterate through each row in the units table
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
            else:
                # If no spikes, append an empty list (or NaN if desired) for this trial
                relative_time.append([])  # Or relative_time.append([float('nan')])


        # Add relative spike times and cluster ID to the results if spikes were found
        if relative_time:  # This checks if relative_time has any elements
            relative_unit.append(relative_time)
            clusters.append(row['cluster_id'])
    
    return relative_unit, clusters



########### LICK TIME ANALYSIS ###########

def check_trials_overlap(array, trial_type_array):
    """
    Filters an array of event times to exclude times that occur within 1 second 
    after any event in a trial_type_array.

    Parameters:
    - array (list or ndarray): The list of event times to filter (e.g., lick times).
    - trial_type_array (list or ndarray): The list of reference event times (e.g., trial start times).

    Returns:
    - array_new (list): Filtered list of event times.
    """

    array_new = []
    
    for time in array:
        # Check if the time is within 1 second of any whisker hit
        if not any((time - hit_time <= 1) and (time - hit_time > 0) for hit_time in trial_type_array):
            array_new.append(time)
    return array_new



def lick_times(arr, interval = 1, auditory_hit = [], auditory_miss = [], whisker_hit = [], whisker_miss = []):
    """
    Filters an array of lick times to ensure:
    1. Consecutive lick times are at least `interval` seconds apart.
    2. Lick times do not occur within 1 second of any stimulus event.

    Parameters:
    - arr (list or ndarray): List of lick times to filter.
    - interval (float, optional): Minimum time difference between consecutive licks. Default is 1 second.
    - auditory_hit (list or ndarray): List of auditory hit times.
    - auditory_miss (list or ndarray): List of auditory miss times.
    - whisker_hit (list or ndarray): List of whisker hit times.
    - whisker_miss (list or ndarray): List of whisker miss times.

    Returns:
    - arr_new (list): Filtered list of lick times.
    """

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
    """
    Filters lick times from NWB data to ensure they are spaced by at least a given interval 
    and are not biased by proximity to specific stimulus events.

    Parameters:
    - nwbfile (NWBFile): The NWB file containing behavioral and trial data.
    - interval (float, optional): Minimum time difference between consecutive licks. Default is 1 second.

    Returns:
    - piezo_lick_time (ndarray): Original array of lick times.
    - filtered_piezo (list): Filtered list of lick times.
    """

    # Access the processing module and behavior data
    processing = nwbfile.processing
    behavior = processing['behavior']

    # Access data interfaces
    data_interfaces = behavior.data_interfaces
    behavioral_events = data_interfaces['BehavioralEvents']  
    piezo_lick_times_series = behavioral_events.time_series['piezo_lick_times']
    piezo_lick_time = piezo_lick_times_series.data[:]  

    auditory_hit = behavioral_events.time_series['auditory_hit_trial'].data[:]
    auditory_miss = behavioral_events.time_series['auditory_miss_trial'].data[:]
    whisker_hit = behavioral_events.time_series['whisker_hit_trial'].data[:]
    whisker_miss = behavioral_events.time_series['whisker_miss_trial'].data[:]

    filtered_piezo = lick_times(piezo_lick_time, interval, auditory_hit = auditory_hit, auditory_miss = auditory_miss, whisker_hit = whisker_hit, whisker_miss = whisker_miss)
    return piezo_lick_time, filtered_piezo


# utiliser iter tools -> iterer pour tous les cas differents au lieu de considérer : pairwise 


########### General functions for dataframe generating :###########

def convert_to_csv(mouse_names):
    main_folder = '/Volumes/LaCie/EPFL/Mastersem3/Semester Project Lsens/Data/'
    # Create a list of Parquet file paths
    file_paths = [f"{main_folder}{mouse}/{mouse}_AUC_Selectivity2.parquet" for mouse in mouse_names]
    for i, file in enumerate(file_paths):
        print(f"Doing file {i+1}/{len(file_paths)}")
        df = pd.read_parquet(file)
        df.to_csv(main_folder+mouse_names[i]+"/"+mouse_names[i]+"_AUC_Selectivity.csv", index=False)


def put_together(main_folder = '', mouse_names = ['AB124_20240815_111810','AB125_20240817_123403','AB126_20240822_114405','AB130_20240902_123634','AB129_20240828_112850','AB128_20240829_112813','AB127_20240821_103757'],
                 has_context = True):
    #mouse_names = ['AB124_20240815_111810','AB125_20240817_123403','AB126_20240822_114405','AB130_20240902_123634','AB129_20240828_112850','AB128_20240829_112813','AB127_20240821_103757',
    #           'AB123_20240806_110231', 'AB122_20240804_134554', 'AB119_20240731_102619', 'AB117_20240723_125437', 'AB116_20240724_102941']

    if has_context:
        ctxt = "_no_context"
    else:
        ctxt = ""
    #main_folder = '/Volumes/LaCie/EPFL/Mastersem3/Semester Project Lsens/Data/'

    #AB116_20240724_102941_AUC_Selectivity2.parquet

    mice_data = []

    for i, mouse in enumerate(mouse_names):
        print(f'{i+1}/{len(mouse_names)}')
        df = pd.read_csv(main_folder+"/"+mouse+"/"+mouse+"_AUC_Selectivity_pre_post.csv")
        mice_data.append(df)

    df_total = pd.concat(mice_data).reset_index(drop=True) 

    df_combined = process_area_acronyms(df_total)

    df_combined.to_csv(main_folder+'/Overall/complete_data'+ctxt+'.csv', index=False)

def combine_files(main_folder):
    
    df_no_context = pd.read_csv(main_folder+'/Overall/complete_data_no_context.csv')
    df_context = pd.read_csv(main_folder+'/Overall/complete_data.csv')
    df_no_context['has context'] = False
    df_context['has context'] = True
    df_total = pd.concat([df_no_context, df_context]).reset_index(drop=True) 
    df_total.to_csv(main_folder+'/Overall/overall_combined.csv', index=False)
 



def get_cortical_areas():
    """
    Retrieve a list of cortical area acronyms.
    :return: List of cortical area acronyms
    """
    return [
        'FRP', 'MOp', 'MOs', 'SSp-bfd', 'SSp-m', 'SSp-ul', 'SSp-ll', 'SSp-un', 'SSp-n', 'SSp-tr',
        'SSs', 'AUDp', 'AUDd', 'AUDv', 'ACA', 'ACAv', 'ACAd', 'VISa', 'VISp', 'VISam', 'VISl',
        'VISpm', 'VISrl', 'VISal', 'PL', 'ILA', 'ORB', 'RSP', 'RSPv', 'RSPd','RSPagl', 'TT', 'SCm',
        'SCsg', 'SCzo', 'SCiw', 'SCop', 'SCs', 'ORBm', 'ORBl', 'ORBvl', 'AId',
        'AIv', 'AIp', 'FRP', 'VISC'
    ]

def process_area_acronyms(peth_table): #TODO: use this function to combine future areas e.g. ACAv,ACAd -> ACA, etc.
    """
    Process and re-assign area acronyms.
    In particular, groups barrel columns together.
    :param peth_table: PETH table with all mice data
    :param params: Plotting parameters
    :return: Updated PETH table with processed area acronyms
    """
    # Assign to SSp-bfd if ccf parent acronym contains SSp-bfd
    peth_table['ccf_parent_acronym'] = peth_table['ccf_parent_acronym'].astype(str)
    peth_table['ccf_parent_acronym'] = peth_table['ccf_parent_acronym'].apply(
        lambda x: 'SSp-bfd' if 'SSp-bfd' in x else x
    )

    # Decide which area acronym to use
    # Use ccf_parent_acronym if layer or part is in the name
    peth_table['area_acronym'] = peth_table.apply(
        lambda row: row['ccf_parent_acronym']
        if ('layer' in row['ccf_name'].lower() or 'part' in row['ccf_name'].lower())
        else row['ccf_acronym'],
        axis=1
    )

    # For cortical areas, use the ccf_parent_acronym
    ctx_areas = get_cortical_areas()
    peth_table['area_acronym'] = [row['ccf_parent_acronym'] if row['ccf_parent_acronym'] in ctx_areas
                               else row['ccf_acronym'] for idx, row in peth_table.iterrows()]


    # Are there any that do not have names?
    print('List of areas without names:')
    print(peth_table[peth_table['ccf_name'].isnull()]['ccf_acronym'].unique())

    return peth_table