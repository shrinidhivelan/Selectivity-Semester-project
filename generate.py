import pandas as pd
import os
from pynwb import NWBHDF5IO
from pathlib import Path
from plot import *
from AUC import *
from helpers import *



def generate_mice_data(folder_path, save_path):
    """
    Process NWB files in a folder to extract selectivity data for multiple mice.

    Args:
        folder_path (str or Path): Path to the folder containing NWB files.
    
    Returns:
        list: A list of mouse IDs extracted from the processed files.
    """

    # Store mouse names for return
    mouse_names = []

    # Store individual mouse data for final concatenation
    mice_data = []

    # Convert folder_path to a Path object if it's not already
    Path_folder = Path(folder_path) 
    
    if not any(Path_folder.glob("*.nwb")):
        raise FileNotFoundError(f"No NWB files found in the directory: {folder_path}")

    for filepath in Path_folder.glob("*.nwb"):
        print(filepath)
        
        if not filepath.name.startswith("._"):  # Skip hidden files 

            print(f"Processing file: {filepath.name}")

            # Extract mouse name from the file name (excluding extension)    
            mouse_name = filepath.name[:-4]  
            mouse_names.append(mouse_name)

            # Read nwbfile
            io = NWBHDF5IO(str(filepath), 'r') 
            nwbfile = io.read()

            
            # Generate data using the spike_detect function from the nwbfile: 
            data_total = spike_detect(nwbfile, 0.2, 0.2)

            # Metadata columns
            data_total['pre_time'] = 0.2
            data_total['post_time'] = 0.2
            data_total['mouse_id'] = mouse_name 

            # Drop problematic columns, if they exist
            if 'electrode_group' in data_total.columns:
                data_total = data_total.drop(columns=['electrode_group'])

            # Create and save individual mouse data to a parquet file
            folder = f'{save_path}/{mouse_name}'
            os.makedirs(folder, exist_ok=True)
            data_total.to_parquet(f'{folder}/{mouse_name}_Selectivity_Dataframe2.parquet', index=False)

            # Append to main mice_data 
            mice_data.append(data_total)

    # Combine all event DataFrames into a single DataFrame
    df_combined = pd.concat(mice_data).reset_index(drop=True)

    # Save the combined data to an overall folder
    os.makedirs(f'{save_path}/Overall', exist_ok=True)
    df_combined.to_parquet(f'{save_path}/Overall/Mice_Selectivity_Dataframe2.parquet', index=False)
    return mouse_names


def AUC_generate(mouse_names = [], save_path = "", start = 0.2, stop = 0.2):
    """
    Generate AUC (Area Under Curve) data for multiple mice, including bootstrapping and saving results.

    Args:
        mouse_names (list): List of mouse names to process.
        save_files (bool): Whether to save the processed files.
        visualize (bool): Whether to visualize the AUC data.
        nb_neurons (int): Number of neurons to visualize (if `visualize` is True).
        pre_vs_post_visualization (bool): Placeholder for potential visualization comparison (not implemented here).
        start (float): Start time for analysis window (pre-event).
        stop (float): Stop time for analysis window (post-event).

    Returns:
        pd.DataFrame: Combined DataFrame with AUC results for all mice.
    """
    mice_data = []  # List to store processed DataFrames for each mouse
    df_combined = []  # Placeholder for the final combined DataFrame

    for i, mouse_name in enumerate(mouse_names):
        df = pd.read_parquet(f'{save_path}/{mouse_name}/{mouse_name}_Selectivity_Dataframe2.parquet')

        print(f"Starting process for Mouse {i+1}/{len(mouse_names)} {mouse_name}")
        
        # Calculate AUC for each type of stimulus
        for type in ['whisker', 'auditory', 'wh_vs_aud', 'spontaneous_licks']:
            
            if type == 'spontaneous_licks':
                # Compute AUC and transform values for spontaneous licks
                df['spontaneous_licks_AUC'] = df.apply(lambda row: compute_AUC(row, type, context), axis=1)
                df['Transformed spontaneous_licks_AUC'] = df.apply(lambda row: 2*row[f'spontaneous_licks_AUC']-1, axis=1)

            else:
                # Compute AUC and transform values for other stimulus types across contexts
                contexts = ['passive', 'active']
                
            for context in contexts:
                df[f'{type}_{context}_AUC'] = df.apply(lambda row: compute_AUC(row, type, context), axis=1)
                df[f'Transformed {type}_{context}_AUC'] = df.apply(lambda row: 2*row[f'{type}_{context}_AUC']-1, axis=1)

        # Bootstrapping process for statistical analysis
        print("Starting bootstrapping process...")
        new_df = bootstrapping(df)

        # Create a combined DataFrame for visualization or saving
        print('Pivotting table...')
        combined_df = create_combined_df_v6(new_df)
        
        # Add metadata to the combined DataFrame
        combined_df['mouse_id'] = mouse_name
        combined_df['pre_time'] = start
        combined_df['post_time'] = stop


        ### save separate parquet files for each mouse :
        combined_df.to_parquet(f'{save_path}/{mouse_name}/{mouse_name}_AUC_Selectivity2.parquet', index=False)
        
        print(f"Process finished for Mouse {i+1}/{len(mouse_names)}!")
        mice_data.append(combined_df)
    

    # Combine all mouse DataFrames into a single DataFrame
    df_combined = pd.concat(mice_data).reset_index(drop=True)
    os.makedirs(f'{save_path}/Overall', exist_ok=True)
    df_combined.to_parquet(f'{save_path}/Overall/Mice_AUC_Selectivity2.parquet', index=False)
    

    return df_combined

    

