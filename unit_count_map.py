import os
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO

import matplotlib.colors as mc
import brainglobe_heatmap as bgh

import NWB_reader_functions as nwb_reader

def plot_unit_count_map(data, output_folder, params):
    """
    Plot unit count map.
    :param data:
    :param output_folder:
    :param params:
    :return:
    """
    # Get plotting parameters
    ccf_level = params.get('ccf_level')
    colormap = params.get('cmap')
    orientation = params.get('orientation')
    vmin = params.get('vmin')
    vmax = params.get('vmax')

    # Filter data
    data = data[data['bc_label'] == 'good']

    # Format area names
    data['ccf_parent_acronym'] = data['ccf_parent_acronym'].fillna('root')

    # Exclude areas
    excluded_areas = [
        'root', 'fiber tracts', 'grey', 'nan', 'fxs', 'lfbst', 'cc', 'mfbc', 'cst', 'fa',
        'VS', 'ar', 'ccb', 'int', 'or', 'ccs', 'cing', 'ec', 'em', 'fi', 'scwm', 'alv', 'chpl', 'opt',
        'VL',
    ]
    data = data[~data['ccf_parent_acronym'].isin(excluded_areas)]

    # Create list of areas with granularity for subcortex only
    if ccf_level == 'parent_for_cortex':
        ctx_areas = [
        'FRP', 'MOp', 'MOs', 'SSp-bfd', 'SSp-m', 'SSp-ul', 'SSp-Il', 'SSp-un', 'SSp-n',
        'SSs', 'AUDp', 'AUDd', 'ACAv', 'ACAd', 'VISa', 'VISp', 'VISam', 'VISl',
        'VISpm', 'VISrl', 'VISal', 'PL', 'ILA', 'ORB', 'RSP', 'TT', 'SCm',
        'SCsg', 'SCzo', 'SCiw', 'SCop', 'SCs', 'ORBm', 'ORBl', 'ORBvl', 'AId',
        'AIv', 'AIp', 'FRP'
        ]
        # Use ccf_parent if area in cortex, ccf_acronym otherwise
        data['area_name'] = data.apply(lambda x: x['ccf_parent_acronym'] if x['ccf_parent_acronym'] in ctx_areas else x['ccf_acronym'], axis=1)
    else:
        data['area_name'] = data['ccf_parent_acronym']

    # Map barrel columns subregions to SSp-bfd
    data['area_name'] = data['area_name'].apply(lambda x: 'SSp-bfd' if 'SSp-bfd' in x else x)

    # Using brainglobe-heatmap, make a heatmap of the unit count per brain area
    unit_count_map = data.groupby('area_name').size().reset_index(name='unit_count')

    # Format as dict for each acronym
    unit_count_dict = dict(zip(unit_count_map['area_name'], unit_count_map['unit_count']))

    # Total number of well-isolated units
    total_units = sum(unit_count_dict.values())
    print(f'Total number of good units: {total_units}')

    if orientation == 'sagittal':
        position = 7000
    elif orientation == 'frontal':
        position = 5000
    elif orientation == 'horizontal':
        position = 3500

    # Plot the heatmap
    scene = bgh.Heatmap(
        values=unit_count_dict,
        position=position,
        orientation=orientation,
        thickness=2000,
        format="2D",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        label_regions=False,
        interactive=True
    )

    fig, axs = plt.subplots(1, 1, figsize=(9, 6), dpi=500)
    scene.plot_subplot(fig=fig, ax=axs, show_cbar=True, hide_axes=True)

    #scenes = []
    #for distance in range(2000, 9000, 1000):
    #    scene = bgh.Heatmap(
    #        unit_count_dict,
    #        position=distance,
    #        orientation=orientation,
    #        thickness=1000,
    #        format="2D",
    #        cmap=colormap,
    #        vmin=vmin,
    #        vmax=vmax,
    #        label_regions=False,
    #        interactive=True
    #    )
    #    scenes.append(scene)
#
    ## Create a figure with 6 subplots and plot the scenes
    #fig, axs = plt.subplots(6, 4, figsize=(18, 12), dpi=300)
    #for scene, ax in zip(scenes, axs.flatten(), strict=False):
    #    scene.plot_subplot(fig=fig, ax=ax, show_cbar=True, hide_axes=True)


    # Adjust plot
    fig.tight_layout()
    plt.show()

    ### ----------
    # Save figures
    ### ----------

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = 'unit_count_{}_{}_map'.format(orientation, colormap.lower())
    fig.savefig(os.path.join(output_folder, f'{filename}.png'), dpi=500, bbox_inches='tight')

    return


# Make a main
if __name__ == '__main__':

    # Set paths
    experimenter = 'Axel_Bisi'

    info_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'mice_info')
    output_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'results')
    root_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'NWBFull')
    proc_data_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'data',
                                  'processed_data')
    all_nwb_names = os.listdir(root_path)
    all_mwb_mice = [name.split('_')[0] for name in all_nwb_names]

    # Load recorded mouse table
    mouse_info_df = pd.read_excel(os.path.join(info_path, 'mouse_reference_weight.xlsx'))
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[mouse_info_df['exclude'] == 0]  # excluded mice
    mouse_info_df = mouse_info_df[mouse_info_df['recording'] == 1]
    subject_ids = mouse_info_df['mouse_id'].unique()

    # For each reward group, show the number of mice
    reward_groups = mouse_info_df['reward_group'].unique()
    for reward_group in reward_groups:
        group_subjects = mouse_info_df[mouse_info_df['reward_group'] == reward_group]['mouse_id'].unique()
        print(
            f"Reward group {reward_group} has {len(mouse_info_df[mouse_info_df['reward_group'] == reward_group])} mice: {group_subjects}.")

    # Select mice to do based on available NWB files
    subject_ids = [mouse for mouse in subject_ids if any(mouse in name for name in all_mwb_mice)]
    subject_ids = [s for s in subject_ids if
                   int(s[2:]) in [50, 51, 52, 54, 56, 58, 59, 68, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 85,
                                  86, 87, 92, 93, 94, 95, 96, 97, 100, 101, 102, 103, 104, 105, 106, 107]]
    subject_ids.extend(['AB{}'.format(i) for i in range(116, 132)])
    #subject_ids = ['AB092', 'AB093', 'AB095', 'AB086', 'AB087']

    # Get list of NWB files for each mouse
    nwb_list = [os.path.join(root_path, name) for name in all_nwb_names]
    nwb_list = [nwb_file for nwb_file in nwb_list if any(mouse in nwb_file for mouse in subject_ids)]

    # Load neural dataframes
    # Keep NWB with ephys data, assuming one recording per mouse
    ephys_nwb_list = []
    for nwb in nwb_list:
        behaviour_typer, day = nwb_reader.get_bhv_type_and_training_day_index(nwb)
        with NWBHDF5IO(nwb, 'r') as io:
            nwbfile = io.read()
            if behaviour_typer == 'whisker' and day == 0 and nwbfile.units is not None:
                ephys_nwb_list.append(nwb)

    print(f"Found {len(ephys_nwb_list)} NWB files with ephys data.")
    print('Available NWB files:', ephys_nwb_list)

    # Combine data from all NWBs
    unit_table_list = []
    for nwb in ephys_nwb_list:
        mouse_id = nwb_reader.get_mouse_id(nwb)
        session_id = nwb_reader.get_session_id(nwb)

        unit_table = nwb_reader.get_unit_table(nwb)
        unit_table['mouse_id'] = mouse_id
        unit_table_list.append(unit_table)

    unit_table = pd.concat(unit_table_list)
    unit_table = unit_table.reset_index(drop=True)

    # Plot
    output_folder = os.path.join(output_path, 'unit_count_map')

    for cmap in ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']:
        for orient in ['horizontal', 'frontal', 'sagittal']:
            params = {
                'ccf_level': 'ccf_parent_acronym',
                'cmap': cmap,
                'orientation': orient,
                'vmin': 0,
                'vmax': 1500,
            }
            plot_unit_count_map(unit_table, output_folder, params)
            