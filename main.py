from generate import *
from plot import *
import dask.dataframe as dd
from helpers import *






def main():

    #### Replace the following by wherever your base data is:
    folder_path = '/Volumes/LaCie/EPFL/Master sem3/Semester Project Lsens/Mice_data/context'
    save_path = '/Volumes/LaCie/EPFL/Master sem3/Semester Project Lsens/Data'

    print("Starting the analysis...")
    
    #### (1) Generate 1 mouse name and create the selectivity parquet file: ####
    #mouse_names = generate_mice_data(folder_path, save_path)

    ### Un comment the following if you do not want to regenerate the whole data : 
    mouse_names = ['AB124_20240815_111810','AB125_20240817_123403','AB126_20240822_114405','AB130_20240902_123634','AB129_20240828_112850','AB128_20240829_112813','AB127_20240821_103757']
    #mouse_names = ['AB126_20240822_114405', 'AB127_20240821_103757', 'AB128_20240829_112813', 'AB129_20240828_112850', 'AB130_20240902_123634']
    mouse_names = ['AB124_20240815_111810','AB125_20240817_123403','AB126_20240822_114405','AB130_20240902_123634','AB129_20240828_112850','AB128_20240829_112813','AB127_20240821_103757',
               'AB123_20240806_110231', 'AB122_20240804_134554', 'AB119_20240731_102619', 'AB117_20240723_125437', 'AB116_20240724_102941']

    """ 
    #### (1a) Generate Raster plots :
    for mouse in mouse_names:
        main_data_per_mouse = os.path.join(folder_path, mouse+".nwb")
        nwbfile = NWBHDF5IO(main_data_per_mouse, mode='r').read()
        Raster_total_context(nwbfile, start=0.5, stop=1, mouse_name=mouse, main_folder = '/Volumes/LaCie/EPFL/Mastersem3/Semester Project Lsens')

 
    #### (2) Create the ROC Analysis parquet file ####
    AUC_generate(mouse_names, save_path)

    #### (2a) Generate AUC plots :
    for mouse_name in mouse_names:
        process_and_save_roc(mouse_name, save_path)
    
    """
    #convert_to_csv(mouse_names)

    #put_together(save_path, mouse_names)



if __name__ == "__main__":
    main()