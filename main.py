from generate import *
from plot import *
import dask.dataframe as dd
from helpers import *



def main():

    #### Replace the following by wherever your base data is:

    folder_path_context = '/Volumes/LaCie/EPFL/Mastersem3/Semester Project Lsens/Mice_data/context'
    folder_path_no_context = '/Volumes/LaCie/EPFL/Mastersem3/Semester Project Lsens/Mice_data/nocontext'
    have_context = []
    save_path = '/Volumes/LaCie/EPFL/Mastersem3/Semester Project Lsens/Data'

    print("Starting the analysis...")
    
    #### (1) Generate 1 mouse name and create the selectivity parquet file: ####

    for has_context in have_context:
        if has_context == True:
            print('Starting the process for data with context information...')
            #mouse_names = generate_mice_data(folder_path_context, save_path, context=True)
            #AUC_generate(mouse_names, save_path, has_context=True)
            #put_together(save_path, mouse_names, True)
            mouse_names = ['AB116_20240724_102941']
            AUC_generate(mouse_names, save_path, has_context=True)


        else: 
            print('Starting the process for data with no context information...')
            mouse_names = generate_mice_data(folder_path_no_context, save_path, context=False)
            AUC_generate(mouse_names, save_path, has_context=False)
            #put_together(save_path, mouse_names, False)
    

    mouse_names_context = ['AB120_20240811_143102','AB121_20240813_125401','AB124_20240815_111810','AB125_20240817_123403','AB126_20240822_114405','AB130_20240902_123634','AB129_20240828_112850','AB128_20240829_112813','AB127_20240821_103757',
               'AB123_20240806_110231', 'AB122_20240804_134554', 'AB119_20240731_102619', 'AB117_20240723_125437', 'AB116_20240724_102941']
    mouse_names_no_context = ['AB077_20230531_143839','AB080_20230622_152205', 'AB082_20230630_101353',
                   'AB085_20231005_152636', 'AB086_20231015_141742', 'AB087_20231017_141901',
                   'AB092_20231205_140109', 'AB093_20231207_111207', 'AB095_20231212_141728',
                   'AB102_20240309_114107', 'AB104_20240313_145433', 'AB107_20240318_121423']
    
    #put_together(save_path, mouse_names_context, True)
    #put_together(save_path, mouse_names_no_context, False)

    #combine_files(save_path)



    """ 
    ## The 12 mice data that don't have the column context
    mouse_names = ['AB077_20230531_143839','AB080_20230622_152205', 'AB082_20230630_101353',
                   'AB085_20231005_152636', 'AB086_20231015_141742', 'AB087_20231017_141901',
                   'AB092_20231205_140109', 'AB093_20231207_111207', 'AB095_20231212_141728',
                   'AB102_20240309_114107', 'AB104_20240313_145433', 'AB107_20240318_121423']
    ### Un comment the following if you do not want to regenerate the whole data : 
    mouse_names = ['AB124_20240815_111810','AB125_20240817_123403','AB126_20240822_114405','AB130_20240902_123634','AB129_20240828_112850','AB128_20240829_112813','AB127_20240821_103757']
    #mouse_names = ['AB126_20240822_114405', 'AB127_20240821_103757', 'AB128_20240829_112813', 'AB129_20240828_112850', 'AB130_20240902_123634']
    ## All the 14 mice data : 
    mouse_names = ['AB120_20240811_143102','AB121_20240813_125401','AB124_20240815_111810','AB125_20240817_123403','AB126_20240822_114405','AB130_20240902_123634','AB129_20240828_112850','AB128_20240829_112813','AB127_20240821_103757',
               'AB123_20240806_110231', 'AB122_20240804_134554', 'AB119_20240731_102619', 'AB117_20240723_125437', 'AB116_20240724_102941']
    mouse_names_single = []
    #### (1a) Generate Raster plots :
    #for mouse in mouse_names:
    for folder_path in [folder_path_context, folder_path_no_context]:
        if folder_path == folder_path_context:
            mouse_names = ['AB120_20240811_143102','AB121_20240813_125401','AB124_20240815_111810','AB125_20240817_123403','AB126_20240822_114405','AB130_20240902_123634','AB129_20240828_112850','AB128_20240829_112813','AB127_20240821_103757',
               'AB123_20240806_110231', 'AB122_20240804_134554', 'AB119_20240731_102619', 'AB117_20240723_125437', 'AB116_20240724_102941']
        else:
            mouse_names = ['AB077_20230531_143839','AB080_20230622_152205', 'AB082_20230630_101353',
                   'AB085_20231005_152636', 'AB086_20231015_141742', 'AB087_20231017_141901',
                   'AB092_20231205_140109', 'AB093_20231207_111207', 'AB095_20231212_141728',
                   'AB102_20240309_114107', 'AB104_20240313_145433', 'AB107_20240318_121423']
        
        for mouse in mouse_names:
            main_data_per_mouse = os.path.join(folder_path, mouse+".nwb")
            nwbfile = NWBHDF5IO(main_data_per_mouse, mode='r').read()
            Raster_total_context(nwbfile, start=0.5, stop=1, mouse_name=mouse, main_folder = '/Volumes/LaCie/EPFL/Mastersem3/Semester Project Lsens')
    """

     
    #### (2) Create the ROC Analysis parquet file ####

    #### (2a) Generate AUC plots :
    for folder_path in [folder_path_context, folder_path_no_context]:
        if folder_path == folder_path_context:
            mouse_names = ['AB120_20240811_143102','AB121_20240813_125401','AB124_20240815_111810','AB125_20240817_123403','AB126_20240822_114405','AB130_20240902_123634','AB129_20240828_112850','AB128_20240829_112813','AB127_20240821_103757',
               'AB123_20240806_110231', 'AB122_20240804_134554', 'AB119_20240731_102619', 'AB117_20240723_125437', 'AB116_20240724_102941']
        else:
            mouse_names = ['AB077_20230531_143839','AB080_20230622_152205', 'AB082_20230630_101353',
                   'AB085_20231005_152636', 'AB086_20231015_141742', 'AB087_20231017_141901',
                   'AB092_20231205_140109', 'AB093_20231207_111207', 'AB095_20231212_141728',
                   'AB102_20240309_114107', 'AB104_20240313_145433', 'AB107_20240318_121423']
        
        for mouse in mouse_names:
            process_and_save_roc(mouse, save_path)
    
    
            
    #convert_to_csv(mouse_names)

    #put_together(save_path, mouse_names)



if __name__ == "__main__":
    main()