from generate import *


def put_together(main_folder = ''):
    mouse_names = ['AB124_20240815_111810','AB125_20240817_123403','AB126_20240822_114405','AB130_20240902_123634','AB129_20240828_112850','AB128_20240829_112813','AB127_20240821_103757',
               'AB123_20240806_110231', 'AB122_20240804_134554', 'AB119_20240731_102619', 'AB117_20240723_125437', 'AB116_20240724_102941']

    main_folder = '/Volumes/LaCie/EPFL/Mastersem3/Semester Project Lsens/Data/'

    #AB116_20240724_102941_AUC_Selectivity2.parquet

    mice_data = []

    for mouse in mouse_names:
        df = pd.read_parquet(main_folder+mouse+"/"+mouse+"_AUC_Selectivity2.parquet")
        mice_data.append(df)

    df_combined = pd.concat(mice_data).reset_index(drop=True) 




def main():

    #### Replace the following by wherever your base data is:
    folder_path = '/Volumes/LaCie/EPFL/Master sem3/Semester Project Lsens/Mice_data/context'
    save_path = '/Volumes/LaCie/EPFL/Master sem3/Semester Project Lsens/Data'

    print("Starting the analysis...")
    
    #### (1) Generate 1 mouse name and create the selectivity parquet file: ####
    #mouse_names = generate_mice_data(folder_path, save_path)

    ### Un comment the following if you do not want to regenerate the whole data : 
    #mouse_names = ['AB124_20240815_111810','AB125_20240817_123403','AB126_20240822_114405','AB130_20240902_123634','AB129_20240828_112850','AB128_20240829_112813','AB127_20240821_103757']
    #mouse_names = ['AB126_20240822_114405', 'AB127_20240821_103757', 'AB128_20240829_112813', 'AB129_20240828_112850', 'AB130_20240902_123634']
 
    
    #### (2) Create the ROC Analysis parquet file ####
    #AUC_generate(mouse_names, save_path)

    put_together()



if __name__ == "__main__":
    main()