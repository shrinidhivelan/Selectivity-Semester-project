from generate import *




def main():

    #### Replace the following by wherever your base data is:
    folder_path = '/Volumes/LaCie/EPFL/Master sem3/Semester Project Lsens/Mice_data'

    print("Starting the analysis...")
    
    #### (1) Generate 1 mouse name and create the selectivity parquet file: ####
    mouse_names = generate_mice_data(folder_path)

    ### Un comment the following if you do not want to regenerate the whole data : 
    # mouse_names = ['AB124_20240815_111810','AB125_20240817_123403','AB126_20240822_114405','AB130_20240902_123634','AB129_20240828_112850','AB128_20240829_112813','AB127_20240821_103757']

 
    
    #### (2) Create the ROC Analysis parquet file ####
    AUC_generate(mouse_names)



if __name__ == "__main__":
    main()