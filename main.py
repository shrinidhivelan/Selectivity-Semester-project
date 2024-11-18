from generate import *
from time import sleep




def main():

    folder_path = '/Volumes/LaCie/EPFL/Master sem3/Semester Project Lsens/Mice_data'
    #mouse_names = ['AB087_20231017_141901', 'AB077_20230531_143839','AB080_20230622_152205']
    print("Starting the analysis...")
    
    #### (1) Generate 1 mouse name and create the selectivity parquet file: ####
    #mouse_names = generate_mice_data(folder_path)

    ### Un comment the following if you do not want to regenerate the whole data : 
    #mice_names = ['AB116_20240724_102941','AB117_20240723_125437','AB119_20240731_102619']
    mouse_names = ['AB124_20240815_111810','AB125_20240817_123403','AB126_20240822_114405','AB130_20240902_123634','AB129_20240828_112850','AB128_20240829_112813','AB127_20240821_103757']

 
    
    #### (2) Create the ROC Analysis parquet file ####
    AUC_generate(mouse_names)#, save_files = False, visualize = False, pre_vs_post_visualization = False)





if __name__ == "__main__":
    main()