from generate import *



#from generate import *
from time import sleep




def main():
    mouse_names = ['AB087_20231017_141901', 'AB077_20230531_143839','AB080_20230622_152205']
    print("Starting the analysis...")
    
    #### (1) Generate 1 mouse name and create the selectivity parquet file: ####
    df, mice_data = generate_mice_data(mouse_names)
    
    #### (2) Create the ROC Analysis parquet file ####
    AUC_generate(mouse_names, save_files = False, visualize = False, pre_vs_post_visualization = False)
    #AUC_plots(mouse_names)




if __name__ == "__main__":
    main()