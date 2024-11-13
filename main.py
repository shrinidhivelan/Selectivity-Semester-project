from generate import *



def main():
    # https://drive.google.com/drive/folders/1FngbNmOJ7lc59VY_g36eqXh7GBsavCzI?usp=sharing

    #### Replace this with the path of the project :
    path = '/Users/shrinidhivelan/Desktop/Selectivity-Semester-project/'
    mouse_names = ['AB087_20231017_141901', 'AB077_20230531_143839','AB080_20230622_152205']
    print("Starting the analysis...")
    
    #### (1) Generate 1 mouse name and create the selectivity parquet file: ####
    df = generate_mice_data(mouse_names, path)

    

    #### (2) Create the ROC Analysis parquet file ####
    AUC_generate(df, mouse_names, save_files = True, visualize = False, pre_vs_post_visualization = False)
    #AUC_plots(mouse_names)




if __name__ == "__main__":
    main()