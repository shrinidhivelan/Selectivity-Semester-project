import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score 
import os

def lick_times(array, interval = 1):
    #arr = array['lick_time'].values
    arr = array[~np.isnan(array)]
    # Create new array 
    arr_new = []

    # Check the first element
    if len(arr) > 1 and (arr[1] - arr[0] > interval):
        arr_new.append(arr[0])  

    # Check all other elements 
    for i in range(1, len(arr) - 1):
        if (arr[i + 1] - arr[i] > interval) and (arr[i] - arr[i - 1] > interval):
            arr_new.append(arr[i])

    # Check the final element of the table
    if len(arr) > 1 and (arr[-1] - arr[-2] > interval):
        arr_new.append(arr[-1])  
    
    return arr_new