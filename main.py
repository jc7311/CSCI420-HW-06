# Ethan Chang
# HW 06 Agglomeration
# CSCI 420

import csv
import pandas as pd
import numpy as np

def main():

    """
    
    Part A.
    
    """
    df = pd.read_csv('HW_CLUSTERING_SHOPPING_CART_v2245a.csv')

    #print(df)

    #print(list(df.columns))
    #print(df.shape)
    #print(df.head(3))

    features = df.iloc[:, 1:] # to drop guest id

    #feature_names = df.columns[1:].tolist()
    #print(feature_names)

    print(features)

    corr_matrix = features.corr().round(2)


    
    corr_matrix.to_csv('correlated_matrix.csv')

    abs_corr = corr_matrix.abs()
    np.fill_diagonal(abs_corr.values, 0)




    #print(abs_corr)


    #print(full_correlation)



    #print(strongly_correlated)
    

if __name__ == "__main__":

    main()