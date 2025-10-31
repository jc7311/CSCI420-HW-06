# Ethan Chang
# HW 06 Agglomeration
# CSCI 420

import csv
import pandas as pd
import numpy as np

def euclidean_distance(center1, center2):

    return np.sqrt(np.sum((center1- center2) ** 2))


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

    #print(features)

    corr_matrix = features.corr().round(2)
    
    corr_matrix.to_csv('correlated_matrix.csv')

    """
    
    Part B
    
    """

    # Start of agglomerative clustering, assign each record to its own cluster protoype
    # Suppose there are 1000+ records meaning start with 1000+ clusters and 1000+ protoypes of those clusters

    clusters = []
    
    n_size = features.shape[0]

    #print(features.values)

    data_points = features.values
    #print(len(data_points))

    #print(data_points)

    for i in range(len(data_points)):

        clusters.append({

            'center': data_points[i], # prototype - book purchase counts for each row
            'clusters': [i] # cluster itself, can represent guest id

        })

    cluster_size = len(clusters)

    min_dist = 0.0
    cluster_i = 0
    cluster_j = 0

    for i in range(cluster_size):
        for j in range(i+1, cluster_size):

            dist = euclidean_distance(clusters[i]['center'], clusters[j]['center'])

            if dist < min_dist:

                min_dist = dist
                cluster_i = i
                cluster_j = j

    print(dist)

        

    

    #print(clusters)

    #print(clusters)

    #print(abs_corr)


    #print(full_correlation)



    #print(strongly_correlated)
    
if __name__ == "__main__":

    main()


