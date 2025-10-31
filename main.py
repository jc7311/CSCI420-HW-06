# Ethan Chang
# HW 06 Agglomeration
# CSCI 420

import csv
import pandas as pd
import numpy as np
import heapq

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
            'clusters': [i], # cluster itself, can represent guest id
            'size': 1

        })

    cluster_size = len(clusters)

    min_dist = 0.0
    cluster_i = 0
    cluster_j = 0

    distance = []
    smallest_sizes = []
    merge_history = []


    distance_heap = []

   
        # 1. Calculate all distances between clusters
    distances = []

    for i in range(len(clusters)):

        for j in range(i+1, len(clusters)):

            dist = np.linalg.norm(clusters[i]['center'] - clusters[j]['center'])

            heapq.heappush(distance_heap, (dist, i, j))

        # 2. Find closest pair
        
    while len(clusters) > 1:

        if len(clusters) % 100 == 0:  # Print progress every 100 merges

            print(f"Clusters remaining: {len(clusters)}")
        
        while True:

            min_dist, i, j = heapq.heappop(distance_heap)

            # Check if these clusters still exist and indices are valid

            if (i < len(clusters) and j < len(clusters) and 
                
                clusters[i] is not None and clusters[j] is not None):

                break

        # 3. Merge clusters i and j
        size_i = clusters[i]['size']
        size_j = clusters[j]['size']
        total_size = size_i + size_j

        smallest_size = min(size_i, size_j)

        smallest_sizes.append(smallest_size)

        merge_history.append((i,j, smallest_size))
        
        # Calculate new center (weighted average)
        new_center = (clusters[i]['center'] * size_i + clusters[j]['center'] * size_j) / total_size
        
        # Combine members
        new_members = clusters[i]['clusters'] + clusters[j]['clusters']
        
        # Create new merged cluster
        new_cluster = {
            'center': new_center,
            'clusters': new_members,
            'size': total_size
        }

        clusters.pop(max(i, j))
        clusters.pop(min(i, j))
        clusters.append(new_cluster)
            
        # 4. Remove old clusters and add new one
        # Remove higher index first to avoid shifting issues
        new_cluster_index = len(clusters) - 1  # The new cluster is at the end

        for k in range(new_cluster_index):

            dist = np.linalg.norm(clusters[k]['center'] - clusters[new_cluster_index]['center'])
            heapq.heappush(distance_heap, (dist, k, new_cluster_index))
    

    print("Size of smallest cluster in last 20 merges:", smallest_sizes[-20:])
    print("Last 10 smallest clusters merged:", smallest_sizes[-10:])
    
    #print(clusters)

    #print(clusters)

    #print(abs_corr)


    #print(full_correlation)



    #print(strongly_correlated)
    
if __name__ == "__main__":

    main()


