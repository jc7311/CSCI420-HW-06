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

    features = df.iloc[:, 1:] # to drop guest id

    corr_matrix = features.corr().round(2)
    
    corr_matrix.to_csv('correlated_matrix.csv')

    """
    
    Part B
    
    """

    # Start of agglomerative clustering, assign each record to its own cluster protoype
    # Suppose there are 1000+ records meaning start with 1000+ clusters and 1000+ protoypes of those clusters

    clusters = []

    data_points = features.values

    cluster_centers = {}
    clusters = {}
    active_ids = set()

    smallest_sizes = []  # Track smallest cluster size in each merge

    for i in range(len(data_points)):

        cluster_id = i
        cluster_centers[cluster_id] = data_points[i]
        clusters[cluster_id] = [i]
        active_ids.add(cluster_id)

    # Build initial heap
    distance_heap = []

    for cluster_i in active_ids:

        for cluster_j in active_ids:

            if cluster_j > cluster_i:  # Avoid duplicates

                dist = euclidean_distance(cluster_centers[cluster_i], cluster_centers[cluster_j])

                heapq.heappush(distance_heap, (dist, cluster_i, cluster_j))

    # Merge loop
    while len(active_ids) > 1:

        # Get valid pair
        while True:

            # (distance_heap = [], (distance, i, j))
            min_dist, i, j = heapq.heappop(distance_heap)

            if i in active_ids and j in active_ids: # Stops the while loop once the pair is active to merge

                break

        
        # Merge j into i
        clusters[i].extend(clusters[j]) # Adds j's elements into i's cluster

        size_i = len(clusters[i]) - len(clusters[j]) # original i's cluster size before add j's elements in previous line

        size_j = len(clusters[j]) # original j's cluster size
        
        smaller_size = min(size_i, size_j)
        smallest_sizes.append(smaller_size)

        cluster_centers[i] = (cluster_centers[i] * size_i + cluster_centers[j] * size_j) / (size_i + size_j)
        
        # Update distances for the merged cluster
        for other_id in active_ids:

            if other_id != i and other_id != j:

                new_dist = euclidean_distance(cluster_centers[i], cluster_centers[other_id])

                heapq.heappush(distance_heap, (new_dist, i, other_id))
        
        # Clean up
        del clusters[j]
        del cluster_centers[j]
        active_ids.remove(j)

    print("Final clusters:", len(clusters))
    print("Size of smallest cluster in last 20 merges:", smallest_sizes[-20:])
    print("Last 10 smallest clusters merged:", smallest_sizes[-10:])

    # Look for when small clusters stop merging with other small clusters
    for i in range(len(smallest_sizes)-10, len(smallest_sizes)):
        print(f"Merge {i}: smallest cluster size = {smallest_sizes[i]}")
        
    for cluster_id, member_list in clusters.items():
        
        cluster_size = len(member_list)
        cluster_center = cluster_centers[cluster_id]
        print(f"Cluster {cluster_id}: {cluster_size} members")
        print(f"  Shopping pattern: {cluster_center}")
            
    #print(clusters)

    #print(clusters)

    #print(abs_corr)


    #print(full_correlation)



    #print(strongly_correlated)
    
if __name__ == "__main__":

    main()


