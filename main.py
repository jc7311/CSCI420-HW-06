"""
CSCI 420 - HW 06: Agglomeration
Authors: Jacky Chan (jc7311), Ethan Chang (elc6696)

"""

import pandas as pd # Data frames
import numpy as np # Number calculations
import heapq # data structure for merging
import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import dendrogram, linkage

def main():

    """
    Part A. Cross Correlation Analysis

    Loads data sets and separate features from guest IDS, calculates relationships between book categories
    then saves into correlated matrix comma separted value file.
    """

    df = pd.read_csv('HW_CLUSTERING_SHOPPING_CART_v2245a.csv')

    features = df.iloc[:, 1:] # to drop guest id

    guest_ids = df['ID'].values

    attribute_names = df.columns[1:].tolist() # Exclude guest column so it's not part of attribute names

    corr_matrix = features.corr().round(2)
    
    corr_matrix.to_csv('correlated_matrix.csv') # Writes intno correlated matrix csv file

    print("\nCross-correlation matrix saved as 'correlated_matrix.csv'\n")

    """
    
    Part B. Agglomerative Clustering

    Sets up initial state that each customer represents an individual cluster themselves. 
    Utilizes euclidean calulations to determine distances of two clusters and center. 

    """

    print("\nPART B: AGGLOMERATIVE CLUSTERING\n")

    # Convert panda DataFram to numpy array
    data_matrix = features.values 

    # num_records: number of customers
    # num_attributes: number of book categories
    num_records, num_attributes = data_matrix.shape 

    # initialize clusters and centers {cluster_id: [list_of_customer_indices]}
    clusters = {i: [i] for i in range(num_records)}

    # Store average shopping pattern for each cluster
    cluster_centers = {i: data_matrix[i].copy() for i in range(num_records)}

    # tracks which cluster ids are active
    active_cluster_ids = set(range(num_records))

    #Merge History for analysis
    merge_history = []

    """
    Calculates straight-line distance between two clusters
    """
    def compute_euclidean_distance(center1, center2):
        return np.sqrt(np.sum((center1 - center2) ** 2))

    """
    Compute average shopping pattern for cluster
    """
    def compute_cluster_center(cluster_member_indices):
        member_data = data_matrix[cluster_member_indices]
        return np.mean(member_data, axis=0)

    print(f"Starting agglomerative clustering with {num_records} records")
    print("Building initial distance heap...")

    min_heap = [] # priority queue to find closest clusters
    distance_cache = {} # store computed distances and finds unique ones

    """
    Create unique, order-independent key for cluster pair (i, j)
    """
    def pair_key(i, j):
        return (min(i, j), max(i, j))

    # Make active cluster to list for iteration
    active_list = list(range(num_records))

    # Compute all pairwise distances between clusters
    for idx_i in range(len(active_list)):

        i = active_list[idx_i] # Get cluster ID
        center_i = cluster_centers[i] # Get cluster centroid

        # Compute distances j > i to avoid duplicates
        for idx_j in range(idx_i + 1, len(active_list)):

            j = active_list[idx_j] # Get cluster 2 ID

            center_j = cluster_centers[j] # Get cluster 2 centroid

            dist = compute_euclidean_distance(center_i, center_j) # Compute euclidean distance between two clusters

            distance_cache[pair_key(i, j)] = dist # Store distance with unique key

            heapq.heappush(min_heap, (dist, i, j)) # Push distance to heap for efficient value retrieval

    print("Initial heap built. Starting clustering...\n")

    iteration_count = 0
    total_merges = num_records - 1

    # Stops when all clusters are merged into one
    while len(active_cluster_ids) > 1:

        iteration_count += 1
        
        # Progress reporting for long-run computing
        if iteration_count % 100 == 0:
            print(f"Progress: {iteration_count}/{total_merges} merges completed, {len(active_cluster_ids)} clusters remaining")

        # Find valid pair to merge
        while True:

            # Reminder: heapq targets smallest-distance value since it's a priority queue
            min_dist, i, j = heapq.heappop(min_heap) # Pops closest cluster pair

            if i in active_cluster_ids and j in active_cluster_ids:
                break # once pair is found, the loop ends
            
        # Assign cluster sizes before merging    
        size_i, size_j = len(clusters[i]), len(clusters[j])

        # Find smallest cluster size
        smaller_cluster_size = min(size_i, size_j)

        # Document merge operations
        merge_history.append({
            'iteration': iteration_count, # Merge number
            'cluster1': i, # id of first cluster
            'cluster2': j, # id of second cluster
            'size1': size_i, # size of first cluster
            'size2': size_j, # size of second cluster
            'smaller_size': smaller_cluster_size, # pull smallest cluster size
            'distance': min_dist, # distance of merge
            'resulting_size': size_i + size_j # total members after merge
        })

        # merge j into i
        clusters[i].extend(clusters[j])

        # recalculate centroid for merged cluster
        cluster_centers[i] = compute_cluster_center(clusters[i])
        new_center = cluster_centers[i]

        # Start updating distance from merged cluster to all active clusters
        for other in active_cluster_ids:

            if other != i and other != j:
                
                # remove old distances that were involved in merging process
                distance_cache.pop(pair_key(j, other), None)
                distance_cache.pop(pair_key(i, other), None)

                # compute new distance from merged cluster to a another cluster
                new_dist = compute_euclidean_distance(new_center, cluster_centers[other])

                # Distance and queue of updated distance
                distance_cache[pair_key(i, other)] = new_dist
                heapq.heappush(min_heap, (new_dist, i, other))

        # Clean up cluster 2
        clusters.pop(j, None) # remove from clusters list
        cluster_centers.pop(j, None) # remove from centroids list
        active_cluster_ids.remove(j) # mark as inactive

    print(f"\nClustering complete! Total merges: {len(merge_history)}")

    # display last 20 merges
    print("\nLAST 20 MERGES - Smallest Cluster Size Tracking\n")
    for m in merge_history[-20:]:
        print(f"Merge {m['iteration']:4d}: Cluster {m['cluster1']:4d} (size {m['size1']:4d}) + "
              f"Cluster {m['cluster2']:4d} (size {m['size2']:4d}) -> "
              f"Smaller cluster size: {m['smaller_size']:4d}, Distance: {m['distance']:.2f}")

    print("\nLAST 10 SMALLEST CLUSTER SIZES IN MERGES:\n")
    last_10_smallest = [m['smaller_size'] for m in merge_history[-10:]]
    print("Sizes:", last_10_smallest)

    # build linkage matrix
    # Format: row = [cluster1, cluster2, distance, count]
    linkage_matrix = np.zeros((len(merge_history), 4))
    cluster_id_map = {i: i for i in range(num_records)}

    next_cluster_id = num_records

    # Convert merge history into linkage format
    for merge_idx, merge_info in enumerate(merge_history):
        c1, c2 = merge_info['cluster1'], merge_info['cluster2']

        # MApped IDS
        linkage_matrix[merge_idx, 0] = cluster_id_map[c1]
        linkage_matrix[merge_idx, 1] = cluster_id_map[c2]
        linkage_matrix[merge_idx, 2] = merge_info['distance']
        linkage_matrix[merge_idx, 3] = merge_info['resulting_size']

        # Both clusters go to new merged ID
        cluster_id_map[c1] = next_cluster_id
        cluster_id_map[c2] = next_cluster_id

        next_cluster_id += 1

    # plot dendrogram for data visualization
    print("\nDENDROGRAM - Last 20 Clusters\n")
    plt.figure(figsize=(14, 8))

    dendrogram(
        linkage_matrix,
        truncate_mode='lastp', # Show last p clusters. p = 20
        p=20,
        show_leaf_counts=True,
        leaf_font_size=10
    )
    plt.title('Dendrogram - Last 20 Clusters\n(Agglomerative Clustering with Centroid Linkage)', fontsize=14)
    plt.xlabel('Cluster Index (or Number of Records in Cluster)', fontsize=12)
    plt.ylabel('Euclidean Distance Between Cluster Centers', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('dendrogram_last_20_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nDendrogram saved as 'dendrogram_last_20_clusters.png'")

    # analyze final clusters
    num_final_clusters = 4

    print(f"\nANALYZING FINAL {num_final_clusters} CLUSTERS\n")

    # Create an array to store which cluster each customer belongs to
    record_to_cluster_assignment = np.zeros(num_records, dtype=int)

    # Temporary cluster structure to track clusters
    clusters_at_k = {i: [i] for i in range(num_records)}

    # Track what clusters are still active
    active_ids_at_k = set(range(num_records))

    # Loops the merge history
    for merge_info in merge_history[:-num_final_clusters + 1]:

        # Extract two clusters that were merged
        ci, cj = merge_info['cluster1'], merge_info['cluster2']

        if ci in active_ids_at_k and cj in active_ids_at_k:


            clusters_at_k[ci].extend(clusters_at_k[cj])

            del clusters_at_k[cj]
            active_ids_at_k.remove(cj)

    # List of final cluster IDS after stopping at a specific number
    final_cluster_list = sorted(active_ids_at_k)

    # Sequence cluster ids for better analysis
    for new_cid, orig_cid in enumerate(final_cluster_list):

        # Each customer in original cluster gets assign to new sequential cluster ID
        for record_index in clusters_at_k[orig_cid]:

            record_to_cluster_assignment[record_index] = new_cid

    # List to store cluster statistics
    cluster_sizes, cluster_prototypes = [], []

    # Calculate size and average for each final cluster
    for cid in range(num_final_clusters):
        
        # Find all customers assigned to this cluster
        members = np.where(record_to_cluster_assignment == cid)[0]

        # Count how many customers are in this cluster
        cluster_sizes.append(len(members))

        # Calculate average shopping pattern
        cluster_prototypes.append(np.mean(data_matrix[members], axis=0))

    # Reports cluster counts from smallest to largest order
    print("\nCLUSTER SIZES (from smallest to largest):\n")
    
    ####################################
    # QUESTION 5
    # Pulls indices that sort cluster sizes array in ascending order (smallest to largest)
    ####################################
    size_order = np.argsort(cluster_sizes)

    for cluster_id in size_order:
        print(f"Cluster {cluster_id}: {cluster_sizes[cluster_id]} members")

    ####################################
    # QUESTION 6
    # Reports average protoype of each cluster
    ####################################
    print("\nCLUSTER PROTOTYPES (Average Attribute Values):\n")

    for cluster_id in range(num_final_clusters):

        print(f"\nCluster {cluster_id} (Size: {cluster_sizes[cluster_id]}):")

        # Create a dataframe to disply cluster's average purchase pattern
        proto_df = pd.DataFrame({

            'Attribute': attribute_names, # Rerpesents category names
            'Average Value': np.round(cluster_prototypes[cluster_id], 2)

        })

        print(proto_df.to_string(index=False))

    # Create a datafram mapping each guest to assigned cluster
    results_df = pd.DataFrame({

        'Guest_ID': guest_ids, # Guest idenfitifcation numbers
        'Cluster_Assignment': record_to_cluster_assignment

    })

    results_df.to_csv('cluster_assignments.csv', index=False)
    print("\nCluster assignments saved to 'cluster_assignments.csv'")

    print(f"\nTotal records processed: {num_records}")
    print(f"Number of attributes: {num_attributes}")
    print(f"Final number of clusters: {num_final_clusters}")

if __name__ == "__main__":
    main()
