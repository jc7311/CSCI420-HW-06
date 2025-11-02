"""
CSCI 420 - HW 06: Agglomeration
Authors: Jacky Chan (jc7311), Ethan Chang (elc6696)

"""

import csv
import pandas as pd
import numpy as np
import heapq
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def main():

    """
    
    Part A.
    
    """
    df = pd.read_csv('HW_CLUSTERING_SHOPPING_CART_v2245a.csv')

    features = df.iloc[:, 1:] # to drop guest id

    guest_ids = df['ID'].values

    attribute_names = df.columns[1:].tolist()

    corr_matrix = features.corr().round(2)
    
    corr_matrix.to_csv('correlated_matrix.csv')

    print("\nCross-correlation matrix saved as 'correlated_matrix.csv'\n")

    """
    
    Part B

    """

    print("\nPART B: AGGLOMERATIVE CLUSTERING\n")

    data_matrix = features.values
    num_records, num_attributes = data_matrix.shape

    # initialize clusters and centers
    clusters = {i: [i] for i in range(num_records)}
    cluster_centers = {i: data_matrix[i].copy() for i in range(num_records)}
    active_cluster_ids = set(range(num_records))
    merge_history = []

    def compute_euclidean_distance(center1, center2):
        return np.sqrt(np.sum((center1 - center2) ** 2))

    def compute_cluster_center(cluster_member_indices):
        member_data = data_matrix[cluster_member_indices]
        return np.mean(member_data, axis=0)

    print(f"Starting agglomerative clustering with {num_records} records")
    print("Building initial distance heap...")

    min_heap = []
    distance_cache = {}

    def pair_key(i, j):
        return (min(i, j), max(i, j))

    active_list = list(range(num_records))
    for idx_i in range(len(active_list)):
        i = active_list[idx_i]
        center_i = cluster_centers[i]
        for idx_j in range(idx_i + 1, len(active_list)):
            j = active_list[idx_j]
            center_j = cluster_centers[j]
            dist = compute_euclidean_distance(center_i, center_j)
            distance_cache[pair_key(i, j)] = dist
            heapq.heappush(min_heap, (dist, i, j))

    print("Initial heap built. Starting clustering...\n")

    iteration_count = 0
    total_merges = num_records - 1

    while len(active_cluster_ids) > 1:
        iteration_count += 1
        if iteration_count % 100 == 0:
            print(f"Progress: {iteration_count}/{total_merges} merges completed, {len(active_cluster_ids)} clusters remaining")

        while True:
            min_dist, i, j = heapq.heappop(min_heap)
            if i in active_cluster_ids and j in active_cluster_ids:
                break

        size_i, size_j = len(clusters[i]), len(clusters[j])
        smaller_cluster_size = min(size_i, size_j)
        merge_history.append({
            'iteration': iteration_count,
            'cluster1': i,
            'cluster2': j,
            'size1': size_i,
            'size2': size_j,
            'smaller_size': smaller_cluster_size,
            'distance': min_dist,
            'resulting_size': size_i + size_j
        })

        # merge j into i
        clusters[i].extend(clusters[j])
        cluster_centers[i] = compute_cluster_center(clusters[i])
        new_center = cluster_centers[i]

        for other in active_cluster_ids:
            if other != i and other != j:
                distance_cache.pop(pair_key(j, other), None)
                distance_cache.pop(pair_key(i, other), None)
                new_dist = compute_euclidean_distance(new_center, cluster_centers[other])
                distance_cache[pair_key(i, other)] = new_dist
                heapq.heappush(min_heap, (new_dist, i, other))

        clusters.pop(j, None)
        cluster_centers.pop(j, None)
        active_cluster_ids.remove(j)

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
    linkage_matrix = np.zeros((len(merge_history), 4))
    cluster_id_map = {i: i for i in range(num_records)}
    next_cluster_id = num_records

    for merge_idx, merge_info in enumerate(merge_history):
        c1, c2 = merge_info['cluster1'], merge_info['cluster2']
        linkage_matrix[merge_idx, 0] = cluster_id_map[c1]
        linkage_matrix[merge_idx, 1] = cluster_id_map[c2]
        linkage_matrix[merge_idx, 2] = merge_info['distance']
        linkage_matrix[merge_idx, 3] = merge_info['resulting_size']
        cluster_id_map[c1] = next_cluster_id
        cluster_id_map[c2] = next_cluster_id
        next_cluster_id += 1

    # plot dendrogram
    print("\nDENDROGRAM - Last 20 Clusters\n")
    plt.figure(figsize=(14, 8))
    dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
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

    record_to_cluster_assignment = np.zeros(num_records, dtype=int)
    clusters_at_k = {i: [i] for i in range(num_records)}
    active_ids_at_k = set(range(num_records))

    for merge_info in merge_history[:-num_final_clusters + 1]:
        ci, cj = merge_info['cluster1'], merge_info['cluster2']
        if ci in active_ids_at_k and cj in active_ids_at_k:
            clusters_at_k[ci].extend(clusters_at_k[cj])
            del clusters_at_k[cj]
            active_ids_at_k.remove(cj)

    final_cluster_list = sorted(active_ids_at_k)
    for new_cid, orig_cid in enumerate(final_cluster_list):
        for record_index in clusters_at_k[orig_cid]:
            record_to_cluster_assignment[record_index] = new_cid

    cluster_sizes, cluster_prototypes = [], []
    for cid in range(num_final_clusters):
        members = np.where(record_to_cluster_assignment == cid)[0]
        cluster_sizes.append(len(members))
        cluster_prototypes.append(np.mean(data_matrix[members], axis=0))

    print("\nCLUSTER SIZES (from smallest to largest):\n")
    size_order = np.argsort(cluster_sizes)
    for cluster_id in size_order:
        print(f"Cluster {cluster_id}: {cluster_sizes[cluster_id]} members")

    print("\nCLUSTER PROTOTYPES (Average Attribute Values):\n")
    for cluster_id in range(num_final_clusters):
        print(f"\nCluster {cluster_id} (Size: {cluster_sizes[cluster_id]}):")
        proto_df = pd.DataFrame({
            'Attribute': attribute_names,
            'Average Value': np.round(cluster_prototypes[cluster_id], 2)
        })
        print(proto_df.to_string(index=False))

    results_df = pd.DataFrame({
        'Guest_ID': guest_ids,
        'Cluster_Assignment': record_to_cluster_assignment
    })
    results_df.to_csv('cluster_assignments.csv', index=False)
    print("\nCluster assignments saved to 'cluster_assignments.csv'")

    print(f"\nTotal records processed: {num_records}")
    print(f"Number of attributes: {num_attributes}")
    print(f"Final number of clusters: {num_final_clusters}")

if __name__ == "__main__":
    main()
