import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import heapq #to optimize part b

# load data
data_file = 'HW_CLUSTERING_SHOPPING_CART_v2245a.csv'
# for smaller sample size
# data_file = 'test.csv' 
dataframe = pd.read_csv(data_file)

# extract guest IDs and attribute names
guest_ids = dataframe['ID'].values
attribute_names = dataframe.columns[1:].tolist()
data_matrix = dataframe.iloc[:, 1:].values

# dimensions
num_records = data_matrix.shape[0]
num_attributes = data_matrix.shape[1]

# part A cross correlation
print("\nPART A: CROSS-CORRELATION ANALYSIS\n")

# compute correlation coefficients (20x20 matrix, values in [-1,1])
correlation_matrix = np.corrcoef(data_matrix.T)

# save matrix
print("\nCross-Correlation Matrix (rounded to 2 decimal places):")
correlation_df = pd.DataFrame(
    np.round(correlation_matrix, 2),
    index=attribute_names,
    columns=attribute_names
)
print(correlation_df)

correlation_df.to_csv('correlation_matrix.csv')
print("\nCorrelation matrix saved to 'correlation_matrix.csv'")

# part B agglomeration
print("\nPART B: AGGLOMERATIVE CLUSTERING\n")

# each starts as their own cluster
clusters = {}
for index in range(num_records):
    clusters[index] = [index]

cluster_centers = {}
for index in range(num_records):
    cluster_centers[index] = data_matrix[index].copy()

active_cluster_ids = set(range(num_records))
merge_history = []

def compute_euclidean_distance(center1, center2):
    """Euclidean distance between two clsuter centers."""
    return np.sqrt(np.sum((center1 - center2) ** 2))

def compute_cluster_center(cluster_member_indices):
    """Mean of all customers in cluster"""
    member_data = data_matrix[cluster_member_indices]
    return np.mean(member_data, axis=0)

print(f"\nStarting agglomerative clustering with {num_records} records")
print("Building initial distance heap...")

# use a prio queue to optimize distance
min_heap = []
distance_cache = {}

def pair_key(cluster_i, cluster_j):
    return (min(cluster_i, cluster_j), max(cluster_i, cluster_j))

active_list = list(range(num_records))
for idx_i in range(len(active_list)):
    cluster_i = active_list[idx_i]
    center_i = cluster_centers[cluster_i]
    
    for idx_j in range(idx_i + 1, len(active_list)):
        cluster_j = active_list[idx_j]
        center_j = cluster_centers[cluster_j]
        
        distance = compute_euclidean_distance(center_i, center_j)
        key = pair_key(cluster_i, cluster_j)
        distance_cache[key] = distance
        heapq.heappush(min_heap, (distance, cluster_i, cluster_j))

print("Initial heap built. Starting clustering...\n")

# iteratively merge the two closest clusters until only one remains
iteration_count = 0
total_merges = num_records - 1

while len(active_cluster_ids) > 1:
    iteration_count += 1
    
    if iteration_count % 100 == 0:
        print(f"Progress: {iteration_count}/{total_merges} merges completed, {len(active_cluster_ids)} clusters remaining")

    # find closest cluster pair using lazy deletion from heap
    # pop until we find a pair where both clusters still exist    
    while True:
        min_distance, cluster_i_to_merge, cluster_j_to_merge = heapq.heappop(min_heap)
        
        if cluster_i_to_merge in active_cluster_ids and cluster_j_to_merge in active_cluster_ids:
            break
    
    # record for analysis
    size_i = len(clusters[cluster_i_to_merge])
    size_j = len(clusters[cluster_j_to_merge])
    smaller_cluster_size = min(size_i, size_j)
    
    merge_history.append({
        'iteration': iteration_count,
        'cluster1': cluster_i_to_merge,
        'cluster2': cluster_j_to_merge,
        'size1': size_i,
        'size2': size_j,
        'smaller_size': smaller_cluster_size,
        'distance': min_distance,
        'resulting_size': size_i + size_j
    })
    
    # merge cluster j into clsuter i
    clusters[cluster_i_to_merge].extend(clusters[cluster_j_to_merge])
    
    cluster_centers[cluster_i_to_merge] = compute_cluster_center(clusters[cluster_i_to_merge])
    
    new_center = cluster_centers[cluster_i_to_merge]
    
    # update and recompute distances from merged cluster to all others
    for other_cluster in active_cluster_ids:
        if other_cluster != cluster_i_to_merge and other_cluster != cluster_j_to_merge:
            # remove old cache
            key_old_j = pair_key(cluster_j_to_merge, other_cluster)
            distance_cache.pop(key_old_j, None)  
            
            key_old_i = pair_key(cluster_i_to_merge, other_cluster)
            distance_cache.pop(key_old_i, None) 
            
            # compute and cache new dist
            other_center = cluster_centers[other_cluster]
            new_distance = compute_euclidean_distance(new_center, other_center)
            
            key_new = pair_key(cluster_i_to_merge, other_cluster)
            distance_cache[key_new] = new_distance
            heapq.heappush(min_heap, (new_distance, cluster_i_to_merge, other_cluster))

    # remove merged cluster
    clusters.pop(cluster_j_to_merge, None)
    cluster_centers.pop(cluster_j_to_merge, None)
    active_cluster_ids.remove(cluster_j_to_merge)

print(f"\nClustering complete! Total merges: {len(merge_history)}")

# cluster analysis
print("\nLAST 20 MERGES - Smallest Cluster Size Tracking\n")

# display last 20 merges
last_20_merges = merge_history[-20:]
for merge_info in last_20_merges:
    print(f"Merge {merge_info['iteration']:4d}: Cluster {merge_info['cluster1']:4d} (size {merge_info['size1']:4d}) + "
          f"Cluster {merge_info['cluster2']:4d} (size {merge_info['size2']:4d}) -> "
          f"Smaller cluster size: {merge_info['smaller_size']:4d}, Distance: {merge_info['distance']:.2f}")

# last 10 smallest cluster size
print("\nLAST 10 SMALLEST CLUSTER SIZES IN MERGES:\n")
last_10_smallest = [merge_info['smaller_size'] for merge_info in merge_history[-10:]]
print("Sizes:", last_10_smallest)

# convert merge history to expected format:
# Each row: [cluster_left, cluster_right, distance, sample_count]
linkage_matrix = np.zeros((len(merge_history), 4))

cluster_id_map = {i: i for i in range(num_records)}
next_cluster_id = num_records

for merge_idx, merge_info in enumerate(merge_history):
    cluster1_original = merge_info['cluster1']
    cluster2_original = merge_info['cluster2']
    
    cluster1_mapped = cluster_id_map[cluster1_original]
    cluster2_mapped = cluster_id_map[cluster2_original]
    
    linkage_matrix[merge_idx, 0] = cluster1_mapped
    linkage_matrix[merge_idx, 1] = cluster2_mapped
    linkage_matrix[merge_idx, 2] = merge_info['distance']
    linkage_matrix[merge_idx, 3] = merge_info['resulting_size']
    
    cluster_id_map[cluster1_original] = next_cluster_id
    next_cluster_id += 1

# create dendrogram
print("\nDENDROGRAM last 20 clusters\n")

plt.figure(figsize=(14, 8))
dendrogram_result = dendrogram(
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
print("\nDendrogram saved as 'dendrogram_last_20_clusters.png'")
plt.show()

# extract final clusters
# based on dendrogram and distance jumps, identified 4 natural clusters
num_final_clusters = 4
final_cluster_merges = merge_history[-(num_final_clusters-1):]

print(f"\nANALYZING FINAL {num_final_clusters} CLUSTERS\n")

record_to_cluster_assignment = np.zeros(num_records, dtype=int)

clusters_at_k = {}
for index in range(num_records):
    clusters_at_k[index] = [index]

active_ids_at_k = set(range(num_records))

# merge history stopping before the last merges
for merge_idx, merge_info in enumerate(merge_history[:-num_final_clusters+1]):
    cluster_i = merge_info['cluster1']
    cluster_j = merge_info['cluster2']
    
    if cluster_i in active_ids_at_k and cluster_j in active_ids_at_k:
        clusters_at_k[cluster_i].extend(clusters_at_k[cluster_j])
        del clusters_at_k[cluster_j]
        active_ids_at_k.remove(cluster_j)

# this assigns each customer to final cluster
final_cluster_list = list(active_ids_at_k)
final_cluster_list.sort()

for new_cluster_id, original_cluster_id in enumerate(final_cluster_list):
    for record_index in clusters_at_k[original_cluster_id]:
        record_to_cluster_assignment[record_index] = new_cluster_id

# cmpute clsuter sizes   and prototypes (avg purchase patterns)
cluster_sizes = []
cluster_prototypes = []

for cluster_id in range(num_final_clusters):
    member_indices = np.where(record_to_cluster_assignment == cluster_id)[0]
    cluster_size = len(member_indices)
    cluster_sizes.append(cluster_size)
    
    cluster_prototype = compute_cluster_center(member_indices)
    cluster_prototypes.append(cluster_prototype)

# display resutls
print("\nCLUSTER SIZES (from smallest to largest):\n")
size_order = np.argsort(cluster_sizes)
for rank, cluster_id in enumerate(size_order):
    print(f"Cluster {cluster_id}: {cluster_sizes[cluster_id]} members")

print("\nCLUSTER PROTOTYPES (Average Attribute Values):\n")
for cluster_id in range(num_final_clusters):
    print(f"\nCluster {cluster_id} (Size: {cluster_sizes[cluster_id]}):")
    prototype_df = pd.DataFrame({
        'Attribute': attribute_names,
        'Average Value': np.round(cluster_prototypes[cluster_id], 2)
    })
    print(prototype_df.to_string(index=False))

# save cluster assignments for marketing team
results_df = pd.DataFrame({
    'Guest_ID': guest_ids,
    'Cluster_Assignment': record_to_cluster_assignment
})
results_df.to_csv('cluster_assignments.csv', index=False)
print("\nCluster assignments saved to 'cluster_assignments.csv'\n")

print(f"\nTotal records processed: {num_records}")
print(f"Number of attributes: {num_attributes}")
print(f"Final number of clusters: {num_final_clusters}")