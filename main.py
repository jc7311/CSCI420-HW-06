import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

data_file = 'HW_CLUSTERING_SHOPPING_CART_v2245a.csv'
# data_file = 'test.csv'
dataframe = pd.read_csv(data_file)

guest_ids = dataframe['ID'].values
attribute_names = dataframe.columns[1:].tolist()
data_matrix = dataframe.iloc[:, 1:].values

num_records = data_matrix.shape[0]
num_attributes = data_matrix.shape[1]

print("\nPART A: CROSS-CORRELATION ANALYSIS\n")

correlation_matrix = np.corrcoef(data_matrix.T)

print("\nCross-Correlation Matrix (rounded to 2 decimal places):")
correlation_df = pd.DataFrame(
    np.round(correlation_matrix, 2),
    index=attribute_names,
    columns=attribute_names
)
print(correlation_df)

correlation_df.to_csv('correlation_matrix.csv')
print("\nCorrelation matrix saved to 'correlation_matrix.csv'")