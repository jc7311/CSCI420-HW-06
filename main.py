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

    corr_matrix = features.corr().round(2)
    
    corr_matrix.to_csv('correlated_matrix.csv')
