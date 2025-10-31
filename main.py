# Ethan Chang
# HW 06 Agglomeration
# CSCI 420

import csv
import pandas as pd
import numpy as np

def main():

    df = pd.read_csv('HW_CLUSTERING_SHOPPING_CART_v2245a.csv')

    print(df)

    print(list(df.columns))
    print(df.shape)
    print(df.head(3))

    guest_id = next(c for c in df.columns if 'id' in c.lower())
    features = df.drop(columns=[guest_id])

    full_correlation = features.corr(numeric_only=True).round(2)

    print(full_correlation)
    

if __name__ == "__main__":

    main()