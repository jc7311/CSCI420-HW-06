# Ethan Chang
# HW 06 Agglomeration
# CSCI 420

import csv


def main():

    with open('HW_CLUSTERING_SHOPPING_CART_v2245a.csv', 'r', newline='') as csvfile:

    # Create a csv.reader object
        csv_reader = csv.reader(csvfile)

        # If your CSV has a header row, you can skip it
        header = next(csv_reader)
        print(f"Header: {header}")

        # Iterate over each row in the CSV file
        #for row in csv_reader:
        #    print(row)
            


if __name__ == "__main__":

    main()