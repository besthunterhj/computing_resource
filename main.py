import numpy as np
import csv
from sklearn.cluster import DBSCAN


# def read_data(file_name: str):



if __name__ == '__main__':

    # with open('popularity.csv', 'r', encoding='utf-8') as file:
    #     csv_reader = csv.reader(file)
    #     csv_reader.__next__()
    #
    #     for line in csv_reader:
    #         print(line)

    cluster = DBSCAN(eps=3, min_samples=2)
    test = np.array(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [3, 3],
            [10, 20]
        ]
    )

    result = cluster.fit_predict(test)
    print(result)