from typing import Dict

import numpy as np
import csv
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def read_possession_data(path: str) -> Dict:
    """
    Read the possession data which is selected
    :param path: the path of the data
    :return: a dictionary, contains the data about 4 type possession
    """

    # init 4 corresponding lists for 4 type possession
    stock_features, bond_features, major_commodities_features, fund_features = [], [], [], []

    with open(path, 'r', encoding='utf-8') as file:
        # construct the csv_reader for reading the data
        csv_reader = csv.reader(file)

        # remove the first line (titles)
        csv_reader.__next__()

        for line in csv_reader:
            # read data from CSV file and change the data type
            current_data = [float(item) for item in line[1:]]

            # init the current features of different possession
            stock_features.append(current_data[:4])
            major_commodities_features.append(current_data[4:6])
            bond_features.append(current_data[6:-1])
            fund_features.append(current_data[-1])

        return {
            "stock": stock_features,
            "major_commodities": major_commodities_features,
            "bond": bond_features,
            "fund": fund_features
        }


if __name__ == '__main__':

    data_dict = read_possession_data(path="data/possession/possession_PCA.csv")

    # the monetary fund only contains 1 feature, so we don't handle it, but we need to store them at arrays
    stock_array = np.array(data_dict["stock"])
    major_commodities_array = np.array(data_dict["major_commodities"])
    bond_array = np.array(data_dict["bond"])
    fund_array = np.array(data_dict["fund"])

    # scale the feature data of stock, major commodities, bond and fund
    scaler = MinMaxScaler()

    # scaling
    scaled_stock = scaler.fit_transform(stock_array)
    scaled_major_commodities = scaler.fit_transform(major_commodities_array)
    scaled_bond = scaler.fit_transform(bond_array)
    scaled_fund = scaler.fit_transform(fund_array.reshape(-1, 1))

    # utilize the PCA and kernel PCA theory to decomposition to 1-d
    kernel_pca = KernelPCA(n_components=1, kernel="poly", fit_inverse_transform=True)
    pca = PCA(n_components=1)
    stock_main_feature = pca.fit_transform(scaled_stock)
    print("Stock: ")
    print("residual information ratio: ")
    print(pca.explained_variance_ratio_[0])
    print("Weights: ")
    print(pca.components_[0])
    print()

    # major_commodities_main_feature = kernel_pca.fit_transform(scaled_major_commodities)
    # print(major_commodities_main_feature)
    # print()
    major_commodities_main_feature = pca.fit_transform(scaled_major_commodities)
    print("Major Commodities: ")
    print("residual information ratio: ")
    print(pca.explained_variance_ratio_)
    print("Weights: ")
    print(pca.components_)
    print()

    bond_main_feature = pca.fit_transform(scaled_bond)
    print("Bond: ")
    print("residual information ratio: ")
    print(pca.explained_variance_ratio_[0])
    print("Weights: ")
    print(pca.components_[0])
    print()

    # correlation analysis
    possession_feature_matrix = np.concatenate(
            (stock_main_feature, major_commodities_main_feature, bond_main_feature, scaled_fund),
            axis=1
    )

    # get the correlation table
    possession_feature_table = pd.DataFrame(
        possession_feature_matrix
    )
    print(possession_feature_table.corr())
