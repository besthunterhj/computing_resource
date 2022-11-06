from typing import Dict, List

import statsmodels.api as sm
import pandas as pd
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from scipy import signal


def read_marco_economy_data(path: str) -> Dict:
    """
    Read the macro_economy data from specific path
    :param path: the path of macro_economy data
    :return: a dictionary, which contains leading indexes and lagging indexes
    """

    leading_dataset, lagging_dataset = [], []
    date = []

    with open(path, 'r', encoding='utf-8') as file:
        # remove the title of the file
        csv_reader = csv.reader(file)
        csv_reader.__next__()

        for line in csv_reader:
            date.append(line[0])
            leading_dataset.append(float(line[1]))
            lagging_dataset.append(float(line[-1]))

    return {
        "date": date,
        "leading": leading_dataset,
        "lagging": lagging_dataset
    }


def construct_time_to_marco_economy_type(date: List[str], important_times: List[str]) -> Dict:

    time_to_marco_economy_type = {}
    for i in range(len(important_times) - 2):
        # store the time which belongs to overheat
        if i % 4 == 0:
            start_overheat_index = 0
            if i != 0:
                start_overheat_index = date.index(important_times[i - 1])

            end_overheat_index = date.index(important_times[i])

            for j in range(start_overheat_index, end_overheat_index):
                time_to_marco_economy_type[date[j]] = "Overheat"

        # store the time which belongs to stagflation
        elif i % 4 == 1:
            start_stagflation_index = date.index(important_times[i - 1])
            end_stagflation_index = date.index(important_times[i])

            for j in range(start_stagflation_index, end_stagflation_index):
                time_to_marco_economy_type[date[j]] = "Stagflation"

        # store the time which belongs to reflation
        elif i % 4 == 2:
            start_reflation_index = date.index(important_times[i - 1])
            end_reflation_index = date.index(important_times[i])

            for j in range(start_reflation_index, end_reflation_index):
                time_to_marco_economy_type[date[j]] = "Reflation"

        # store the time which belongs to recovery
        else:
            start_recovery_index = date.index(important_times[i - 1])
            end_recovery_index = date.index(important_times[i])

            for j in range(start_recovery_index, end_recovery_index):
                time_to_marco_economy_type[date[j]] = "Recovery"

    # handle the complex station between 2019-09 to 2021-12
    for i in range(len(important_times) - 2, len(important_times)):
        start_index = date.index(important_times[i - 1])
        end_index = date.index(important_times[i])

        if i == len(important_times) - 2:
            for j in range(start_index, end_index):
                time_to_marco_economy_type[date[j]] = "Overheat"
                continue

        if i == len(important_times) - 1:
            for j in range(start_index, end_index):
                time_to_marco_economy_type[date[j]] = "Recovery"

    for i in range(date.index(important_times[-1]), len(date)):
        time_to_marco_economy_type[date[i]] = "Reflation"

    return time_to_marco_economy_type


if __name__ == '__main__':
    # load the marco_economy data and init with DataFrame
    marco_data = read_marco_economy_data("data/2001_2021/marco_economy_data.csv")
    leading_data_frame = pd.DataFrame(marco_data["leading"])
    lagging_data_frame = pd.DataFrame(marco_data["lagging"])
    date = marco_data["date"]

    # clear the data by hp_filter
    leading_cycle, leading_trend = sm.tsa.filters.hpfilter(leading_data_frame)
    lagging_cycle, lagging_trend = sm.tsa.filters.hpfilter(lagging_data_frame)

    # construct the feature matrix and table
    cleared_leading = np.array(leading_trend).reshape(-1, 1)
    cleared_lagging = np.array(lagging_trend).reshape(-1, 1)
    marco_economy_features = np.concatenate((cleared_leading, cleared_lagging), axis=1)
    # feature_table = pd.DataFrame(data=marco_economy_features, index=date, columns=["leading", "lagging"])
    # feature_table.to_csv("test.csv")

    # search for the relmax, relmin values of leading and lagging data
    leading_relmax = signal.argrelmax(cleared_leading)[0]
    leading_relmin = signal.argrelmin(cleared_leading)[0]
    lagging_relmax = signal.argrelmax(cleared_lagging)[0]
    lagging_relmin = signal.argrelmin(cleared_lagging)[0]

    print("leading relmax")
    print([date[i] for i in leading_relmax])
    print()
    print("leading relmin")
    print([date[i] for i in leading_relmin])
    print()

    print("lagging relmax")
    print([date[i] for i in lagging_relmax])
    print()
    print("lagging relmin")
    print([date[i] for i in lagging_relmin])
    print()

    # we aggregate these 4 type values to construct the important time points list
    # each important time point means a relative maximum value or relative minimum value
    important_time_points = np.sort(np.concatenate((leading_relmax, leading_relmin, lagging_relmax, lagging_relmin)))
    important_times = [date[i] for i in important_time_points]
    print(important_times)
    print()

    # construct a dictionary to store the time and its corresponding marco economy type
    time_to_marco_economy_type = construct_time_to_marco_economy_type(date=date, important_times=important_times)
    # write the dictionary to a json file
    with open('time_to_marco_economy_type.json', 'w', encoding='utf-8') as f_obj:
        f_obj.write(json.dumps(time_to_marco_economy_type))

    # test for clustering
    # K-Means
    # leading_data_array = np.array(leading_data_frame)
    # lagging_data_array = np.array(lagging_data_frame)
    # marco_economy_features_without_filter = np.concatenate((leading_data_array, lagging_data_array), axis=1)
    # scaler = MinMaxScaler()
    # scaled_features = scaler.fit_transform(marco_economy_features_without_filter)
    #
    # kmeans = KMeans(n_clusters=4)
    # kmeans.fit(scaled_features)
    # print(kmeans.predict(scaled_features))

    # with filter
    # kmeans = KMeans(n_clusters=4)
    # kmeans.fit_transform(marco_economy_features)
    # kmeans_result = kmeans.predict(marco_economy_features)
    # print(kmeans_result)
    # print(len(kmeans_result))

    # draw the line picture
    # plt.plot(date[120:], feature_table["leading"]["2011-01":], color='blue', label="leading")
    # plt.plot(date[120:], feature_table["lagging"]["2011-01":], color='orange', label="lagging")
    # plt.ylabel("values")
    # plt.xlabel("year and month")
    # plt.tick_params(axis='both', labelsize=10)
    # plt.xticks(rotation=90, fontsize=5)
    # plt.show()
    # Example

    # data = sm.datasets.macrodata.load_pandas().data
    # print(data.realgdp)
    # print()
    # print(type(data.realgdp))
    # exit()
    #
    # cycle, trend = sm.tsa.filters.hpfilter(data.realgdp, 1600)
    #
    # print(trend)
    # print()
    #
    # # gdp_decomp is a DataFrame but data.realgdp is not
    # gdp_decomp = data[['realgdp']]
    # # gdp_decomp["cycle"] = cycle
    # # gdp_decomp["trend"] = trend


