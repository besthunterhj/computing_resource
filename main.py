import numpy as np
import csv


# def read_data(file_name: str):



if __name__ == '__main__':

    with open('popularity.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        csv_reader.__next__()

        for line in csv_reader:
            print(line)
