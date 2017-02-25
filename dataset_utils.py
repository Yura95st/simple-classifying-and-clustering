import csv
import operator
import random
import itertools
from config import CLASS_ID_POSITION


class DatasetUtils:

    @staticmethod
    def load_from_cvs(filename):
        dataset = []

        with open(filename, 'r') as csvfile:
            lines = csv.reader([line for line in csvfile if line.strip()])

            for line in lines:
                dataset.append([DatasetUtils._parse_float(val)
                                for val in line])

        return dataset

    @staticmethod
    def split(dataset, ratio):
        training_set = []
        test_set = []

        dataset.sort(key=operator.itemgetter(CLASS_ID_POSITION))

        for _, group in itertools.groupby(dataset, lambda item: item[CLASS_ID_POSITION]):
            items = [g for g in group]

            random.shuffle(items)

            n = round(len(items) * ratio)
            training_set += items[:n]
            test_set += items[n:]

        return training_set, test_set

    @staticmethod
    def normalize(dataset):
        normalized_dataset = dataset[:]

        for i, item in enumerate(normalized_dataset):
            normalized_dataset[i] = item[-1:] + item[:-1]

        for i in range(1, len(normalized_dataset[0])):
            column = [item[i] for item in normalized_dataset]

            max_val = max(column)
            min_val = min(column)

            for item in normalized_dataset:
                item[i] = (item[i] - min_val) / (max_val - min_val)

        return normalized_dataset

    @staticmethod
    def write_to_cvs(dataset, output_file_name):
        with open(output_file_name, 'w', newline='') as datafile:
            writer = csv.writer(datafile)

            for item in dataset:
                writer.writerow(item)

    @staticmethod
    def _parse_float(val):
        try:
            return float(val)
        except ValueError:
            return val
