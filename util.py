import csv
import math
import operator
import random
import itertools


class Util:
    @staticmethod
    def euclidean_distance(item_one, item_two):
        distance = 0

        for i in range(len(item_one)):
            distance += pow((item_one[i] - item_two[i]), 2)

        return math.sqrt(distance)

    @staticmethod
    def load_dataset(filename):
        dataset = []

        with open(filename, 'r') as csvfile:
            lines = csv.reader(csvfile)

            for line in lines:
                dataset.append([float(val) for val in line])

        return dataset

    @staticmethod
    def split_dataset(dataset, ratio):
        training_set = []
        test_set = []

        dataset.sort(key=operator.itemgetter(-1))

        for _, group in itertools.groupby(dataset, lambda item: item[-1]):
            items = [g for g in group]

            random.shuffle(items)

            n = round(len(items) * ratio)
            training_set += items[:n]
            test_set += items[n:]

        return training_set, test_set

    @staticmethod
    def get_accuracy(result):
        hits_count = 0

        for item, class_num in result:
            if item[-1] == class_num:
                hits_count += 1

        return hits_count / len(result)
