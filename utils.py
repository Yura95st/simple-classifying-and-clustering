import math
import operator
import itertools
from config import CLASS_ID_POSITION


class Utils:

    @staticmethod
    def euclidean_distance(item_one, item_two):
        distance = 0

        for i in range(len(item_one)):
            distance += pow((item_one[i] - item_two[i]), 2)

        return math.sqrt(distance)

    @staticmethod
    def get_accuracy(result):
        hits_count = 0

        for class_num, item in result:
            if item[CLASS_ID_POSITION] == class_num:
                hits_count += 1

        return hits_count / len(result)

    @staticmethod
    def remove_class_id(item):
        new_item = item[:]

        del new_item[CLASS_ID_POSITION]

        return new_item

    @staticmethod
    def get_clustering_quality(clusters):
        total_items_count = sum([len(cluster) for cluster in clusters])

        errors_count = sum([Utils.get_clustering_errors_number(cluster)
                            for cluster in clusters])

        quality = 1.0 - (errors_count / total_items_count)

        return quality

    @staticmethod
    def get_clustering_errors_number(cluster):
        cluster.sort(key=operator.itemgetter(-1))

        right_items_count = max(
            [len(list(group)) for _, group in itertools.groupby(cluster, lambda item: item[CLASS_ID_POSITION])])

        errors_count = len(cluster) - right_items_count

        return errors_count
