import operator
from util import Util

class KNearestNeighborsClassifier:

    def __init__(self, training_set, k):
        self.training_set = training_set
        self.k = k

    def _get_neighbors(self, test_item):
        distances = []

        for item in self.training_set:
            dist = Util.euclidean_distance(Util.remove_class_id(item), Util.remove_class_id(test_item))

            distances.append((item, dist))

        distances.sort(key=operator.itemgetter(1))

        return [distances[i][0] for i in range(self.k)]

    def _get_mode_class_number(self, neighbors):
        classes = {}

        for neighbor in neighbors:
            class_num = neighbor[-1]

            if class_num in classes:
                classes[class_num] += 1
            else:
                classes[class_num] = 1

        classes = sorted(classes, key=classes.get, reverse=True)

        return next(iter(classes))

    def classify(self, item):
        neighbors = self._get_neighbors(item)

        class_num = self._get_mode_class_number(neighbors)

        return class_num
