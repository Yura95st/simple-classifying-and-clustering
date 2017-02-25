import operator
from utils import Utils
from config import CLASS_ID_POSITION


class KNearestNeighborsClassifier:

    def __init__(self, training_set, k):
        self.training_set = training_set
        self.k = k

    def _get_neighbors(self, test_item):
        distances = []

        for item in self.training_set:
            dist = Utils.euclidean_distance(Utils.remove_class_id(
                item), Utils.remove_class_id(test_item))

            distances.append((item, dist))

        distances.sort(key=operator.itemgetter(1))

        return [distances[i][0] for i in range(self.k)]

    def _get_most_common_class_number(self, neighbors):
        classes = [neighbor[CLASS_ID_POSITION] for neighbor in neighbors]

        return max(set(classes), key=classes.count)

    def classify(self, item):
        neighbors = self._get_neighbors(item)

        class_num = self._get_most_common_class_number(neighbors)

        return class_num
