import random
from utils import Utils


class CMeansClustering:

    def __init__(self, alpha):
        self.alpha = alpha

    def _get_cluster_id(self, cluster_centers, item):
        cluster_id = -1
        min_distance = float('inf')

        for i, center in enumerate(cluster_centers):
            dist = Utils.euclidean_distance(
                Utils.remove_class_id(center), Utils.remove_class_id(item))

            if dist < min_distance:
                min_distance = dist
                cluster_id = i

        return cluster_id

    def _get_cluster_center(self, cluster):
        center = cluster[0][:]

        for item in cluster[1:]:
            for i, val in enumerate(item):
                center[i] += val

        length = len(cluster)

        return [val / length for val in center]

    def _are_cluster_centers_changed(self, cluster_centers, new_cluster_centers):
        for i, center in enumerate(cluster_centers):
            dist = Utils.euclidean_distance(
                Utils.remove_class_id(center), Utils.remove_class_id(new_cluster_centers[i]))

            if dist > self.alpha:
                return True

        return False

    def perform(self, dataset, n):
        indexes = [random.randint(0, len(dataset) - 1) for _ in range(n)]

        cluster_centers = [dataset[i] for i in indexes]
        clusters = {}

        while True:
            clusters.clear()

            for item in dataset[:]:
                cluster_id = self._get_cluster_id(cluster_centers, item)

                if cluster_id in clusters:
                    clusters[cluster_id].append(item)
                else:
                    clusters[cluster_id] = [item]

            new_cluster_centers = [self._get_cluster_center(
                cluster) for cluster in clusters.values()]

            if not self._are_cluster_centers_changed(cluster_centers, new_cluster_centers):
                break

            cluster_centers = new_cluster_centers

        return clusters.values()
