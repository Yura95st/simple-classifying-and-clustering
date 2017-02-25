from utils import Utils


class NearestNeighborsClustering:

    def _get_distance(self, cluster_one, cluster_two):
        min_distance = float('inf')

        for item_one in cluster_one:
            for item_two in cluster_two:
                dist = Utils.euclidean_distance(Utils.remove_class_id(item_one), Utils.remove_class_id(item_two))

                if dist < min_distance:
                    min_distance = dist

        return min_distance

    def _get_clusters_to_join(self, clusters):
        length = len(clusters)

        min_distance = float('inf')
        clusters_to_join = (-1, -1)

        for i in range(length):
            for j in range(i + 1, length):
                dist = self._get_distance(clusters[i], clusters[j])

                if dist < min_distance:
                    min_distance = dist
                    clusters_to_join = i, j

        return clusters_to_join

    def perform(self, dataset, n):
        clusters = [[item] for item in dataset]

        while n < len(clusters):
            clusters_to_join = self._get_clusters_to_join(clusters)

            clusters[clusters_to_join[0]] += clusters[clusters_to_join[1]]
            del clusters[clusters_to_join[1]]

        return clusters
