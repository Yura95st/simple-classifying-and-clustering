from util import Util
from k_nearest_neighbors_classifier import KNearestNeighborsClassifier
from nearest_neighbors_clustering import NearestNeighborsClustering
from c_means_clustering import CMeansClustering


def main_classifying(k=3, training_set_ratio=0.2):
    dataset = Util.load_dataset('data/iris.data')

    training_set, test_set = Util.split_dataset(dataset, training_set_ratio)

    classifier = KNearestNeighborsClassifier(training_set, k)

    result = [(item, classifier.classify(item)) for item in test_set]

    for item in result:
        print(item)
    print()
    print('Accuracy: {}'.format(Util.get_accuracy(result)))


def main_clustering(clusters_num=3, alpha=0.005):
    dataset = Util.load_dataset('data/iris.data')

    # clusterings = [NearestNeighborsClustering(), CMeansClustering(alpha)]
    clusterings = [CMeansClustering(alpha)]

    for clustering in clusterings:
        clusters = clustering.perform(dataset, clusters_num)

        for cluster in clusters:
            for item in cluster:
                print(item)
            print()


if __name__ == '__main__':
    # main_classifying()
    main_clustering()
