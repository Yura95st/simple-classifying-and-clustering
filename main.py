from utils import Utils
from dataset_utils import DatasetUtils

from k_nearest_neighbors_classifier import KNearestNeighborsClassifier

from c_means_clustering import CMeansClustering
from nearest_neighbors_clustering import NearestNeighborsClustering


def classifying(dataset, k=5, training_set_ratio=0.75):
    training_set, test_set = DatasetUtils.split(dataset, training_set_ratio)

    classifier = KNearestNeighborsClassifier(training_set, k)

    result = [(classifier.classify(item), item) for item in test_set]

    for item in result:
        print(item)
    print()
    print('Accuracy: {:.2}'.format(Utils.get_accuracy(result)))


def clustering(dataset, clusters_num=3, alpha=0.005):
    clusterings = [NearestNeighborsClustering(), CMeansClustering(alpha)]

    for clustering in clusterings[1:]:
        clusters = clustering.perform(dataset, clusters_num)

        for cluster in clusters:
            for item in cluster:
                print(item)
            print()

        print('Quality: {:.2}'.format(Utils.get_clustering_quality(clusters)))


def main():
    names = ['abalone', 'abalone_grouped', 'iris', 'wine']

    for name in names[2:]:
        dataset = DatasetUtils.load_from_cvs('data/{0}/{0}.normilized.data'.format(name))

        classifying(dataset)
        # clustering(dataset)


if __name__ == '__main__':
    main()
