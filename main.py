from utils import Utils
from dataset_utils import DatasetUtils

from k_nearest_neighbors_classifier import KNearestNeighborsClassifier

from c_means_clustering import CMeansClustering
from nearest_neighbors_clustering import NearestNeighborsClustering


def classifying(dataset, k=5, n_folds=5):
    dataset_folds = DatasetUtils.cross_validation_split(dataset, n_folds)

    accuracies = []

    for test_set in dataset_folds:
        training_set = dataset_folds[:]
        training_set.remove(test_set)
        training_set = sum(training_set, [])

        classifier = KNearestNeighborsClassifier(training_set, k)

        result = [(classifier.classify(item), item) for item in test_set]

        accuracies.append(Utils.get_accuracy(result))

    print('Accuracies: {}'.format(', '.join('{:.3}'.format(a) for a in accuracies)))
    print('Mean Accuracy: {:.3}'.format(
        sum(accuracies) / len(accuracies)))
    print()


def clustering(dataset, clusters_num=3, alpha=0.001):
    clusterings = [NearestNeighborsClustering(), CMeansClustering(alpha)]

    for clustering in clusterings:
        clusters = clustering.perform(dataset, clusters_num)

        for cluster in clusters:
            for item in cluster:
                print(item)
            print()

        print('Quality: {:.2}'.format(Utils.get_clustering_quality(clusters)))


def main():
    names = ['abalone', 'abalone_grouped', 'iris', 'wine']

    for name in names:
        dataset = DatasetUtils.load_from_cvs(
            'data/{0}/{0}.normilized.data'.format(name))

        print('Dataset: {}'.format(name))
        classifying(dataset)
        clustering(dataset)


if __name__ == '__main__':
    main()
