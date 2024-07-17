import yaml
import numpy as np
import pandas as pd

from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.neural_network import MLPClassifier
from sklearn.random_projection import GaussianRandomProjection


def kmeans_run(dataset, total_clusters, filename):
    clusters = range(2, total_clusters + 1, 1)

    run_results = {}
    for cluster in clusters:
        run_results[cluster] = {}

        model = KMeans(init='k-means++',
                       n_clusters=cluster,
                       n_init=10,
                       random_state=1,
                       max_iter=100)

        start_time = time()

        model.fit(dataset[0])

        run_results[cluster]['time'] = time() - start_time

        predictions = model.predict(dataset[0])

        run_results[cluster]['homo'] = float(metrics.homogeneity_score(dataset[2], predictions))
        run_results[cluster]['complete'] = float(metrics.completeness_score(dataset[2], predictions))
        run_results[cluster]['vms'] = float(metrics.v_measure_score(dataset[2], predictions))
        run_results[cluster]['ars'] = float(metrics.adjusted_rand_score(dataset[2], predictions))
        run_results[cluster]['amis'] = float(metrics.adjusted_mutual_info_score(dataset[2], predictions))
        run_results[cluster]['accuracy'] = float(metrics.accuracy_score(dataset[2], predictions))
        run_results[cluster]['silo'] = float(metrics.silhouette_score(dataset[0], predictions, sample_size=400))

    with open(f'{filename}.yml', 'w') as results:
        yaml.dump(run_results, results)


def em_run(dataset, total_clusters, filename):
    clusters = range(2, total_clusters + 1, 1)

    run_results = {}
    for cluster in clusters:
        run_results[cluster] = {}

        model = GaussianMixture(n_components=cluster,
                                covariance_type='full',
                                n_init=100)

        start_time = time()

        model.fit(dataset[0])

        run_results[cluster]['time'] = time() - start_time

        predictions = model.predict(dataset[0])

        run_results[cluster]['homo'] = float(metrics.homogeneity_score(dataset[2], predictions))
        run_results[cluster]['complete'] = float(metrics.completeness_score(dataset[2], predictions))
        run_results[cluster]['vms'] = float(metrics.v_measure_score(dataset[2], predictions))
        run_results[cluster]['ars'] = float(metrics.adjusted_rand_score(dataset[2], predictions))
        run_results[cluster]['amis'] = float(metrics.adjusted_mutual_info_score(dataset[2], predictions))
        run_results[cluster]['accuracy'] = float(metrics.accuracy_score(dataset[2], predictions))
        run_results[cluster]['silo'] = float(metrics.silhouette_score(dataset[0], predictions, sample_size=400))

    with open(f'{filename}.yml', 'w') as results:
        yaml.dump(run_results, results)


def pca_run(dataset, dataset_name=None):
    variance_results = np.around(PCA(svd_solver='full').fit(dataset[0]).explained_variance_ratio_.cumsum(), 2)

    if dataset_name is None:
        plt.plot(range(1, len(dataset[0].columns) + 1), variance_results)
        plt.xlabel('# of Components')
        plt.ylabel('Explained Variance (%)')
        plt.title(f'PCA - Explained Variance {dataset_name}')
        plt.savefig(f'results/pca-{dataset_name}.jpg')
        plt.close()

    dim_red_model = PCA(svd_solver='full',
                        n_components=int(np.where(variance_results == max(variance_results))[0][0]) + 1)

    dataset[0] = dim_red_model.fit_transform(dataset[0])
    dataset[1] = dim_red_model.fit_transform(dataset[1])
    return dataset


def ica_run(dataset, dataset_name):
    dim_red_model = FastICA()

    kurtosis_val = []
    for component in range(1, len(dataset[0].columns) + 1):
        dim_red_model.set_params(n_components=component)
        kurt_score = pd.DataFrame(dim_red_model.fit_transform(dataset[0])).kurt(axis=0)
        kurtosis_val.append(kurt_score.abs().mean())

    plt.plot(range(1, len(dataset[0].columns) + 1), kurtosis_val)
    plt.xlabel('# of Components')
    plt.ylabel('Average Kurtosis')
    plt.title(f'ICA - Average Kurtosis {dataset_name}')
    plt.savefig(f'results/ica-{dataset_name}.jpg')
    plt.close()

    dim_red_model = PCA(svd_solver='full', n_components=kurtosis_val.index(max(kurtosis_val)) + 1)

    dataset[0] = dim_red_model.fit_transform(dataset[0])
    dataset[1] = dim_red_model.fit_transform(dataset[1])

    return dataset


def grp_run(dataset, dataset_name):
    dim_red_model = GaussianRandomProjection()

    kurtosis_val = []
    for component in range(1, len(dataset[0].columns) + 1):
        dim_red_model.set_params(n_components=component)
        kurt_score = pd.DataFrame(dim_red_model.fit_transform(dataset[0])).kurt(axis=0)
        kurtosis_val.append(kurt_score.abs().mean())

    plt.plot(range(1, len(dataset[0].columns) + 1), kurtosis_val)
    plt.xlabel('# of Components')
    plt.ylabel('Average Kurtosis (%)')
    plt.title(f'GRP - Average Kurtosis {dataset_name}')
    plt.savefig(f'results/grp-{dataset_name}.jpg')
    plt.close()

    dim_red_model = GaussianRandomProjection(n_components=kurtosis_val.index(max(kurtosis_val)) + 1)

    dataset[0] = dim_red_model.fit_transform(dataset[0])
    dataset[1] = dim_red_model.fit_transform(dataset[1])

    return dataset


def nn_run(dataset, name):
    model = MLPClassifier(hidden_layer_sizes=(512, 256, 128),
                          alpha=0.01,
                          learning_rate_init=0.01,
                          max_iter=1000)

    model.fit(dataset[0], dataset[2])

    training_predict = model.predict(dataset[0])
    testing_predict = model.predict(dataset[1])
    training_accuracy = metrics.accuracy_score(dataset[2], training_predict)
    testing_accuracy = metrics.accuracy_score(dataset[3], testing_predict)

    with open(f'results/nn_results.csv', 'a') as nn_file:
        nn_file.write(f'{name}_Training Accuracy: {training_accuracy}\n')
        nn_file.write(f'{name}_Testing Accuracy: {testing_accuracy}\n')


def nn_kmeans_run(dataset, name):
    cluster = KMeans(init='k-means++',
                     n_clusters=2,
                     n_init=10,
                     random_state=1,
                     max_iter=100)

    cluster.fit(dataset[0])
    np.append(dataset[0], cluster.predict(dataset[0]))
    np.append(dataset[1], cluster.predict(dataset[1]))

    model = MLPClassifier(hidden_layer_sizes=(512, 256, 128),
                          alpha=0.01,
                          learning_rate_init=0.01,
                          max_iter=1000)

    model.fit(dataset[0], dataset[2])

    training_predict = model.predict(dataset[0])
    testing_predict = model.predict(dataset[1])
    training_accuracy = metrics.accuracy_score(dataset[2], training_predict)
    testing_accuracy = metrics.accuracy_score(dataset[3], testing_predict)

    with open(f'results/nn_kmeans_results.csv', 'a') as nn_file:
        nn_file.write(f'{name}_Training Accuracy: {training_accuracy}\n')
        nn_file.write(f'{name}_Testing Accuracy: {testing_accuracy}\n')


def nn_em_run(dataset, name):
    cluster = GaussianMixture(n_components=6,
                              covariance_type='full',
                              n_init=100)

    cluster.fit(dataset[0])
    np.append(dataset[0], cluster.predict(dataset[0]))
    np.append(dataset[1], cluster.predict(dataset[1]))

    model = MLPClassifier(hidden_layer_sizes=(512, 256, 128),
                          alpha=0.01,
                          learning_rate_init=0.01,
                          max_iter=1000)

    model.fit(dataset[0], dataset[2])

    training_predict = model.predict(dataset[0])
    testing_predict = model.predict(dataset[1])
    training_accuracy = metrics.accuracy_score(dataset[2], training_predict)
    testing_accuracy = metrics.accuracy_score(dataset[3], testing_predict)

    with open(f'results/nn_em_results.csv', 'a') as nn_file:
        nn_file.write(f'{name}_Training Accuracy: {training_accuracy}\n')
        nn_file.write(f'{name}_Testing Accuracy: {testing_accuracy}\n')
