import os
import copy
import utils
from models import *

diabetes_data = utils.clean_dataset('dataset/diabetes_prediction_dataset.csv', 'diabetes')
heart_data = utils.clean_dataset('dataset/heart_attack_prediction_dataset.csv', 'HeartDiseaseorAttack')

if not os.path.isdir('results'):
    os.makedirs('results')

# Normal run with no data transformation
# kmeans_run(heart_data, 10, 'results/kmeans_heart_data')
# kmeans_run(diabetes_data, 10, 'results/kmeans_diabetes_data')
#
# em_run(heart_data, 10, 'results/em_heart_data')
# em_run(diabetes_data, 10, 'results/em_diabetes_data')

# PCA data transformation
pca_heart_data = pca_run(copy.deepcopy(heart_data), 'Heart Dataset')
pca_diabetes_data = pca_run(copy.deepcopy(diabetes_data), 'Diabetes Dataset')

# kmeans_run(pca_heart_data, 10, 'results/pca_kmeans_heart_data')
# kmeans_run(pca_diabetes_data, 10, 'results/pca_kmeans_diabetes_data')
#
# em_run(pca_heart_data, 10, 'results/pca_em_heart_data')
# em_run(pca_diabetes_data, 10, 'results/pca_em_diabetes_data')
#
# nn_run(pca_heart_data, 'PCA_Heart_Dataset')
# nn_kmeans_run(pca_heart_data, 'KMEANS_PCA_Heart_Dataset')
# nn_em_run(pca_heart_data, 'EM_PCA_Heart_Dataset')

# ICA data transformation
# ica_heart_data = ica_run(copy.deepcopy(heart_data), "Heart Dataset")
# ica_diabetes_data = ica_run(copy.deepcopy(diabetes_data), 'Diabetes Dataset')
#
# kmeans_run(ica_heart_data, 10, 'results/ica_kmeans_heart_data')
# kmeans_run(ica_diabetes_data, 10, 'results/ica_kmeans_diabetes_data')
#
# em_run(ica_heart_data, 10, 'results/ica_em_heart_data')
# em_run(ica_diabetes_data, 10, 'results/ica_em_diabetes_data')
#
# nn_run(ica_heart_data, 'ICA_Heart_Dataset')
# nn_kmeans_run(ica_heart_data, 'KMEANS_ICA_Heart_Dataset')
# nn_em_run(ica_heart_data, 'EM_ICA_Heart_Dataset')
#
# # GRP data transformation
# grp_heart_data = grp_run(copy.deepcopy(heart_data), 'Heart Dataset')
# grp_diabetes_data = grp_run(copy.deepcopy(diabetes_data), 'Diabetes Dataset')
#
# kmeans_run(grp_heart_data, 10, 'results/grp_kmeans_heart_data')
# kmeans_run(grp_diabetes_data, 10, 'results/grp_kmeans_diabetes_data')
#
# em_run(grp_heart_data, 10, 'results/grp_em_heart_data')
# em_run(grp_diabetes_data, 10, 'results/grp_em_diabetes_data')
#
# nn_run(grp_heart_data, 'GRP_Heart_Dataset')
# nn_kmeans_run(grp_heart_data, 'KMEANS_GRP_Heart_Dataset')
# nn_em_run(grp_heart_data, 'EM_GRP_Heart_Dataset')
