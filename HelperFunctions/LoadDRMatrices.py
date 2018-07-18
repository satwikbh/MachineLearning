import glob

import numpy as np

from PrepareData.LoadData import LoadData
from Utils.LoggerUtil import LoggerUtil


class LoadDRMatrices:
    def __init__(self, use_pruned_data):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.load_data = LoadData()
        self.use_pruned_data = use_pruned_data

    def get_dr_matrices(self, labels_path, base_data_path, pca_model_path, tsne_model_path, sae_model_path, num_rows):
        """
        Takes the dimensionality reduction techniques model_path's, loads the matrices.
        Returns the matrices as a dict.
        :param labels_path:
        :param base_data_path:
        :param pca_model_path:
        :param tsne_model_path:
        :param sae_model_path:
        :param num_rows:
        :return:
        """
        dr_matrices = dict()

        input_matrix, input_matrix_indices, labels = self.load_data.get_data_with_labels(num_rows=num_rows,
                                                                                         data_path=base_data_path,
                                                                                         labels_path=labels_path)
        del input_matrix_indices
        dr_matrices['base_data'] = input_matrix

        if self.use_pruned_data:
            pca_file_name = pca_model_path + "/" + "pca_reduced_pruned_matrix_" + str(num_rows) + ".npy"
        else:
            pca_file_name = pca_model_path + "/" + "pca_reduced_unpruned_matrix_" + str(num_rows) + ".npy"
        pca_reduced_matrix = np.load(pca_file_name)
        dr_matrices["pca"] = pca_reduced_matrix

        if self.use_pruned_data:
            tsne_random_file_name = glob.glob(tsne_model_path + "/" + "tsne_reduced_pruned_matrix_init_random_*")[0]
        else:
            tsne_random_file_name = glob.glob(tsne_model_path + "/" + "tsne_reduced_unpruned_matrix_init_random_*")[0]
        tsne_random_reduced_matrix = np.load(tsne_random_file_name)['arr']
        dr_matrices["tsne_random"] = tsne_random_reduced_matrix

        if self.use_pruned_data:
            tsne_pca_file_name = glob.glob(tsne_model_path + "/" + "tsne_reduced_pruned_matrix_init_pca_*")[0]
        else:
            tsne_pca_file_name = glob.glob(tsne_model_path + "/" + "tsne_reduced_unpruned_matrix_init_pca_*")[0]
        tsne_pca_reduced_matrix = np.load(tsne_pca_file_name)['arr']
        dr_matrices["tsne_pca"] = tsne_pca_reduced_matrix

        if self.use_pruned_data:
            sae_file_name = sae_model_path + "/" + "sae_reduced_pruned_matrix_" + str(num_rows) + ".npz"
        else:
            sae_file_name = sae_model_path + "/" + "sae_reduced_unpruned_matrix_" + str(num_rows) + ".npz"
        sae_reduced_matrix = np.load(sae_file_name)['arr_0']
        dr_matrices['sae'] = sae_reduced_matrix

        return dr_matrices, labels
