import logging
from multiprocessing import Pool, get_logger
from time import time

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from HelperFunctions.HelperFunction import HelperFunction
from HelperFunctions.Plotting import Plotting
from PrepareData.LoadData import LoadData
from Utils.ConfigUtil import ConfigUtil

config = ConfigUtil().get_config_instance()
helper = HelperFunction()
plot = Plotting()
load_data = LoadData()


def create_logger():
    logger = get_logger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("logs/multi_process.log")
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def plot_matrix(reduced_matrix, plot_path, init, perplexity, learning_rate):
    plt = plot.plot_it_2d(reduced_matrix)
    plt.savefig(plot_path + "/" + "tsne_2d_" + str(init) + "_" + str(perplexity) + "_" + str(learning_rate) + ".png")
    plt.close()
    plt = plot.plot_it_3d(reduced_matrix)
    plt.savefig(plot_path + "/" + "tsne_3d_" + str(init) + "_" + str(perplexity) + "_" + str(learning_rate) + ".png")
    plt.close()


def tsne_model(args):
    """
    The arguments must be received in the same order as mentioned here.
    n_components, perplexity, learning_rate, init, input_matrix, plot_path
    :param args: n_components, perplexity, learning_rate, init, input_matrix, plot_path
    :return: dictionary which contains the model object, reduced representation of matrix and the parameters used.
    """
    n_components, perplexity, learning_rate, init, input_matrix, plot_path = args
    model = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate,
                 init=init)
    reduced_matrix = model.fit_transform(input_matrix.toarray())
    plot_matrix(reduced_matrix, plot_path, init, perplexity, learning_rate)
    params = {"init": init, "perplexity": perplexity,
              "learning_rate": learning_rate,
              "kl_divergence": model.kl_divergence_}
    log.info("Saving 2d & 3d plots")
    log.info("Model Params : {}".format(params))
    return {"reduced_matrix": reduced_matrix, "params": params}


def perform_tsne(n_components, plot_path, input_matrix):
    perplexity_list = range(5, 55, 5)
    learning_rate_list = range(10, 1100, 100)
    init_list = ["random", "pca"]

    final_result = list()

    for init in init_list:
        temp = list()
        for perplexity in perplexity_list:
            log.info("Number of threads used are : {}".format(len(learning_rate_list)))
            pool = Pool(processes=len(learning_rate_list))
            params = [[n_components, perplexity, place_holder, init, input_matrix, plot_path] for place_holder in
                      learning_rate_list]
            p_place_holder = pool.map_async(tsne_model, params)
            result = p_place_holder.get()
            pool.close()
            pool.join()
            temp.append(result)
        final_result.append(temp)
    return final_result


def get_best_params(tsne_model_list):
    log.info("Finding the best params for least error")
    tsne_reduced_matrix_dict = dict()

    for init in tsne_model_list:
        main_list = list()
        for perplexity in init:
            for learning_rate in perplexity:
                main_list.append(learning_rate)
        df = pd.DataFrame(main_list)
        reduced_matrix = df.loc[df['kl_divergence'].idxmin()]['reduced_matrix']
        tsne_reduced_matrix_dict[init] = reduced_matrix

    return tsne_reduced_matrix_dict


def main(num_rows):
    start_time = time()
    plot_path = config['plots']['tsne']
    tsne_model_path = config['models']['tsne']
    tsne_results_path = config['results']['iterations']['tsne']

    all_params_dir = tsne_model_path + "/" + "all_params"
    helper.create_dir_if_absent(all_params_dir)

    n_components = 3
    input_matrix, input_matrix_indices = load_data.main(num_rows=num_rows)
    tsne_model_list = perform_tsne(n_components=n_components,
                                   plot_path=plot_path,
                                   input_matrix=input_matrix)

    log.info("Saving the TSNE model & Reduced Matrix at : {}".format(tsne_model_path))

    tsne_model_fname = tsne_model_path + "/all_params/" + "tsne_model_" + str(num_rows)
    np.savez_compressed(file=tsne_model_fname, arr=tsne_model_list)

    tsne_reduced_matrix_dict = get_best_params(tsne_model_list)

    tsne_reduced_matrix_fname = tsne_results_path + "/" + "tsne_reduced_matrix_all_inits_" + str(num_rows)
    np.savez_compressed(file=tsne_reduced_matrix_fname, arr=tsne_reduced_matrix_dict)

    # TODO
    # Add clustering code.

    log.info("Total time taken : {}".format(time() - start_time))


log = create_logger()
main(num_rows=25000)
