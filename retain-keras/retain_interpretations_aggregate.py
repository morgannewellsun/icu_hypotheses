
import argparse
import pickle as pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.models import load_model, Model
from keras.preprocessing import sequence
from keras.constraints import Constraint
from keras.utils.data_utils import Sequence

from retain_interpretations import *
from fake_data.generate_fake_interact4 import MedSpec


def main(args):

    # model loading and setup for interpretation
    model, model_with_attention = import_model(args.path_model)
    model_parameters = get_model_parameters(model)
    data, dictionary = read_data(model_parameters, args.path_data, args.path_dictionary)
    probabilities = get_predictions(model, data, model_parameters, args)
    args.batch_size = 1
    data_generator = SequenceBuilder(data, model_parameters, args)

    # unpickle information about variables, medications, and interactions
    with open(args.path_meds_all, 'rb') as readfile:
        meds_all = pickle.load(readfile)
    n_meds = len(meds_all)

    # initialize lists of importances
    med_importances_pairwise_all = [[[] for _ in range(n_meds)] for _ in range(n_meds)]
    med_importances_pairwise_surviving = [[[] for _ in range(n_meds)] for _ in range(n_meds)]
    med_importances_pairwise_mortality = [[[] for _ in range(n_meds)] for _ in range(n_meds)]

    # iterate over patients and collect medication importances
    for patient_id in range(len(data[0])):
        print(f"{patient_id} out of {len(data[0])} patients processed") if patient_id % 1000 == 0 else None
        mortality_prob = probabilities[patient_id]
        patient_data = data_generator.__getitem__(patient_id)
        proba, alphas, betas = model_with_attention.predict_on_batch(patient_data)
        visits = get_importances(alphas[0], betas[0], patient_data, model_parameters, dictionary)
        for visit in visits:
            visit_reset = visit.reset_index(drop=True)
            features = visit_reset['feature'].tolist()
            importances = visit_reset['importance_feature'].tolist()
            visit_meds_and_importances = []
            for feature, importance in zip(features, importances):
                for med_index in range(n_meds):
                    if f"m_{med_index}" in feature:
                        visit_meds_and_importances.append((med_index, importance))
            for med_index_i, importance_i in visit_meds_and_importances:
                for med_index_j, _ in visit_meds_and_importances:
                    med_importances_pairwise_all[med_index_i][med_index_j].append(importance_i)
                    if mortality_prob > 0.5:
                        med_importances_pairwise_mortality[med_index_i][med_index_j].append(importance_i)
                    else:
                        med_importances_pairwise_surviving[med_index_i][med_index_j].append(importance_i)

    # take means of each cell in pairwise importance matrices
    med_importances_pairwise_mean_all = np.empty((n_meds, n_meds))
    med_importances_pairwise_mean_mortality = np.empty((n_meds, n_meds))
    med_importances_pairwise_mean_surviving = np.empty((n_meds, n_meds))
    for med_index_i in range(n_meds):
        for med_index_j in range(n_meds):
            med_importances_pairwise_mean_all[med_index_i][med_index_j] = np.mean(
                med_importances_pairwise_all[med_index_i][med_index_j])
            med_importances_pairwise_mean_mortality[med_index_i][med_index_j] = np.mean(
                med_importances_pairwise_mortality[med_index_i][med_index_j])
            med_importances_pairwise_mean_surviving[med_index_i][med_index_j] = np.mean(
                med_importances_pairwise_surviving[med_index_i][med_index_j])

    # extract diagonal elements from pairwise importance matrices
    # diagonal elements represent independent importances of individual meds
    med_importances_individual_mean_all = np.diagonal(med_importances_pairwise_mean_all)
    med_importances_individual_mean_mortality = np.diagonal(med_importances_pairwise_mean_mortality)
    med_importances_individual_mean_surviving = np.diagonal(med_importances_pairwise_mean_surviving)

    # calculate ratio between individual and pairwise importances
    # this represents the impact of interactions between other medications
    med_importances_pairwise_ratio_all = (
            med_importances_pairwise_mean_all / med_importances_individual_mean_all.reshape((n_meds, 1)))
    med_importances_pairwise_ratio_mortality = (
            med_importances_pairwise_mean_mortality / med_importances_individual_mean_mortality.reshape((n_meds, 1)))
    med_importances_pairwise_ratio_surviving = (
            med_importances_pairwise_mean_surviving / med_importances_individual_mean_surviving.reshape((n_meds, 1)))

    # print results
    print("\nAll patients: ")
    print("\t\t\t\t", end="")
    for i in range(n_meds):
        print(f"m_{i}({'+' if meds_all[i].is_beneficial else '-'})\t", end="")
    print("")
    for i in range(n_meds):
        print(f"m_{i}({'+' if meds_all[i].is_beneficial else '-'})\t", end="")
        print(f"{med_importances_individual_mean_all[i]:1.3f}\t", end="")
        for j in range(n_meds):
            print(f"{med_importances_pairwise_ratio_all[i, j]:1.3f}\t", end="")
        print("")
    print("\nMortality patients: ")
    print("\t\t\t\t", end="")
    for i in range(n_meds):
        print(f"m_{i}({'+' if meds_all[i].is_beneficial else '-'})\t", end="")
    print("")
    for i in range(n_meds):
        print(f"m_{i}({'+' if meds_all[i].is_beneficial else '-'})\t", end="")
        print(f"{med_importances_individual_mean_mortality[i]:1.3f}\t", end="")
        for j in range(n_meds):
            print(f"{med_importances_pairwise_ratio_mortality[i, j]:1.3f}\t", end="")
        print("")
    print("\nSurviving patients: ")
    print("\t\t\t\t", end="")
    for i in range(n_meds):
        print(f"m_{i}({'+' if meds_all[i].is_beneficial else '-'})\t", end="")
    print("")
    for i in range(n_meds):
        print(f"m_{i}({'+' if meds_all[i].is_beneficial else '-'})\t", end="")
        print(f"{med_importances_individual_mean_surviving[i]:1.3f}\t", end="")
        for j in range(n_meds):
            print(f"{med_importances_pairwise_ratio_surviving[i, j]:1.3f}\t", end="")
        print("")


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_model', type=str, help='Path to the model to evaluate')
    parser.add_argument('--path_data', type=str, help='Path to evaluation data')
    parser.add_argument('--path_dictionary', type=str, help='Path to codes dictionary')
    parser.add_argument('--path_meds_all', type=str, help='Path to medications list')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for initial probability predictions')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)