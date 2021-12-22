
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

    if args.verbose:
        print("\nAll patients:")
        for patient in data[0]:
            print(list(patient))

    probabilities = get_predictions(model, data, model_parameters, args)

    if args.verbose:
        print("\nAll target predictions:")
        for probability in probabilities:
            print(probability >= 0.5)

    args.batch_size = 1
    data_generator = SequenceBuilder(data, model_parameters, args)

    # unpickle information about variables, medications, and interactions
    with open(args.path_meds_all, 'rb') as readfile:
        meds_all = pickle.load(readfile)
    n_meds = len(meds_all)

    # initialize lists of importances
    med_importance_ratios_pairwise_all = [[[] for _ in range(n_meds)] for _ in range(n_meds)]
    med_importance_ratios_pairwise_surviving = [[[] for _ in range(n_meds)] for _ in range(n_meds)]
    med_importance_ratios_pairwise_mortality = [[[] for _ in range(n_meds)] for _ in range(n_meds)]

    # iterate over patients and collect medication importances
    print("")
    for patient_id in range(len(data[0])):
        print(f"{patient_id} out of {len(data[0])} patients processed") if patient_id % 1000 == 0 else None
        mortality_prob = probabilities[patient_id]
        patient_data = data_generator.__getitem__(patient_id)

        if args.verbose:
            print("")
            print("=" * 80)
            print(f"Patient {patient_id}: ")
            print("=" * 80)
            print("\nPatient history: ")
            try:
                assert len(patient_data) == 1
            except AssertionError:
                raise Exception("patient_data has a length other than 1. This doesn't make sense.")
            print(patient_data[0])
            print("\nMortality probability: ", mortality_prob)

        proba, alphas, betas = model_with_attention.predict_on_batch(patient_data)
        visits = get_importances(alphas[0], betas[0], patient_data, model_parameters, dictionary)

        visits_meds_and_importances = []  # L[L[T[int, float]]]
        for visit_index, visit in enumerate(visits):

            if args.verbose:
                print(f"\nVisit number {visit_index}: ")

            visit_reset = visit.reset_index(drop=True)

            if args.verbose:
                print(visit_reset)

            features = visit_reset['feature'].tolist()
            importances = visit_reset[f'importance_{"visit" if args.use_visit_importance else "feature"}'].tolist()
            visit_meds_and_importances = []  # L[T[int, float]]
            for feature, importance in zip(features, importances):
                for med_index in range(n_meds):
                    if f"m_{med_index} " in feature:
                        visit_meds_and_importances.append((med_index, importance))
            visits_meds_and_importances.append(visit_meds_and_importances)
        med_importances_collected = [[] for _ in range(n_meds)]  # L[L[float]]
        for visit_meds_and_importances in visits_meds_and_importances:
            for med_index, importance in visit_meds_and_importances:
                med_importances_collected[med_index].append(importance)

        if args.verbose:
            print("")
            for med_index, importances in enumerate(med_importances_collected):
                print(f"Collected importances for m_{med_index}: {importances}")

        med_importances_average = [
            (np.mean(importances) if len(importances) != 0 else np.nan)
            for importances
            in med_importances_collected]  # L[float]

        if args.verbose:
            print("")
            for med_index, avg_importance in enumerate(med_importances_average):
                print(f"Average importances for m_{med_index}: {avg_importance}")

        med_importances_pairwise_collected = [[[] for _ in range(n_meds)] for _ in range(n_meds)]  # L[L[L[float]]]
        med_importances_complement_collected = [[[] for _ in range(n_meds)] for _ in range(n_meds)]  # L[L[L[float]]]
        for visit_meds_and_importances in visits_meds_and_importances:
            visit_meds = set()
            for med_index, _ in visit_meds_and_importances:
                visit_meds.add(med_index)
            for med_index_i, importance_i in visit_meds_and_importances:
                for med_index_j in range(n_meds):
                    if med_index_j in visit_meds:
                        med_importances_pairwise_collected[med_index_i][med_index_j].append(importance_i)
                    else:
                        med_importances_complement_collected[med_index_i][med_index_j].append(importance_i)

        if args.verbose:
            print("")
            for med_index_i in range(n_meds):
                for med_index_j in range(n_meds):
                    print(f"Collected importances for m_{med_index_i} | m_{med_index_j}: "
                          f"{med_importances_pairwise_collected[med_index_i][med_index_j]}")
                    print(f"Collected importances for m_{med_index_i} | !m_{med_index_j}: "
                          f"{med_importances_complement_collected[med_index_i][med_index_j]}")

        med_importances_pairwise_average = [
            [
                (np.mean(importances) if len(importances) != 0 else np.nan)
                for importances
                in row
            ]
            for row
            in med_importances_pairwise_collected  # L[L[float]]
        ]
        med_importances_complement_average = [
            [
                (np.mean(importances) if len(importances) != 0 else np.nan)
                for importances
                in row
            ]
            for row
            in med_importances_complement_collected  # L[L[float]]
        ]

        if args.verbose:
            print("")
            for med_index_i in range(n_meds):
                for med_index_j in range(n_meds):
                    print(f"Average importances for m_{med_index_i} | m_{med_index_j}: "
                          f"{med_importances_pairwise_average[med_index_i][med_index_j]}")
                    print(f"Average importances for m_{med_index_i} | !m_{med_index_j}: "
                          f"{med_importances_complement_average[med_index_i][med_index_j]}")

        if args.denominator_complement:
            np_med_importances_pairwise_ratios = (
                    np.array(med_importances_pairwise_average) / np.array(med_importances_complement_average))
        else:
            np_med_importances_pairwise_ratios = (
                    np.array(med_importances_pairwise_average) / np.array(med_importances_average).reshape((n_meds, 1)))
        for med_index_i in range(n_meds):
            for med_index_j in range(n_meds):
                val = np_med_importances_pairwise_ratios[med_index_i, med_index_j]
                if np.isfinite(val):
                    med_importance_ratios_pairwise_all[med_index_i][med_index_j].append(val)
                    if mortality_prob >= 0.5:
                        med_importance_ratios_pairwise_mortality[med_index_i][med_index_j].append(val)
                    else:
                        med_importance_ratios_pairwise_surviving[med_index_i][med_index_j].append(val)

        if args.verbose:
            print("")
            for med_index_i in range(n_meds):
                for med_index_j in range(n_meds):
                    print(f"Importance ratio for m_{med_index_i} | m_{med_index_j}: "
                          f"{np_med_importances_pairwise_ratios[med_index_i][med_index_j]}")

    if args.verbose:
        print("")
        print("=" * 80)
        print("Aggregate analysis: ")
        print("=" * 80)
        print("")
        for med_index_i in range(n_meds):
            for med_index_j in range(n_meds):
                print(f"Collected importance ratios for m_{med_index_i} | m_{med_index_j}: "
                      f"{med_importance_ratios_pairwise_all[med_index_i][med_index_j]}")
        print("")
        for med_index_i in range(n_meds):
            for med_index_j in range(n_meds):
                print(f"Collected importance ratios for m_{med_index_i} | m_{med_index_j}, mort: "
                      f"{med_importance_ratios_pairwise_mortality[med_index_i][med_index_j]}")
        print("")
        for med_index_i in range(n_meds):
            for med_index_j in range(n_meds):
                print(f"Collected importance ratios for m_{med_index_i} | m_{med_index_j}, surv: "
                      f"{med_importance_ratios_pairwise_surviving[med_index_i][med_index_j]}")

    # take mean ratio for each pair of medications
    med_importance_ratios_pairwise_mean_all = np.empty((n_meds, n_meds))
    med_importance_ratios_pairwise_mean_mortality = np.empty((n_meds, n_meds))
    med_importance_ratios_pairwise_mean_surviving = np.empty((n_meds, n_meds))
    agg_func = np.median if args.use_median else np.mean
    for med_index_i in range(n_meds):
        for med_index_j in range(n_meds):
            if med_index_i == med_index_j:
                med_importance_ratios_pairwise_mean_all[med_index_i, med_index_j] = np.nan
                med_importance_ratios_pairwise_mean_mortality[med_index_i, med_index_j] = np.nan
                med_importance_ratios_pairwise_mean_surviving[med_index_i, med_index_j] = np.nan
            else:
                med_importance_ratios_pairwise_mean_all[med_index_i, med_index_j] = agg_func(
                    med_importance_ratios_pairwise_all[med_index_i][med_index_j])
                med_importance_ratios_pairwise_mean_mortality[med_index_i, med_index_j] = agg_func(
                    med_importance_ratios_pairwise_mortality[med_index_i][med_index_j])
                med_importance_ratios_pairwise_mean_surviving[med_index_i, med_index_j] = agg_func(
                    med_importance_ratios_pairwise_surviving[med_index_i][med_index_j])

    # print results
    print("\nAll patients: ")
    print("\t\t\t\t", end="")
    for i in range(n_meds):
        print(f"m_{i}({'+' if meds_all[i].is_beneficial else '-'})\t", end="")
    print("")
    for i in range(n_meds):
        print(f"m_{i}({'+' if meds_all[i].is_beneficial else '-'})\t", end="")
        print(f"{None}\t", end="")
        for j in range(n_meds):
            print(f"{med_importance_ratios_pairwise_mean_all[i, j]:1.3f}\t", end="")
        print("")
    print("\nMortality patients: ")
    print("\t\t\t\t", end="")
    for i in range(n_meds):
        print(f"m_{i}({'+' if meds_all[i].is_beneficial else '-'})\t", end="")
    print("")
    for i in range(n_meds):
        print(f"m_{i}({'+' if meds_all[i].is_beneficial else '-'})\t", end="")
        print(f"{None}\t", end="")
        for j in range(n_meds):
            print(f"{med_importance_ratios_pairwise_mean_mortality[i, j]:1.3f}\t", end="")
        print("")
    print("\nSurviving patients: ")
    print("\t\t\t\t", end="")
    for i in range(n_meds):
        print(f"m_{i}({'+' if meds_all[i].is_beneficial else '-'})\t", end="")
    print("")
    for i in range(n_meds):
        print(f"m_{i}({'+' if meds_all[i].is_beneficial else '-'})\t", end="")
        print(f"{None}\t", end="")
        for j in range(n_meds):
            print(f"{med_importance_ratios_pairwise_mean_surviving[i, j]:1.3f}\t", end="")
        print("")


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_model', type=str, help='Path to the model to evaluate')
    parser.add_argument('--path_data', type=str, help='Path to evaluation data')
    parser.add_argument('--path_dictionary', type=str, help='Path to codes dictionary')
    parser.add_argument('--path_meds_all', type=str, help='Path to medications list')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for initial probability predictions')
    parser.add_argument('--denominator_complement', action='store_true',
                        help='Whether to use the complement as the denominator.')
    parser.add_argument('--use_visit_importance', action='store_true', help='Whether to use the visit importances')
    parser.add_argument('--use_median', action='store_true', help='Whether to use the median instead of mean')
    parser.add_argument('--verbose', action='store_true', help='Verbosity')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
