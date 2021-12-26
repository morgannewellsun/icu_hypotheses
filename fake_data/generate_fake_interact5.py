
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main(
        out_directory,
        n_patients,
        n_timesteps,
        period,
        train_proportion,
        val_proportion,
        verbose):

    patients = []
    patient_morts = []
    for _ in range(n_patients):
        phase = np.random.randint(period)
        visits = []
        for _ in range(n_timesteps):
            if phase == 0:
            # if phase > period // 2:
                visits.append([1])
            else:
                visits.append([0])
            phase = (phase + 1) % period
        if verbose >= 2:
            print(np.array(visits).flatten())
        patients.append(visits)
        patient_morts.append(np.random.binomial(1, 0.5))

    # for RNN
    all_data = pd.DataFrame(data={'codes': patients}, columns=['codes']).reset_index()
    all_targets = pd.DataFrame(data={'target': patient_morts}, columns=['target']).reset_index()
    data_train, data_val_test = train_test_split(all_data, train_size=train_proportion, random_state=12345)
    target_train, target_val_test = train_test_split(all_targets, train_size=train_proportion, random_state=12345)
    val_proportion_adjusted = val_proportion / (1 - train_proportion)
    data_val, data_test = train_test_split(data_val_test, train_size=val_proportion_adjusted, random_state=12345)
    target_val, target_test = train_test_split(target_val_test, train_size=val_proportion_adjusted, random_state=12345)
    data_train.sort_index().to_pickle(out_directory + '/patients_train.pkl')
    data_val.sort_index().to_pickle(out_directory + '/patients_val.pkl')
    data_test.sort_index().to_pickle(out_directory + '/patients_test.pkl')
    target_train.sort_index().to_pickle(out_directory + '/outcomes_train.pkl')
    target_val.sort_index().to_pickle(out_directory + '/outcomes_val.pkl')
    target_test.sort_index().to_pickle(out_directory + '/outcomes_test.pkl')

    # pickled dictionary of string lookup table for medical codes
    dictionary = {0: "medical_code_zero", 1: "medical_code_one"}
    with open(out_directory + '/dictionary.pkl', "wb") as writefile:
        pickle.dump(dictionary, writefile)


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out_directory', type=str, required=True)
    parser.add_argument('--n_patients', type=int, default=40)
    parser.add_argument('--n_timesteps', type=int, default=40)
    parser.add_argument('--period', type=int, default=10)
    parser.add_argument('--train_proportion', type=float, default=0.8)
    parser.add_argument('--val_proportion', type=float, default=0.1)
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(
        args.out_directory,
        args.n_patients,
        args.n_timesteps,
        args.period,
        args.train_proportion,
        args.val_proportion,
        args.verbose)
