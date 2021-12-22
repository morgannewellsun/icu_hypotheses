
import argparse
import pickle

import keras.backend as keras_backend
from keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf

from fake_data.generate_fake_interact4 import MedSpec
from generator.code_generator_modified import read_data, process


def import_model(path):
    """Import model from given path and assign it to appropriate devices"""
    keras_backend.clear_session()
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    tfsess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(tfsess)
    model = load_model(path)
    return model


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def main(args):

    n_codes = args.n_codes
    maxlen = args.maxlen
    probe_length = args.probe_length
    n_probes = args.n_probes
    n_replacements = args.n_replacements
    n_predictions = args.n_predictions
    temperature = args.temperature

    with open(args.path_meds_all, 'rb') as f:
        meds_all = pickle.load(f)
    n_meds = len(meds_all)
    med_index_to_codes = dict()
    for med_index, med_spec in enumerate(meds_all):
        # increment all codes by 1 since code 0 is padding (same done during LSTM training)
        med_index_to_codes.update({med_index: [code + 1 for code in med_spec.codes]})

    # codes for mortality and release
    mortality_code = n_codes + 1
    release_code = n_codes + 2

    print("Medication index to codes: ")
    for key, value in med_index_to_codes.items():
        print(f"{key}: {value}")

    # load model
    model = import_model(args.path_model)

    # define ordered pairs of meds to run virtual experiments on
    eval_grid = np.ones((n_meds, n_meds), dtype=bool)

    # determine which meds need to be run on their own
    meds_to_eval = np.where(np.any([eval_grid.any(axis=0), eval_grid.any(axis=1)], axis=0))[0]


    # =========================================================================
    # This code section manually generates probe data populations.
    
    # generate probe data populations
    probe_data_dict_single = dict()
    for med_index in meds_to_eval:
        probe_data_list_single = []
        for _ in range(n_probes):
            sequence = np.zeros((1, maxlen))
            sequence[0, -probe_length:] = np.random.choice(med_index_to_codes[med_index], probe_length)
            probe_data_list_single.append(sequence.astype(int))
        probe_data_dict_single.update({med_index: probe_data_list_single})

    # generate interaction variations of probe data populations
    # these variations replace n_replacements of the original codes with new codes
    probe_data_dict_pairs = dict()
    for med_index_i in range(n_meds):
        probe_data_list_single = probe_data_dict_single[med_index_i]
        for med_index_j in range(n_meds):
            if med_index_i == med_index_j:
                continue
            if not eval_grid[med_index_i, med_index_j]:
                continue
            probe_data_list_pair = []
            for probe_index in range(n_probes):
                new_sequence = probe_data_list_single[probe_index].copy()
                # print("OLD", new_sequence)
                random_indexer = np.concatenate([
                    [False for _ in range(probe_length - n_replacements)],
                    [True for _ in range(n_replacements)]])
                np.random.shuffle(random_indexer)
                random_indexer = np.concatenate(
                    [np.zeros(maxlen - probe_length, dtype=bool), random_indexer]).reshape((1, -1))
                random_replacements = np.random.choice(med_index_to_codes[med_index_j], n_replacements)
                new_sequence[random_indexer] = random_replacements
                probe_data_list_pair.append(new_sequence.astype(int))
            probe_data_dict_pairs.update({(med_index_i, med_index_j): probe_data_list_pair})
    '''

    # =========================================================================
    
    # This code section truncates and sorts the test data to generate probe snippets.
    # This code doesn't work.

    probe_data_dict_single = dict()
    probe_data_dict_pairs = dict()
    for med_index_i in range(n_meds):
        probe_data_dict_single.update({med_index_i: []})
        for med_index_j in range(n_meds):
            if med_index_i == med_index_j:
                continue
            probe_data_dict_pairs.update({(med_index_i, med_index_j): []})
    x_test, y_test = read_data(path_data=args.path_data_test, path_target=args.path_target_test)
    x_test, y_test = process(x_test, y_test, num_codes=n_codes, maxlen=maxlen)
    x_test_trunc = x_test[:, :probe_length]
    y_test_trunc = y_test[:, :probe_length]
    for x_snippet, y_snippet in zip(x_test_trunc, y_test_trunc):
        meds_in_snippet = set()
        for med_index in range(n_meds):
            med_codes_to_check = med_index_to_codes[med_index]
            for med_code_to_check in med_codes_to_check:
                if med_code_to_check in x_snippet:
                    meds_in_snippet.add(med_index)
                    break
        if len(meds_in_snippet) == 1:
            probe_data_dict_single[meds_in_snippet.pop()].append(x_snippet)
        elif len(meds_in_snippet) == 2:
            med_index_i = meds_in_snippet.pop()
            med_index_j = meds_in_snippet.pop()
            probe_data_dict_pairs[(med_index_i, med_index_j)].append(x_snippet)
            probe_data_dict_pairs[(med_index_j, med_index_i)].append(x_snippet)
            
    '''

    # =========================================================================

    # run virtual experiments for all single probe sequences
    single_results = dict()
    for med_index in meds_to_eval:
        probe_data_list_single = probe_data_dict_single[med_index]
        morts = []
        for sequence in probe_data_list_single:
            patient_mortality = False
            for t in range(n_predictions):
                preds = model.predict(sequence, verbose=0)[0]
                next_code = sample(preds, temperature)
                if next_code == mortality_code:
                    patient_mortality = True
                    # print(f"Patient mortality at t={t+probe_length}")
                    break
                elif next_code == release_code:
                    # print(f"Patient released at t={t+probe_length}")
                    break
                else:
                    sequence = np.roll(sequence, shift=-1, axis=1)
                    sequence[-1] = next_code
            morts.append(patient_mortality)
        mortality_rate = np.mean(morts)
        single_results.update({med_index: mortality_rate})
        print(f"Medication {med_index} mortality rate: {mortality_rate}")

    # run virtual experiments for all interaction probe sequences
    pair_results = dict()
    for med_index_i in range(n_meds):
        for med_index_j in range(n_meds):
            if med_index_i == med_index_j:
                continue
            if not eval_grid[med_index_i, med_index_j]:
                continue
            probe_data_list_pair = probe_data_dict_pairs[(med_index_i, med_index_j)]
            morts = []
            for sequence in probe_data_list_pair:
                patient_mortality = False
                for _ in range(n_predictions):
                    preds = model.predict(sequence, verbose=0)[0]
                    next_code = sample(preds, temperature)
                    if next_code == mortality_code:
                        patient_mortality = True
                        break
                    elif next_code == release_code:
                        break
                    else:
                        sequence = np.roll(sequence, shift=-1, axis=1)
                        sequence[-1] = next_code
                morts.append(patient_mortality)
            mortality_rate = np.mean(morts)
            pair_results.update({(med_index_i, med_index_j): mortality_rate})
            print(f"Interaction {med_index_i} and {med_index_j} mortality rate: {mortality_rate}")

    # construct table of mortality rates
    results_table_np = np.full(shape=(n_meds, n_meds+1), fill_value=np.nan, dtype=float)
    for med_index in meds_to_eval:
        results_table_np[med_index, 0] = single_results[med_index]
    for med_index_i, med_index_j in zip(*np.where(eval_grid)):
        if med_index_i == med_index_j:
            continue
        results_table_np[med_index_i, med_index_j + 1] = pair_results[(med_index_i, med_index_j)]
    column_labels = ['single'] + [f'int_{med_index}' for med_index in range(n_meds)]
    results_table_pd = pd.DataFrame(data=results_table_np, columns=column_labels)
    print(results_table_pd)
    results_table_np[:, 1:] /= results_table_np[:, 0:1]
    results_table_pd = pd.DataFrame(data=results_table_np, columns=column_labels)
    print(results_table_pd)


def parse_arguments():
    """Read user arguments"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--path_meds_all', type=str, help='path to medications list')
    parser.add_argument('--path_model', type=str, help='path to the trained LSTM model')
    parser.add_argument('--path_data_test', type=str, help='path to the test data')
    parser.add_argument('--path_target_test', type=str, help='path to the test targets')
    parser.add_argument('--n_codes', type=int, help='total number of medical codes')
    parser.add_argument('--maxlen', type=int, default=3, help='maxlen used when training LSTM')
    parser.add_argument('--probe_length', type=int, default=3, help='length of probe sequences')
    parser.add_argument('--n_probes', type=int, default=100, help='number of probe sequences to generate per med')
    parser.add_argument('--n_replacements', type=int, default=1, help='number of replacements for interaction probes')
    parser.add_argument('--n_predictions', type=int, default=10, help='max number of LSTM predictions per probe')
    parser.add_argument('--temperature', type=float, default=0.8, help='temperature for sampling predictions')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
