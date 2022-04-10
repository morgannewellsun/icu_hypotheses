
import argparse
import os
import time

import keras as k
import numpy as np
import pandas as pd
import tensorflow as tf


def main(preprocessed_mimic_filepath, lstm_model_filepath):

    # =========================================================================
    # Hyperparameters
    # =========================================================================

    # dataset filtering - these parameters MUST be identical across all scripts
    use_truncated_codes = True
    proportion_event_instances = 0.9  # {0.5, 0.8, 0.9, 0.95, 0.99}
    admissions_per_patient_incl_min = 1
    medications_per_patient_incl_min = 50  # patients with less will be excluded entirely
    medications_per_patient_incl_max = 100  # patients with more or equal will have early medications truncated

    # data processing
    use_separator_token = True  # MUST be true for output to be usable as RETAIN input!
    truncate_whole_visits = False
    probe_length = 10  # MUST be identical to probe_length used in run_virtual_experiments.py
    train_val_test_splits = (0.8, 0.1, 0.1)  # MUST be identical across all scripts

    # model architecture and training
    input_length = 100  # MUST be identical to input_length used in train_lstm_on_mimic.py
    batch_size = 128

    # evaluation parameters
    top_k_values = [10, 20, 30]  # evaluate top-k accuracy

    # checks and derived hyperparameters
    event_code_col_name = "event_code_trunc" if use_truncated_codes else "event_code_full"
    event_code_count_col_name = "event_code_trunc_count" if use_truncated_codes else "event_code_full_count"
    assert proportion_event_instances in {0.5, 0.8, 0.9, 0.95, 0.99}
    selector_dict = {
        (True, 0.5): 8993,
        (True, 0.8): 2107,
        (True, 0.9): 825,
        (True, 0.95): 376,
        (True, 0.99): 65,
        (False, 0.5): 8474,
        (False, 0.8): 1931,
        (False, 0.9): 780,
        (False, 0.95): 350,
        (False, 0.99): 59}  # values derived from cumulative_event_count_medications plots
    event_code_count_incl_min = selector_dict[(use_truncated_codes, proportion_event_instances)]
    assert abs(1 - sum(train_val_test_splits)) < 0.00001

    # =========================================================================
    # Data loading and standardized filtering
    # =========================================================================

    print("[INFO] Loading and preparing data")

    # load medication data
    mimic_df = pd.read_csv(preprocessed_mimic_filepath)
    mimic_df = mimic_df[mimic_df["event_type"] == "M"]

    # keep only most common medications
    mimic_df = mimic_df[mimic_df[event_code_count_col_name] >= event_code_count_incl_min]

    # keep only patients with enough admissions
    mimic_df = mimic_df[mimic_df["patient_admission_count"] >= admissions_per_patient_incl_min]

    # keep only patients with enough medications
    mimic_df = mimic_df[mimic_df["patient_medications_count"] >= medications_per_patient_incl_min]

    # truncate earliest medications for patients with too many medications
    truncated_patient_dfs = []
    for patient_id, patient_df in mimic_df.groupby("patient_id", sort=False):
        truncated_patient_dfs.append(patient_df.iloc[-1 * medications_per_patient_incl_max:])
    mimic_df = pd.concat(truncated_patient_dfs, axis=0)

    # =========================================================================
    # Further script-specific data preparation
    # =========================================================================

    # map string NDC codes to integer indices
    code_idx_to_str_map = ["PADDING", "OUTCOME_SURVIVAL", "OUTCOME_MORTALITY"]
    if use_separator_token:
        code_idx_to_str_map.append("SEPARATOR")
    code_idx_to_str_map.extend(list(mimic_df[event_code_col_name].value_counts().index))
    n_codes = len(code_idx_to_str_map)
    code_str_to_idx_map = dict([(code_str, code_idx) for code_idx, code_str in enumerate(code_idx_to_str_map)])
    mimic_df["event_code_idx"] = mimic_df[event_code_col_name].map(code_str_to_idx_map)

    # unpack dataframe into nested list: [patient_idx, admission_idx, event_idx]
    patients_nested = []
    patient_mortalities = []
    for patient_id, patient_df in mimic_df.groupby("patient_id", sort=False):
        patient_mortalities.append(patient_df["patient_mortality"].iloc[0])
        patient_admissions = []
        for admission_id, admission_df in patient_df.groupby("admission_id", sort=False):
            patient_admissions.append(list(admission_df["event_code_idx"]))
        patients_nested.append(patient_admissions)

    print(f"[INFO] Number of different medical codes: {n_codes}")
    print(f"[INFO] Number of patients: {len(patients_nested)}")
    print(f"[INFO] Average patient mortality rate: {np.mean(patient_mortalities)}")

    # truncate oldest visits from each patient so the history fits inside input_length+1
    # this block of code performs truncation in whole-visit chunks
    if truncate_whole_visits:
        patients_nested_truncated = []
        for patient_admissions in patients_nested:
            patient_admissions_truncated_reversed = []
            patient_num_tokens = 1  # all patients contain an outcome token
            for admission_events in patient_admissions[::-1]:
                patient_num_tokens += len(admission_events) + (1 if use_separator_token else 0)
                if patient_num_tokens <= input_length + 1:
                    patient_admissions_truncated_reversed.append(admission_events)
                else:
                    break
            patients_nested_truncated.append(patient_admissions_truncated_reversed[::-1])
    else:
        patients_nested_truncated = patients_nested

    # flatten each patient into a sequence: [patient_idx, event_idx]
    # include separator tokens between visits, as well as a token representing final outcome
    patients_seq = []
    for patient_admissions, patient_mortality in zip(patients_nested_truncated, patient_mortalities):
        patient_seq = []
        for admission_events in patient_admissions:
            patient_seq.extend(admission_events)
            if use_separator_token:
                patient_seq.append(code_str_to_idx_map["SEPARATOR"])
        patient_seq.append(
            code_str_to_idx_map["OUTCOME_MORTALITY"] if patient_mortality else code_str_to_idx_map["OUTCOME_SURVIVAL"])
        patients_seq.append(patient_seq)

    # truncate oldest visits from each patient so the history fits inside input_length+1
    # this block of code performs truncation down to the individual code
    if not truncate_whole_visits:
        patients_seq_truncated = []
        for patient_seq in patients_seq:
            patients_seq_truncated.append(patient_seq[-1 * (input_length + 1):])
        patients_seq = patients_seq_truncated

    # determine accuracy of a strategy where the most common code is always guessed
    patients_seq_flattened = []
    for patient_seq in patients_seq:
        patients_seq_flattened.extend(patient_seq[probe_length:])
    values, counts = np.unique(patients_seq_flattened, return_counts=True)
    naive_strategy_accuracy = np.max(counts) / len(patients_seq_flattened)
    print(f"[INFO] Accuracy of an 'always guess most common code' strategy: {naive_strategy_accuracy}")
    frequency_sorted_code_indices = values[np.argsort(counts)[::-1]]

    # pad flattened sequences to length input_length+1
    # the +1 is because x and y will be extracted from this array using [:-1] and [1:]
    patients_seq_np = np.full(
        shape=(len(patients_seq), input_length + 1),
        fill_value=code_str_to_idx_map["PADDING"],
        dtype=float)
    masks_np = np.zeros_like(patients_seq_np, dtype=float)
    for patient_idx, patient_seq in enumerate(patients_seq):
        patients_seq_np[patient_idx, :len(patient_seq)] = patient_seq
        masks_np[patient_idx, :len(patient_seq)] = 1

    # mask the first probe_length codes
    masks_np[:, :probe_length] = 0

    # extract x and y sequences of length input_length
    patients_seq_x_np = patients_seq_np[:, :-1]
    patients_seq_y_np = patients_seq_np[:, 1:]
    masks_y_np = masks_np[:, 1:]

    # split train/val/test
    n_train = int(len(patients_seq_x_np) * train_val_test_splits[0])
    n_val = int(len(patients_seq_x_np) * train_val_test_splits[1])
    n_test = int(len(patients_seq_x_np) - (n_train + n_val))
    rng = np.random.default_rng(seed=12345)
    train_val_test_indexer = np.array([0] * n_train + [1] * n_val + [2] * n_test)
    rng.shuffle(train_val_test_indexer)  # in-place
    # patients_seq_x_train_np = patients_seq_x_np[train_val_test_indexer == 0]
    # patients_seq_y_train_np = patients_seq_y_np[train_val_test_indexer == 0]
    # masks_y_train_np = masks_y_np[train_val_test_indexer == 0]
    patients_seq_x_val_np = patients_seq_x_np[train_val_test_indexer == 1]
    patients_seq_y_val_np = patients_seq_y_np[train_val_test_indexer == 1]
    masks_y_val_np = masks_y_np[train_val_test_indexer == 1]
    # patients_seq_x_test_np = patients_seq_x_np[train_val_test_indexer == 2]
    # patients_seq_y_test_np = patients_seq_y_np[train_val_test_indexer == 2]
    # masks_y_test_np = masks_y_np[train_val_test_indexer == 2]

    # =========================================================================
    # Evaluate top-k accuracy
    # =========================================================================

    print(f"[INFO] Evaluating LSTM top-k accuracy")

    # use LSTM to make predictions on validation set
    lstm_model = k.models.load_model(lstm_model_filepath, compile=False)
    val_predictions_np = lstm_model.predict(x=patients_seq_x_val_np, batch_size=batch_size)  # (?, input_length, n_codes)
    for top_k in top_k_values:
        val_predictions_top_k_indices_np = np.argsort(val_predictions_np, axis=2)[:, :, -1 * top_k:][:, :, ::-1]  # (?, input_length, k)
        top_k_matched = np.any(val_predictions_top_k_indices_np == patients_seq_y_val_np[:, :, np.newaxis], axis=2)  # (?, input_length)
        top_k_acc = np.sum(top_k_matched.astype(int) * masks_y_val_np) / np.sum(masks_y_val_np)
        print(f"[INFO] Top-{top_k} accuracy: {top_k_acc}")
        naive_top_k_matched = np.any(
            frequency_sorted_code_indices[np.newaxis, np.newaxis, :top_k] == patients_seq_y_val_np[:, :, np.newaxis],
            axis=2)
        naive_top_k_acc = np.sum(naive_top_k_matched.astype(int) * masks_y_val_np) / np.sum(masks_y_val_np)
        print(f"[INFO] Top-{top_k} accuracy of guess-most-common strategy: {naive_top_k_acc}")


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--preprocessed_mimic_filepath', type=str, required=True)
    parser.add_argument('--lstm_model_filepath', type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(
        args.preprocessed_mimic_filepath,
        args.lstm_model_filepath)
