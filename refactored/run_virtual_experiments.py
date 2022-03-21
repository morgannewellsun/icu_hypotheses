
import argparse

import keras as k
import numpy as np
import pandas as pd
import tensorflow as tf


def main(preprocessed_mimic_filepath, ndc_interactions_filepath, lstm_model_filepath, output_directory):

    # =========================================================================
    # Hyperparameters
    # =========================================================================

    # dataset filtering - these parameters MUST be identical across all scripts
    use_truncated_codes = True
    proportion_event_instances = 0.9  # {0.5, 0.8, 0.9, 0.95, 0.99}
    admissions_per_patient_incl_min = 1
    medications_per_patient_incl_min = 50  # patients with less will be excluded entirely
    medications_per_patient_incl_max = 100  # patients with more or equal will have early medications truncated

    # probe sequence parameters
    use_separator_token = True  # MUST be true for output to be usable as RETAIN input!
    train_val_test_splits = (0.8, 0.1, 0.1)  # MUST be identical across all scripts
    n_probes = 128
    probe_length = 10  # MUST be identical to probe_length used in train_lstm_on_mimic.py

    # lstm model parameters
    input_length = 100  # MUST be identical to input_length used in train_lstm_on_mimic.py
    batch_size = 128

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

    # # unpack dataframe into nested list: [patient_idx, admission_idx, event_idx]
    # patients_nested = []
    # patient_mortalities = []
    # for patient_id, patient_df in mimic_df.groupby("patient_id", sort=False):
    #     patient_mortalities.append(patient_df["patient_mortality"].iloc[0])
    #     patient_admissions = []
    #     for admission_id, admission_df in patient_df.groupby("admission_id", sort=False):
    #         patient_admissions.append(list(admission_df["event_code_idx"]))
    #     patients_nested.append(patient_admissions)
    #
    # # flatten each patient into a sequence: [patient_idx, event_idx]
    # # include separator tokens between visits
    # patients_seq = []
    # for patient_admissions, patient_mortality in zip(patients_nested, patient_mortalities):
    #     patient_seq = []
    #     for admission_events in patient_admissions:
    #         patient_seq.extend(admission_events)
    #         if use_separator_token:
    #             patient_seq.append(code_str_to_idx_map["SEPARATOR"])
    #     patients_seq.append(patient_seq)
    #
    # # split train/val/test
    # patients_seq_np_ragged = np.array(patients_seq)
    # patient_mortalities_np = np.array(patient_mortalities)
    # n_train = int(len(patients_nested) * train_val_test_splits[0])
    # n_val = int(len(patients_nested) * train_val_test_splits[1])
    # n_test = int(len(patients_nested) - (n_train + n_val))
    # rng = np.random.default_rng(seed=12345)
    # train_val_test_indexer = np.array([0] * n_train + [1] * n_val + [2] * n_test)
    # rng.shuffle(train_val_test_indexer)  # in-place
    # patients_seq_test_np_ragged = patients_seq_np_ragged[train_val_test_indexer == 2]
    # patient_mortalities_test_np = patient_mortalities_np[train_val_test_indexer == 2]

    # load interactions (original data from drugbank, translated to NDC codes)
    ndc_interactions_df = pd.read_csv(ndc_interactions_filepath, dtype={"ndc_a": str, "ndc_b": str})
    ndc_interactions_df = ndc_interactions_df[ndc_interactions_df["mapping_successful"]]
    if use_truncated_codes:
        ndc_interactions_df["event_code_a"] = "M_" + ndc_interactions_df["ndc_a"].str.slice(0, 9)
        ndc_interactions_df["event_code_b"] = "M_" + ndc_interactions_df["ndc_b"].str.slice(0, 9)
    else:
        ndc_interactions_df["event_code_a"] = "M_" + ndc_interactions_df["ndc_a"]
        ndc_interactions_df["event_code_b"] = "M_" + ndc_interactions_df["ndc_b"]
    ndc_interactions_df["event_code_idx_a"] = ndc_interactions_df["event_code_a"].map(code_str_to_idx_map)
    ndc_interactions_df["event_code_idx_b"] = ndc_interactions_df["event_code_b"].map(code_str_to_idx_map)
    ndc_interactions_df = ndc_interactions_df[
        ~ndc_interactions_df[["event_code_idx_a", "event_code_idx_b"]].isnull().any(axis=1)]
    ndc_interactions_df["event_code_idx_a"] = ndc_interactions_df["event_code_idx_a"].astype(int)
    ndc_interactions_df["event_code_idx_b"] = ndc_interactions_df["event_code_idx_b"].astype(int)
    interaction_index_pairs = []  # List[Tuple[int, int]]
    for _, row in ndc_interactions_df.iterrows():
        interaction_index_pairs.append((int(row["event_code_idx_a"]), int(row["event_code_idx_b"])))
    n_interactions = len(interaction_index_pairs)

    # =========================================================================
    # Generate probe sequences
    # =========================================================================

    print("[INFO] Generating probe sequences")

    # determine number of occurences for each medication
    code_value_counts = [0, 0, 0]
    if use_separator_token:
        code_value_counts.append(0)
    code_value_counts.extend(list(mimic_df[event_code_col_name].value_counts().values))
    code_value_counts_np = np.array(code_value_counts)
    code_probs_np = code_value_counts_np.astype(float) / np.sum(code_value_counts_np)

    # set up rng
    rng = np.random.default_rng(seed=12345)

    # randomly generate non-interacting pairs of codes
    non_interaction_index_pairs = []  # List[Tuple[int, int]]
    while len(non_interaction_index_pairs) < n_interactions:
        non_interacting_code_idx_a = int(rng.choice(n_codes, p=code_probs_np))
        non_interacting_code_idx_b = int(rng.choice(n_codes, p=code_probs_np))
        if non_interacting_code_idx_a == non_interacting_code_idx_b:
            continue
        if (non_interacting_code_idx_a, non_interacting_code_idx_b) in interaction_index_pairs:
            continue
        if (non_interacting_code_idx_b, non_interacting_code_idx_a) in interaction_index_pairs:
            continue
        non_interaction_index_pairs.append((non_interacting_code_idx_a, non_interacting_code_idx_b))
    interaction_index_pairs.extend(non_interaction_index_pairs)
    interaction_index_pair_is_real = [True] * n_interactions + [False] * n_interactions

    # generate a set of baseline probe sequences for each interaction (real or fake)
    # baseline probe sequences cannot contain instances of either interacting code
    # each probe sequence has three variations with only a, only b, or both
    probe_sequences_np = np.full(
        shape=(2 * n_interactions, n_probes, 3, input_length),
        fill_value=code_str_to_idx_map["PADDING"])
    for interaction_idx, (interacting_code_idx_a, interacting_code_idx_b) in enumerate(interaction_index_pairs):
        # compute probability distribution for generating random codes
        code_value_counts_adjusted_np = np.copy(code_value_counts_np)
        code_value_counts_adjusted_np[interacting_code_idx_a] = 0
        code_value_counts_adjusted_np[interacting_code_idx_b] = 0
        code_probs_adjusted_np = code_value_counts_adjusted_np.astype(float) / np.sum(code_value_counts_adjusted_np)
        # randomly generate probe sequences, making three identical copies of each
        probe_sequences_np[interaction_idx, :, :, :probe_length] = rng.choice(
            n_codes, size=(n_probes, 1, probe_length), p=code_probs_adjusted_np)
        # randomly swap in interacting codes
        for probe_idx in range(n_probes):
            swap_idx_a, swap_idx_b = rng.choice(probe_length, size=(2,), replace=False)
            probe_sequences_np[interaction_idx, probe_idx, [0, 2], swap_idx_a] = interacting_code_idx_a
            probe_sequences_np[interaction_idx, probe_idx, [1, 2], swap_idx_b] = interacting_code_idx_b

    # =========================================================================
    # Run virtual experiments
    # =========================================================================

    print("[INFO] Extending probe sequences using LSTM")

    print(probe_sequences_np.shape)

    print(probe_sequences_np[0, 0, 0])
    print(probe_sequences_np[0, 0, 1])
    print(probe_sequences_np[0, 0, 2])
    print("")

    # import time
    # start_time = time.time()

    # use LSTM to extend probe sequences
    lstm_model = k.models.load_model(lstm_model_filepath, compile=False)
    probe_sequences_chunks = list(probe_sequences_np)  # need to do one at a time because memory constraints
    probe_sequences_chunks_completed = []
    for chunk_idx, probe_sequences_chunk_np in enumerate(probe_sequences_chunks):

        # try:
        #     print(((time.time() - start_time) * (len(probe_sequences_chunks) - chunk_idx) / chunk_idx) / 3600)
        # except:
        #     pass

        probe_sequences_chunk_np = probe_sequences_chunk_np.reshape((-1, input_length))
        for prediction_timestep_idx in range(probe_length, input_length):
            lstm_predictions = lstm_model.predict(x=probe_sequences_chunk_np, batch_size=batch_size)
            probe_sequences_chunk_np[:, prediction_timestep_idx] = np.argmax(
                lstm_predictions[:, prediction_timestep_idx - 1, :], axis=1)

            print(probe_sequences_chunk_np[0])

        probe_sequences_chunk_np = probe_sequences_chunk_np.reshape((n_probes, 3, input_length))
        probe_sequences_chunks_completed.append(probe_sequences_chunk_np)
    probe_sequences_completed_np = np.concatenate(probe_sequences_chunks_completed, axis=0)

    print(probe_sequences_completed_np[0, 0, 0])
    print(probe_sequences_completed_np[0, 0, 1])
    print(probe_sequences_completed_np[0, 0, 2])




def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--preprocessed_mimic_filepath', type=str, required=True)
    parser.add_argument('--ndc_interactions_filepath', type=str, required=True)
    parser.add_argument('--lstm_model_filepath', type=str, required=True)
    parser.add_argument('--output_directory', type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(
        args.preprocessed_mimic_filepath,
        args.ndc_interactions_filepath,
        args.lstm_model_filepath,
        args.output_directory)
