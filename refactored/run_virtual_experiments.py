
import argparse
import os
import time

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
    n_probes = 128
    probe_length = 10  # MUST be identical to probe_length used in train_lstm_on_mimic.py
    sampling_temperature = .95

    # lstm model parameters
    input_length = 100  # MUST be identical to input_length used in train_lstm_on_mimic.py
    batch_size = 1028

    # for debugging
    n_interactions_to_run = None  # int to limit number, or None for all

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
    if n_interactions_to_run is not None:
        print(f"[WARNING] Only running virtual experiments for the first {n_interactions_to_run} interactions")

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
    if n_interactions_to_run is not None:
        interaction_index_pairs = interaction_index_pairs[:n_interactions_to_run]
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
    probe_sequence_swap_indices_np = np.zeros(shape=(2 * n_interactions, n_probes, 2))
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
            probe_sequence_swap_indices_np[interaction_idx, probe_idx] = [swap_idx_a, swap_idx_b]

    # =========================================================================
    # Run virtual experiments
    # =========================================================================

    print("[INFO] Extending probe sequences using LSTM")

    # use LSTM to extend probe sequences
    lstm_model = k.models.load_model(lstm_model_filepath, compile=False)
    interaction_probe_sequences = list(probe_sequences_np)  # need to do one at a time because memory constraints
    interaction_probe_sequences_completed = []
    start_time = time.time()
    for interaction_idx, interaction_probe_sequences_np in enumerate(interaction_probe_sequences):
        if (interaction_idx - 1) % 10 == 0:
            estimated_time_remaining = (
                (time.time() - start_time) * (len(interaction_probe_sequences) - interaction_idx) / interaction_idx)
            print(f"[INFO] Estimated time remaining: {np.around(estimated_time_remaining / 3600, 2)} hours")
        interaction_probe_sequences_np = interaction_probe_sequences_np.reshape((-1, input_length))
        for prediction_timestep_idx in range(probe_length, input_length):
            # original probability weights for next timestep from LSTM
            lstm_predictions = lstm_model.predict(x=interaction_probe_sequences_np, batch_size=batch_size)
            lstm_predictions = lstm_predictions[:, prediction_timestep_idx - 1, :]
            # adjust probability weights using sampling_temperature
            lstm_predictions = np.exp(np.log(lstm_predictions) / sampling_temperature)
            # remove chance of sampling either of the interacting codes
            lstm_predictions[:, interaction_index_pairs[interaction_idx]] = 0
            # parallelized random sampling
            lstm_predictions = np.cumsum(lstm_predictions, axis=1)
            lstm_predictions = lstm_predictions / lstm_predictions[:, -1:]
            lstm_predictions = np.sum(lstm_predictions < rng.uniform(size=(lstm_predictions.shape[0], 1)), axis=1)
            # add samples to the sequence
            interaction_probe_sequences_np[:, prediction_timestep_idx] = lstm_predictions
        interaction_probe_sequences_np = interaction_probe_sequences_np.reshape((n_probes, 3, input_length))
        interaction_probe_sequences_completed.append(interaction_probe_sequences_np)

    # =========================================================================
    # Format and save virtual experiments output
    # =========================================================================

    print("[INFO] Formatting and saving virtual experiments results")

    interaction_ids = []
    interaction_event_codes_a = []
    interaction_event_codes_b = []
    interaction_is_real = []
    patient_ids = []
    patient_types = []  # List[{"a", "b", "both"}]
    admission_ids = []
    event_codes = []

    patient_id = 0
    admission_id = 0
    for interaction_id, (sequences_completed_np, code_index_pair, is_real) in enumerate(
            zip(interaction_probe_sequences_completed, interaction_index_pairs, interaction_index_pair_is_real)):
        interaction_event_code_a = code_idx_to_str_map[code_index_pair[0]]
        interaction_event_code_b = code_idx_to_str_map[code_index_pair[1]]
        for three_sequences_np in sequences_completed_np:
            for sequence, sequence_type in zip(three_sequences_np, ["a", "b", "both"]):
                admission_code_indices = set()
                for code_idx in sequence:
                    code_str = code_idx_to_str_map[code_idx]
                    if code_str == "PADDING":
                        continue
                    elif (code_str == "OUTCOME_SURVIVAL") or (code_str == "OUTCOME_MORTALITY"):
                        admission_id += 1
                        break
                    elif code_str == "SEPARATOR":
                        admission_id += 1
                        admission_code_indices = set()
                        continue
                    elif code_idx in admission_code_indices:  # enforce uniqueness of events within visits
                        continue
                    else:
                        admission_code_indices.add(code_idx)
                        interaction_ids.append(interaction_id)
                        interaction_event_codes_a.append(interaction_event_code_a)
                        interaction_event_codes_b.append(interaction_event_code_b)
                        interaction_is_real.append(is_real)
                        patient_ids.append(patient_id)
                        patient_types.append(sequence_type)
                        admission_ids.append(admission_id)
                        event_codes.append(code_str)
                admission_id += 1
            patient_id += 1

    virtual_experiments_df = pd.DataFrame({
        "interaction_id": interaction_ids,
        "interaction_event_code_a": interaction_event_codes_a,
        "interaction_event_code_b": interaction_event_codes_b,
        "interaction_is_real": interaction_is_real,
        "patient_id": patient_ids,
        "patient_type": patient_types,
        "admission_id": admission_ids,
        "event_code": event_codes,
    })

    virtual_experiments_df.to_csv(os.path.join(output_directory, "virtual_experiments_df.csv"))


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
