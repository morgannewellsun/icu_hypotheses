
import argparse
import os

import numpy as np
import pandas as pd


def generate_patient_variants(
        patient_admissions_raw,
        interacting_code_idx_a, interacting_code_idx_b,
        rng, n_codes, code_probs_adjusted_np):
    patient_admissions_only_a = patient_admissions_raw[:-1]
    patient_admissions_only_b = patient_admissions_raw[:-1]
    patient_admissions_both = patient_admissions_raw[:-1]
    reverse_last_admission_events_only_a = []
    reverse_last_admission_events_only_b = []
    reverse_last_admission_events_both = []
    for event_code_idx in patient_admissions_raw[-1][::-1]:
        if event_code_idx == interacting_code_idx_a:
            random_code = rng.choice(n_codes, p=code_probs_adjusted_np)
            reverse_last_admission_events_only_b.append(random_code)
            if interacting_code_idx_a not in reverse_last_admission_events_only_a:
                reverse_last_admission_events_only_a.append(interacting_code_idx_a)
                reverse_last_admission_events_both.append(interacting_code_idx_a)
            else:
                reverse_last_admission_events_only_a.append(random_code)
                reverse_last_admission_events_both.append(random_code)
        elif event_code_idx == interacting_code_idx_b:
            random_code = rng.choice(n_codes, p=code_probs_adjusted_np)
            reverse_last_admission_events_only_a.append(random_code)
            if interacting_code_idx_b not in reverse_last_admission_events_only_b:
                reverse_last_admission_events_only_b.append(interacting_code_idx_b)
                reverse_last_admission_events_both.append(interacting_code_idx_b)
            else:
                reverse_last_admission_events_only_b.append(random_code)
                reverse_last_admission_events_both.append(random_code)
        else:
            reverse_last_admission_events_only_a.append(event_code_idx)
            reverse_last_admission_events_only_b.append(event_code_idx)
            reverse_last_admission_events_both.append(event_code_idx)
    patient_admissions_only_a.append(reverse_last_admission_events_only_a[::-1])
    patient_admissions_only_b.append(reverse_last_admission_events_only_b[::-1])
    patient_admissions_both.append(reverse_last_admission_events_both[::-1])
    return patient_admissions_only_a, patient_admissions_only_b, patient_admissions_both


def main(preprocessed_mimic_filepath, ndc_interactions_filepath, output_directory):

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
    train_val_test_splits = (0.8, 0.1, 0.1)  # MUST be identical across all scripts
    n_patients_per_interaction = 128  # should be the same as n_probes in run_virtual_experiments

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

    # split train/val/test and extract test set patients (train and val patients not needed past this point)
    n_train = int(len(patients_nested) * train_val_test_splits[0])
    n_val = int(len(patients_nested) * train_val_test_splits[1])
    n_test = int(len(patients_nested) - (n_train + n_val))
    rng = np.random.default_rng(seed=12345)
    train_val_test_indexer = np.array([0] * n_train + [1] * n_val + [2] * n_test)
    rng.shuffle(train_val_test_indexer)  # in-place
    patients_nested_test = []
    for patient_admissions, indexer_value in zip(patients_nested, train_val_test_indexer):
        if indexer_value == 2:
            patients_nested_test.append(patient_admissions)

    # =========================================================================
    # Sample and process patient data
    # =========================================================================

    print("[INFO] Sampling and processing patient data")

    # determine number of occurences for each medication
    code_value_counts = [0, 0, 0]
    if use_separator_token:
        code_value_counts.append(0)
    code_value_counts.extend(list(mimic_df[event_code_col_name].value_counts().values))
    code_value_counts_np = np.array(code_value_counts)
    code_probs_np = code_value_counts_np.astype(float) / np.sum(code_value_counts_np)

    # randomly generate non-interacting pairs of codes
    rng = np.random.default_rng(seed=12345)
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

    # determine which drugs each patient was prescribed for the last admission
    patient_drugs_np = np.full(shape=(len(patients_nested_test), n_codes), fill_value=False, dtype=bool)
    for patient_idx, patient_admissions in enumerate(patients_nested_test):
        patient_drugs_np[patient_idx, patient_admissions[-1]] = True

    # sample n_patients_per_interaction patients for each interaction pair
    total_n_both = 0
    total_n_one = 0
    total_n_neither = 0
    data = []  # (n_interactions, n_patients_per_interaction, 3, admissions, events)
    for interacting_code_idx_a, interacting_code_idx_b in interaction_index_pairs:

        # compute probability distribution for generating random codes that are not a or b
        code_value_counts_adjusted_np = np.copy(code_value_counts_np)
        code_value_counts_adjusted_np[interacting_code_idx_a] = 0
        code_value_counts_adjusted_np[interacting_code_idx_b] = 0
        code_probs_adjusted_np = code_value_counts_adjusted_np.astype(float) / np.sum(code_value_counts_adjusted_np)

        # collect data
        interaction_data = []  # (n_patients_per_interaction, 3, admissions, events)

        # first sample patients who were prescribed both interacting drugs
        patient_indices_both = np.where(
            patient_drugs_np[:, interacting_code_idx_a] & patient_drugs_np[:, interacting_code_idx_b])[0]
        for patient_idx in patient_indices_both:
            if len(interaction_data) == n_patients_per_interaction:
                break
            patient_admissions_raw = patients_nested_test[patient_idx]
            total_n_both += 1
            patient_admissions_only_a, patient_admissions_only_b, patient_admissions_both = generate_patient_variants(
                patient_admissions_raw,
                interacting_code_idx_a, interacting_code_idx_b,
                rng, n_codes, code_probs_adjusted_np)
            interaction_data.append((patient_admissions_only_a, patient_admissions_only_b, patient_admissions_both))

        # if more patients are needed, use patients who were administered one of the interacting drugs
        # collate the only_a and only_b sets to achieve as balanced a mix as possible
        patient_indices_only_a = np.where(
            patient_drugs_np[:, interacting_code_idx_a] & ~patient_drugs_np[:, interacting_code_idx_b])[0]
        patient_indices_only_b = np.where(
            ~patient_drugs_np[:, interacting_code_idx_a] & patient_drugs_np[:, interacting_code_idx_b])[0]
        patient_indices_only_one = []
        patient_contains_code_a = []
        for i in range(max(len(patient_indices_only_a), len(patient_indices_only_b))):
            if i < len(patient_indices_only_a):
                patient_indices_only_one.append(patient_indices_only_a[i])
                patient_contains_code_a.append(True)
            if i < len(patient_indices_only_b):
                patient_indices_only_one.append(patient_indices_only_b[i])
                patient_contains_code_a.append(False)
        for patient_idx, contains_code_a in zip(patient_indices_only_one, patient_contains_code_a):
            if len(interaction_data) == n_patients_per_interaction:
                break
            patient_admissions_raw = patients_nested_test[patient_idx]
            if len(patient_admissions_raw[-1]) < 2:
                continue
            total_n_one += 1
            interacting_code_idx_present = interacting_code_idx_a if contains_code_a else interacting_code_idx_b
            interacting_code_idx_absent = interacting_code_idx_b if contains_code_a else interacting_code_idx_a
            swap_idx = rng.choice(np.where(np.array(patient_admissions_raw[-1]) != interacting_code_idx_present)[0])
            last_admission_events_raw = []
            for event_idx, event_code_idx in enumerate(patient_admissions_raw[-1]):
                if event_idx == swap_idx:
                    last_admission_events_raw.append(interacting_code_idx_absent)
                else:
                    last_admission_events_raw.append(event_code_idx)
            patient_admissions_raw = patient_admissions_raw[:-1]
            patient_admissions_raw.append(last_admission_events_raw)
            patient_admissions_only_a, patient_admissions_only_b, patient_admissions_both = generate_patient_variants(
                patient_admissions_raw,
                interacting_code_idx_a, interacting_code_idx_b,
                rng, n_codes, code_probs_adjusted_np)
            interaction_data.append((patient_admissions_only_a, patient_admissions_only_b, patient_admissions_both))

        # if more patients are needed, use patients who were administered neither interacting drug
        patient_indices_neither = np.where(
            ~patient_drugs_np[:, interacting_code_idx_a] & patient_drugs_np[:, interacting_code_idx_b])[0]
        while len(interaction_data) < n_patients_per_interaction:
            patient_admissions_raw = patients_nested_test[rng.choice(patient_indices_neither)]
            if len(patient_admissions_raw[-1]) < 2:
                continue
            total_n_neither += 1
            swap_idx_a, swap_idx_b = rng.choice(len(patient_admissions_raw[-1]), size=2, replace=False)
            last_admission_events_raw = []
            for event_idx, event_code_idx in enumerate(patient_admissions_raw[-1]):
                if event_idx == swap_idx_a:
                    last_admission_events_raw.append(interacting_code_idx_a)
                elif event_idx == swap_idx_b:
                    last_admission_events_raw.append(interacting_code_idx_b)
                else:
                    last_admission_events_raw.append(event_code_idx)
            patient_admissions_raw = patient_admissions_raw[:-1]
            patient_admissions_raw.append(last_admission_events_raw)
            patient_admissions_only_a, patient_admissions_only_b, patient_admissions_both = generate_patient_variants(
                patient_admissions_raw,
                interacting_code_idx_a, interacting_code_idx_b,
                rng, n_codes, code_probs_adjusted_np)
            interaction_data.append((patient_admissions_only_a, patient_admissions_only_b, patient_admissions_both))

        # collect data
        data.append(interaction_data)

    print(f"[INFO] total_n_both    : {total_n_both}")
    print(f"[INFO] total_n_one     : {total_n_one}")
    print(f"[INFO] total_n_neither : {total_n_neither}")

    # =========================================================================
    # Format and save data
    # =========================================================================

    print("[INFO] Formatting and saving data")

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
    for interaction_id, (interaction_data, code_index_pair, is_real) in enumerate(
            zip(data, interaction_index_pairs, interaction_index_pair_is_real)):
        interaction_event_code_a = code_idx_to_str_map[code_index_pair[0]]
        interaction_event_code_b = code_idx_to_str_map[code_index_pair[1]]
        for three_patient_versions in interaction_data:
            for patient_admissions, version_type in zip(three_patient_versions, ["a", "b", "both"]):
                for admission_events in patient_admissions:
                    for event_code_idx in admission_events:
                        interaction_ids.append(interaction_id)
                        interaction_event_codes_a.append(interaction_event_code_a)
                        interaction_event_codes_b.append(interaction_event_code_b)
                        interaction_is_real.append(is_real)
                        patient_ids.append(patient_id)
                        patient_types.append(version_type)
                        admission_ids.append(admission_id)
                        event_codes.append(code_idx_to_str_map[event_code_idx])
                    admission_id += 1
            patient_id += 1

    sample_and_swap_df = pd.DataFrame({
        "interaction_id": interaction_ids,
        "interaction_event_code_a": interaction_event_codes_a,
        "interaction_event_code_b": interaction_event_codes_b,
        "interaction_is_real": interaction_is_real,
        "patient_id": patient_ids,
        "patient_type": patient_types,
        "admission_id": admission_ids,
        "event_code": event_codes,
    })

    sample_and_swap_df.to_csv(os.path.join(output_directory, "sample_and_swap_df.csv"))


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--preprocessed_mimic_filepath', type=str, required=True)
    parser.add_argument('--ndc_interactions_filepath', type=str, required=True)
    parser.add_argument('--output_directory', type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args.preprocessed_mimic_filepath, args.ndc_interactions_filepath, args.output_directory)
