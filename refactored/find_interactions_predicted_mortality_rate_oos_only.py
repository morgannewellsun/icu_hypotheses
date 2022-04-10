
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, PrecisionRecallDisplay


def main(
        preprocessed_mimic_filepath,
        retain_interpretations_filepath,
        input_data_type,
        output_directory):

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
    train_val_test_splits = (0.8, 0.1, 0.1)  # MUST be identical across all scripts

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
    use_separator_token = True

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
    # n_codes = len(code_idx_to_str_map)
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

    # split train/val/test and extract test set patients (val and test patients not needed past this point)
    n_train = int(len(patients_nested) * train_val_test_splits[0])
    n_val = int(len(patients_nested) * train_val_test_splits[1])
    n_test = int(len(patients_nested) - (n_train + n_val))
    rng = np.random.default_rng(seed=12345)
    train_val_test_indexer = np.array([0] * n_train + [1] * n_val + [2] * n_test)
    rng.shuffle(train_val_test_indexer)  # in-place
    patients_nested_train = []
    for patient_admissions, indexer_value in zip(patients_nested, train_val_test_indexer):
        if indexer_value == 0:
            patients_nested_train.append(patient_admissions)

    # =========================================================================
    # Find interactions using RETAIN interpretations
    # =========================================================================

    retain_df = pd.read_csv(retain_interpretations_filepath)

    # did the predicted mortality rate go up when combining the two medications?
    retain_df["mortality_prob_max"] = retain_df[["mortality_prob_only_a", "mortality_prob_only_b"]].max(axis=1)
    # retain_df["mortality_prob_increased"] = (
    #     (retain_df["mortality_prob_only_a"] < retain_df["mortality_prob_both"])
    #     & (retain_df["mortality_prob_only_b"] < retain_df["mortality_prob_both"]))
    retain_df["patient_score"] = retain_df["mortality_prob_both"] - retain_df["mortality_prob_max"]

    # aggregate across all patients for each interaction
    interaction_scores = []
    interaction_is_real = []
    n_out_of_sample_tp = 0
    n_out_of_sample_tn = 0
    for interaction_id, interaction_df in retain_df.groupby("interaction_id", sort=False):

        # filter out any in-sample interaction pairs
        interacting_code_idx_a = code_str_to_idx_map[interaction_df["interaction_code_str_a"].iloc[0]]
        interacting_code_idx_b = code_str_to_idx_map[interaction_df["interaction_code_str_b"].iloc[0]]
        is_real = interaction_df["interaction_is_real"].iloc[0]
        in_sample = False
        for patient_admissions in patients_nested_train:
            for admission_events in patient_admissions:
                if (interacting_code_idx_a in admission_events) and (interacting_code_idx_b in admission_events):
                    in_sample = True
                    break
            if in_sample:
                break
        if in_sample:
            continue
            pass
        else:
            if is_real:
                n_out_of_sample_tp += 1
            else:
                n_out_of_sample_tn += 1

        # aggregate remaining scores
        values = interaction_df["patient_score"].values
        interaction_scores.append(np.mean(values))
        interaction_is_real.append(is_real)

    print(f"[INFO] Number of out-of-sample TP interaction pairs: {n_out_of_sample_tp}")
    print(f"[INFO] Number of out-of-sample TN interaction pairs: {n_out_of_sample_tn}")
    interaction_scores = np.array(interaction_scores)
    interaction_scores -= min(interaction_scores)
    interaction_scores /= max(interaction_scores)

    # precision recall for sum method
    precision, recall, thresholds = precision_recall_curve(
        np.array(interaction_is_real), np.array(interaction_scores))
    print(f"[INFO] Average Precision score for {input_data_type}, predicted mortality OOS: "
          f"{average_precision_score(np.array(interaction_is_real), interaction_scores)}")
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.ylim(-0.05, 1.05)
    plt.title(f"PR curve for {input_data_type} OOS, RETAIN predicted mortality method")
    plt.savefig(os.path.join(output_directory, f"{input_data_type}_predicted_mortality_oos_precision_recall.png"))


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--preprocessed_mimic_filepath', type=str, required=True)
    parser.add_argument('--retain_interpretations_filepath', type=str, required=True)
    parser.add_argument('--input_data_type', type=str, required=True)
    parser.add_argument('--output_directory', type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(
        args.preprocessed_mimic_filepath,
        args.retain_interpretations_filepath,
        args.input_data_type,
        args.output_directory)