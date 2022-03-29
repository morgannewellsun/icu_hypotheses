
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, PrecisionRecallDisplay

warnings.filterwarnings("error")


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
    n_interactions = len(interaction_index_pairs)

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
    interaction_is_real = [True] * n_interactions + [False] * n_interactions

    # =========================================================================
    # Calculate mortality rates
    # =========================================================================

    print("[INFO] Calculating mortality rates")

    # determine which drugs each patient was prescribed for the last admission
    patient_drugs_np = np.full(shape=(len(patients_nested), n_codes), fill_value=False, dtype=bool)
    for patient_idx, patient_admissions in enumerate(patients_nested):
        patient_drugs_np[patient_idx, patient_admissions[-1]] = True

    # calculate mortality rates
    mortality_rates_only_a = []
    mortality_rates_only_b = []
    mortality_rates_both = []
    patient_mortalities = np.array(patient_mortalities)
    for interacting_code_idx_a, interacting_code_idx_b in interaction_index_pairs:
        try:
            mortality_rates_only_a.append(np.mean(
                patient_mortalities[
                    patient_drugs_np[:, interacting_code_idx_a] & ~patient_drugs_np[:, interacting_code_idx_b]]))
            mortality_rates_only_b.append(np.mean(
                patient_mortalities[
                    ~patient_drugs_np[:, interacting_code_idx_a] & patient_drugs_np[:, interacting_code_idx_b]]))
            mortality_rates_both.append(np.mean(
                patient_mortalities[
                    patient_drugs_np[:, interacting_code_idx_a] & patient_drugs_np[:, interacting_code_idx_b]]))
        except RuntimeWarning:
            mortality_rates_only_a.append(0)
            mortality_rates_only_b.append(0)
            mortality_rates_both.append(0)

    # by what amount did the mortality rate increase?
    interaction_scores = []
    for mortality_rate_only_a, mortality_rate_only_b, mortality_rate_both in zip(
            mortality_rates_only_a, mortality_rates_only_b, mortality_rates_both):
        interaction_scores.append(max(0, mortality_rate_both - max(mortality_rate_only_a, mortality_rate_only_b)))
    interaction_scores = np.array(interaction_scores)
    interaction_scores /= max(interaction_scores)

    # precision recall
    precision, recall, thresholds = precision_recall_curve(np.array(interaction_is_real), interaction_scores)
    print(f"[INFO] Average Precision score for statistical mortality method: "
          f"{average_precision_score(np.array(interaction_is_real), np.array(interaction_scores))}")
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.ylim(-0.05, 1.05)
    plt.title("PR curve for statistical mortality method")
    plt.savefig(os.path.join(output_directory, f"statistical_mortality_precision_recall.png"))


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--preprocessed_mimic_filepath', type=str, required=True)
    parser.add_argument('--ndc_interactions_filepath', type=str, required=True)
    parser.add_argument('--output_directory', type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(
        args.preprocessed_mimic_filepath,
        args.ndc_interactions_filepath,
        args.output_directory)
