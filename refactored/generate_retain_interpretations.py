
import os

from retain_interpretations import *


def main(preprocessed_mimic_filepath, input_data_filepath, retain_weights_filepath, output_directory):

    # =========================================================================
    # Hyperparameters
    # =========================================================================

    # dataset filtering - these parameters MUST be identical across all scripts
    use_truncated_codes = True
    proportion_event_instances = 0.9  # {0.5, 0.8, 0.9, 0.95, 0.99}
    admissions_per_patient_incl_min = 1
    medications_per_patient_incl_min = 50  # patients with less will be excluded entirely
    medications_per_patient_incl_max = 100  # patients with more or equal will have early medications truncated

    # parameters for running RETAIN interpretations
    batch_size = 128
    input_data_type = "virtual_experiments"  # {"virtual_experiments", "sample_and_swap"}

    # for debugging
    interactions_to_run = 4  # int to limit number, or None for all

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
    assert input_data_type in {"virtual_experiments", "sample_and_swap"}
    if input_data_type == "virtual_experiments":
        interpretation_visit_idx = 0
    else:  # input_data_type == "sample_and_swap":
        interpretation_visit_idx = -1

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
    # padding is represented by the categorical index n_codes (the last index)
    # note that the mapping between meaning and categorical index for RETAIN
    # differs from that used by the LSTM, so LSTM predictions should be first
    # translated to NDC codes before being input into RETAIN
    code_idx_to_str_map = ["OUTCOME_SURVIVAL", "OUTCOME_MORTALITY"]
    code_idx_to_str_map.extend(list(mimic_df[event_code_col_name].value_counts().index))
    # n_codes = len(code_idx_to_str_map)
    code_str_to_idx_map = dict([(code_str, code_idx) for code_idx, code_str in enumerate(code_idx_to_str_map)])
    # mimic_df["event_code_idx"] = mimic_df[event_code_col_name].map(code_str_to_idx_map)

    # load data to do interpretation on, and translate event code strings to indices
    data_df = pd.read_csv(input_data_filepath)
    data_df["event_code_idx"] = data_df["event_code"].map(code_str_to_idx_map)

    # unpack data into a nested list
    interaction_code_idx_pairs = []
    interaction_is_real = []
    data_nested = []  # (interactions, patients, 3, admissions, events)
    interactions_ran = 0
    for interaction_id, interaction_df in data_df.groupby("interaction_id", sort=False):
        if (interactions_to_run is not None) and (interactions_ran >= interactions_to_run):
            break
        interactions_ran += 1
        interaction_code_idx_a = code_str_to_idx_map[interaction_df["interaction_event_code_a"].iloc[0]]
        interaction_code_idx_b = code_str_to_idx_map[interaction_df["interaction_event_code_b"].iloc[0]]
        interaction_code_idx_pairs.append((interaction_code_idx_a, interaction_code_idx_b))
        interaction_is_real.append(interaction_df["interaction_is_real"].iloc[0])
        interaction_patients_nested = []
        for patient_id, patient_df in interaction_df.groupby("patient_id", sort=False):
            patient_only_a_df = patient_df[patient_df["patient_type"] == "a"]
            patient_only_b_df = patient_df[patient_df["patient_type"] == "b"]
            patient_both_df = patient_df[patient_df["patient_type"] == "both"]
            patient_variants_nested = []
            for patient_variant_df in [patient_only_a_df, patient_only_b_df, patient_both_df]:
                patient_variant_admissions_nested = []
                for admission_id, admission_df in patient_variant_df.groupby("admission_id", sort=False):
                    patient_variant_admissions_nested.append(list(admission_df["event_code_idx"]))
                patient_variants_nested.append(patient_variant_admissions_nested)
            interaction_patients_nested.append(patient_variants_nested)
        data_nested.append(interaction_patients_nested)

    # =========================================================================
    # Run RETAIN prediction and interpretation
    # =========================================================================

    print("[INFO] Running RETAIN prediction and interpretation")

    # coerce data into a RETAIN-friendly format
    n_interactions = len(data_nested)
    n_patients_per_interaction = len(data_nested[0])
    data_for_retain = []  # (n_interactions * n_patients_per_interaction * 3, admissions, events)
    for interaction_patients_nested in data_nested:
        for patient_variants_nested in interaction_patients_nested:
            for patient_variant_admissions_nested in patient_variants_nested:
                data_for_retain.append(patient_variant_admissions_nested)
    data_for_retain = [np.array(data_for_retain)]
    code_idx_to_str_map.append("PADDING")  # last index is used for padding

    # run RETAIN predictions and set up for RETAIN interpretations
    retain_model, retain_model_with_attention = import_model(retain_weights_filepath)
    retain_model_parameters = get_model_parameters(retain_model)
    ARGS = argparse.Namespace
    ARGS.batch_size = batch_size
    predicted_mortality_probs = get_predictions(retain_model, data_for_retain, retain_model_parameters, ARGS)
    ARGS.batch_size = 1
    data_generator_for_retain = SequenceBuilder(data_for_retain, retain_model_parameters, ARGS)

    # run RETAIN interpretations
    interaction_patient_importances = []  # (interactions, patients, [only_a, only_b, both_a, both_b])
    interaction_patient_mortality_probs = []  # (interactions, patients, [only_a, only_b, both])
    for interaction_idx in range(n_interactions):
        interaction_code_idx_a, interaction_code_idx_b = interaction_code_idx_pairs[interaction_idx]
        interaction_code_str_a = code_idx_to_str_map[interaction_code_idx_a]
        interaction_code_str_b = code_idx_to_str_map[interaction_code_idx_b]
        patient_importances = []
        patient_mortality_probs = []
        for patient_idx in range(n_patients_per_interaction):
            importance_only_a = None
            importance_only_b = None
            importance_both_a = None
            importance_both_b = None
            mortality_prob_only_a = None
            mortality_prob_only_b = None
            mortality_prob_both = None
            for patient_variant_idx, patient_variant_type in enumerate(["a", "b", "both"]):
                retain_input_idx = (
                    (((interaction_idx * n_patients_per_interaction) + patient_idx) * 3) + patient_variant_idx)
                retain_input = data_generator_for_retain.__getitem__(retain_input_idx)
                _, alphas, betas = retain_model_with_attention.predict_on_batch(retain_input)
                admit_interpretation_dfs = get_importances(
                    alphas[0], betas[0], retain_input, retain_model_parameters, code_idx_to_str_map)
                admit_interpretation_df = admit_interpretation_dfs[interpretation_visit_idx]
                if patient_variant_type == "a":
                    importance_only_a = admit_interpretation_df[
                        admit_interpretation_df["feature"] == interaction_code_str_a]["importance_feature"].values[0]
                    mortality_prob_only_a = predicted_mortality_probs[retain_input_idx][0][0]
                elif patient_variant_type == "b":
                    importance_only_b = admit_interpretation_df[
                        admit_interpretation_df["feature"] == interaction_code_str_b]["importance_feature"].values[0]
                    mortality_prob_only_b = predicted_mortality_probs[retain_input_idx][0][0]
                else:
                    importance_both_a = admit_interpretation_df[
                        admit_interpretation_df["feature"] == interaction_code_str_a]["importance_feature"].values[0]
                    importance_both_b = admit_interpretation_df[
                        admit_interpretation_df["feature"] == interaction_code_str_b]["importance_feature"].values[0]
                    mortality_prob_both = predicted_mortality_probs[retain_input_idx][0][0]
            patient_importances.append((importance_only_a, importance_only_b, importance_both_a, importance_both_b))
            patient_mortality_probs.append((mortality_prob_only_a, mortality_prob_only_b, mortality_prob_both))
        interaction_patient_importances.append(patient_importances)
        interaction_patient_mortality_probs.append(patient_mortality_probs)

    # =========================================================================
    # Format and save importance scores and mortality probabilities
    # =========================================================================

    print("[INFO] Formatting and saving RETAIN output")

    output_interaction_id = []
    output_interaction_code_str_a = []
    output_interaction_code_str_b = []
    output_interaction_is_real = []
    output_patient_id = []
    output_importance_only_a = []
    output_importance_only_b = []
    output_importance_both_a = []
    output_importance_both_b = []
    output_mortality_prob_only_a = []
    output_mortality_prob_only_b = []
    output_mortality_prob_both = []

    for interaction_idx in range(n_interactions):
        interaction_code_idx_a, interaction_code_idx_b = interaction_code_idx_pairs[interaction_idx]
        interaction_code_str_a = code_idx_to_str_map[interaction_code_idx_a]
        interaction_code_str_b = code_idx_to_str_map[interaction_code_idx_b]
        is_real = interaction_is_real[interaction_idx]
        for patient_idx in range(n_patients_per_interaction):
            importance_only_a, importance_only_b, importance_both_a, importance_both_b = (
                interaction_patient_importances[interaction_idx][patient_idx])
            mortality_prob_only_a, mortality_prob_only_b, mortality_prob_both = (
                interaction_patient_mortality_probs[interaction_idx][patient_idx])
            output_interaction_id.append(interaction_idx)
            output_interaction_code_str_a.append(interaction_code_str_a)
            output_interaction_code_str_b.append(interaction_code_str_b)
            output_interaction_is_real.append(is_real)
            output_patient_id.append(patient_idx)
            output_importance_only_a.append(importance_only_a)
            output_importance_only_b.append(importance_only_b)
            output_importance_both_a.append(importance_both_a)
            output_importance_both_b.append(importance_both_b)
            output_mortality_prob_only_a.append(mortality_prob_only_a)
            output_mortality_prob_only_b.append(mortality_prob_only_b)
            output_mortality_prob_both.append(mortality_prob_both)
            
    output_df = pd.DataFrame({
        "interaction_id": output_interaction_id,
        "interaction_code_str_a": output_interaction_code_str_a,
        "interaction_code_str_b": output_interaction_code_str_b,
        "interaction_is_real": output_interaction_is_real,
        "patient_id": output_patient_id,
        "importance_only_a": output_importance_only_a,
        "importance_only_b": output_importance_only_b,
        "importance_both_a": output_importance_both_a,
        "importance_both_b": output_importance_both_b,
        "mortality_prob_only_a": output_mortality_prob_only_a,
        "mortality_prob_only_b": output_mortality_prob_only_b,
        "mortality_prob_both": output_mortality_prob_both,
    })

    output_df.to_csv(os.path.join(output_directory, f"retain_interpretations_{input_data_type}.csv"))


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--preprocessed_mimic_filepath', type=str, required=True)
    parser.add_argument('--input_data_filepath', type=str, required=True)
    parser.add_argument('--retain_weights_filepath', type=str, required=True)
    parser.add_argument('--output_directory', type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(
        args.preprocessed_mimic_filepath,
        args.input_data_filepath,
        args.retain_weights_filepath,
        args.output_directory)
