
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, PrecisionRecallDisplay


def main(retain_interpretations_filepath, input_data_type, output_directory):

    retain_df = pd.read_csv(retain_interpretations_filepath)

    # did the sum of the importances for a and b go up when combining the two medications?
    retain_df["importance_sum_only"] = retain_df["importance_only_a"] + retain_df["importance_only_b"]
    retain_df["importance_sum_both"] = retain_df["importance_both_a"] + retain_df["importance_both_b"]
    retain_df["importance_sum_delta"] = retain_df["importance_sum_both"] - retain_df["importance_sum_only"]
    retain_df["patient_score"] = retain_df["importance_sum_delta"].abs()

    # aggregate across all patients for each interaction
    interaction_scores = []
    interaction_is_real = []
    for interaction_id, interaction_df in retain_df.groupby("interaction_id", sort=False):
        values = interaction_df["patient_score"].values
        interaction_scores.append(np.mean(values))
        interaction_is_real.append(interaction_df["interaction_is_real"].iloc[0])
    interaction_scores = np.array(interaction_scores)
    interaction_scores -= min(interaction_scores)
    interaction_scores /= max(interaction_scores)

    # precision recall for sum method
    precision, recall, thresholds = precision_recall_curve(
        np.array(interaction_is_real), np.array(interaction_scores))
    print(f"[INFO] Average Precision score for {input_data_type}, RETAIN interpretations: "
          f"{average_precision_score(np.array(interaction_is_real), interaction_scores)}")
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.ylim(-0.05, 1.05)
    plt.title(f"PR curve for {input_data_type}, RETAIN interpretations")
    plt.savefig(os.path.join(output_directory, f"{input_data_type}_retain_interpretations_precision_recall.png"))


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--retain_interpretations_filepath', type=str, required=True)
    parser.add_argument('--input_data_type', type=str, required=True)
    parser.add_argument('--output_directory', type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(
        args.retain_interpretations_filepath,
        args.input_data_type,
        args.output_directory)
