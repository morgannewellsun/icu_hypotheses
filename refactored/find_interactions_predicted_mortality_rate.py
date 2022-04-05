
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, PrecisionRecallDisplay


def main(retain_interpretations_filepath, input_data_type, output_directory):

    retain_df = pd.read_csv(retain_interpretations_filepath)

    # did the predicted mortality rate go up when combining the two medications?
    retain_df["mortality_prob_max"] = retain_df[["mortality_prob_only_a", "mortality_prob_only_b"]].max(axis=1)
    # retain_df["mortality_prob_increased"] = (
    #     (retain_df["mortality_prob_only_a"] < retain_df["mortality_prob_both"])
    #     & (retain_df["mortality_prob_only_b"] < retain_df["mortality_prob_both"]))
    retain_df["mortality_prob_increased"] = retain_df["mortality_prob_both"] - retain_df["mortality_prob_max"]

    # aggregate across all patients for each interaction
    interaction_scores = []
    interaction_is_real = []
    for interaction_id, interaction_df in retain_df.groupby("interaction_id", sort=False):
        values = interaction_df["mortality_prob_increased"].values
        interaction_scores.append(np.sum(values) / len(values))
        interaction_is_real.append(interaction_df["interaction_is_real"].iloc[0])

    # precision recall
    precision, recall, thresholds = precision_recall_curve(np.array(interaction_is_real), np.array(interaction_scores))
    print(f"[INFO] Average Precision score for {input_data_type} predicted mortality method: "
          f"{average_precision_score(np.array(interaction_is_real), np.array(interaction_scores))}")
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.ylim(-0.05, 1.05)
    plt.title(f"PR curve for {input_data_type}, RETAIN predicted mortality method")
    plt.savefig(os.path.join(output_directory, f"{input_data_type}_retain_mortality_precision_recall.png"))


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
