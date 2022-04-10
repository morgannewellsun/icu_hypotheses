
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, PrecisionRecallDisplay


def main(output_directory):

    oos_only = True
    n_iters = 100000

    ap_scores_to_test = (  # [ss_interpretations, ve_interpretations, ss_mort, ve_mort, (statistics)]
        [0.5328553219708758, 0.6904926852314915, 0.5634411066633507, 0.5969031781574752]
        if oos_only
        else [0.6380275722410786, 0.6093312875498686, 0.4944314042580634, 0.4896839992737583, 0.5329816615339392])

    name_str = "oos_only" if oos_only else "all_pairs"

    # =========================================================================
    # Monte Carlo simulation of AP score distribution for random uniform score strategy
    # =========================================================================

    n_tp = 44 if oos_only else 711
    n_tn = 69 if oos_only else 711

    ap_scores = []
    interaction_scores_all = []  # List[ndarray]
    interaction_is_real_all = []  # List[ndarray]
    rng = np.random.default_rng(seed=12345)
    for _ in range(n_iters):
        interaction_scores = rng.uniform(size=(n_tp + n_tn,)).astype(np.float32)
        interaction_scores_all.append(interaction_scores)
        interaction_is_real = np.array([True] * n_tp + [False] * n_tn)
        interaction_is_real_all.append(interaction_is_real)
        ap_scores.append(average_precision_score(interaction_is_real, interaction_scores))
    ap_scores = np.array(ap_scores)
    interaction_scores_all = np.array(interaction_scores_all).flatten()
    interaction_is_real_all = np.array(interaction_is_real_all).flatten()

    print(f"[INFO] Average Precision score for naive random uniform score strategy, {name_str}: "
          f"{average_precision_score(interaction_is_real_all, interaction_scores_all)}")
    precision, recall, thresholds = precision_recall_curve(
        np.array(interaction_is_real_all), np.array(interaction_scores_all))
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.ylim(-0.05, 1.05)
    plt.title(f"PR curve for naive random uniform scores, {name_str}")
    plt.savefig(os.path.join(output_directory, f"naive_random_uniform_score_{name_str}.png"))
    plt.cla()

    # warning: distribution is not normal!
    print(f"[INFO] Mean of AP score distribution for {name_str}: {np.mean(ap_scores)}")
    print(f"[INFO] StDev of AP score distribution for {name_str}: {np.std(ap_scores)}")

    plt.hist(ap_scores, bins=np.linspace(0, 1, 501))
    plt.title(f"AP score distribution for naive random uniform scores, {name_str}")
    plt.savefig(os.path.join(output_directory, f"naive_random_uniform_score_ap_distr_{name_str}.png"))

    for ap_score_to_test in ap_scores_to_test:
        p = np.mean(ap_scores > ap_score_to_test)
        print(f"[INFO] AP score of {ap_score_to_test} corresponds to p={p}")


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output_directory', type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args.output_directory)
