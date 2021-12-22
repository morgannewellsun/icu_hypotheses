
from collections import deque, namedtuple
import math
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression


def discretize(value, bound_lower, bound_upper, bins):
    return int(bins * (value - bound_lower) / (bound_upper - bound_lower))


PatientVarSpec = namedtuple(
    "PatientVarSpec",
    ["target_var_index", "impact_coeff", "is_beneficial", "is_visible", "physician_knowledge", "codes"])
MedSpec = namedtuple(
    "MedSpec",
    ["target_var_index", "impact_coeff", "impact_delay", "is_beneficial", "physician_knowledge", "codes"])
InteractionSpec = namedtuple(
    "InteractionSpec",
    ["med_index_a", "coeff_a", "med_index_b", "coeff_b"])


def main(out_directory, n_patients, n_timesteps_total, n_timesteps_trunc, train_proportion, val_proportion, verbose):
    # ==========================================================================
    # hard-coded model parameters
    # ==========================================================================

    # number of variables affecting health directly
    n_vars_fo = 0
    # maximum per-timestep impact of a first-order variable on health
    max_var_fo_impact = 1.0
    # number of variables affecting a first-order variable
    n_vars_so = 0
    # maximum per-timestep impact of a second-order variable on its target
    max_var_so_impact = 0.2
    # number of variables whose values are visible
    n_vars_visible = 0

    # number of medications affecting health directly
    n_meds_health = 4
    # maximum per-administration impact of a medication on health
    max_med_health_impact = 1.0
    # number of medications affecting a variable
    n_meds_vars = 0
    # maximum per-administration impact of a medication on a variable
    max_med_var_impact = 0.2
    # maximum number of timesteps the effects of medication will be delayed
    max_med_effect_delay = 0
    # number of medication interactions
    n_interactions = 0
    # whether or not to randomly select a subset of medications for each patient
    med_subset_per_patient = True

    # mean of initial patient health values
    patient_health_offset = 10

    # number of bins to quantize variable observations into
    n_bins_vars = 3
    # number of bins to quantize medication administrations into
    n_bins_meds = 3

    # scale factor from 0 to 1 controlling impact of physician knowledge on medicine administration probabilities
    physician_knowledge_sf = 0.0

    # coefficient controlling the degree of randomization of the effects of medications
    med_impact_randomness = 0.1

    interact_params = {
        "n_vars_fo": n_vars_fo,
        "max_var_fo_impact": max_var_fo_impact,
        "n_vars_so": n_vars_so,
        "max_var_so_impact": max_var_so_impact,
        "n_vars_visible": n_vars_visible,
        "n_meds_health": n_meds_health,
        "max_med_health_impact": max_med_health_impact,
        "n_meds_vars": n_meds_vars,
        "max_med_var_impact": max_med_var_impact,
        "max_med_effect_delay": max_med_effect_delay,
        "n_interactions": n_interactions,
        "patient_health_offset": patient_health_offset,
        "n_bins_vars": n_bins_vars,
        "n_bins_meds": n_bins_meds,
        "physician_knowledge_sf": physician_knowledge_sf,
        "med_impact_randomness": med_impact_randomness,
    }

    # ==========================================================================
    # global setup of variables and medications
    # ==========================================================================

    # enumerate types of possible interactions
    interaction_coeffs_aligned = [(2, 2), (0, 0), (-1, -1)]
    interaction_coeffs_opposing = [(-1, 1), (1, -1), (0, 1), (1, 0), (0, 2), (2, 0)]

    # determine which variables will be visible
    n_vars = n_vars_fo + n_vars_so
    n_vars_invisible = n_vars - n_vars_visible
    visibilities = ([True] * n_vars_visible) + ([False] * n_vars_invisible)
    np.random.shuffle(visibilities)

    # randomly generate the effects of first-order variables
    vars_all = []
    vars_fo = []
    for i in range(n_vars_fo):
        new_target_var_index = None
        new_impact_coeff = np.random.uniform(-1 * max_var_fo_impact, max_var_fo_impact)
        new_is_beneficial = new_impact_coeff >= 0
        new_is_visible = visibilities.pop()
        new_physician_knowledge = np.random.uniform(0, 1) if new_is_visible else None
        new_codes = [n_bins_vars * i + j for j in range(n_bins_vars)]
        new_var = PatientVarSpec(
            target_var_index=new_target_var_index,
            impact_coeff=new_impact_coeff,
            is_beneficial=new_is_beneficial,
            is_visible=new_is_visible,
            physician_knowledge=new_physician_knowledge,
            codes=new_codes)
        vars_all.append(new_var)
        vars_fo.append(new_var)

    # randomly generate the effects of second-order variables
    vars_so = []
    for i in range(n_vars_so):
        new_target_var_index = np.random.randint(n_vars_fo)
        new_impact_coeff = np.random.uniform(-1 * max_var_so_impact, max_var_so_impact)
        new_is_beneficial = (
            new_impact_coeff >= 0
            if vars_fo[new_target_var_index].is_beneficial
            else new_impact_coeff <= 0)
        new_is_visible = visibilities.pop()
        new_physician_knowledge = np.random.uniform(0, 1) if new_is_visible else None
        new_codes = [n_bins_vars * i + j for j in range(n_bins_vars)]
        new_var = PatientVarSpec(
            target_var_index=new_target_var_index,
            impact_coeff=new_impact_coeff,
            is_beneficial=new_is_beneficial,
            is_visible=new_is_visible,
            physician_knowledge=new_physician_knowledge,
            codes=new_codes)
        vars_all.append(new_var)
        vars_so.append(new_var)

    # printing
    if verbose >= 1:
        print("\nRandomly generated variables: ")
        for i, var in enumerate(vars_fo):
            print(
                f"\tVariable v_{i}\n"
                f"\t\tTarget: Health\n"
                f"\t\tImpact coefficient: {var.impact_coeff}\n"
                f"\t\tIs beneficial: {var.is_beneficial}\n"
                f"\t\tIs visible: {var.is_visible}\n"
                f"\t\tPhysician knowledege: {var.physician_knowledge}"
                f"\t\tCodes: {var.codes}")
        for i, var in enumerate(vars_so):
            print(
                f"\tVariable v_{i + n_vars_fo}\n"
                f"\t\tTarget: Variable v_{var.target_var_index}\n"
                f"\t\tImpact coefficient: {var.impact_coeff}\n"
                f"\t\tIs beneficial: {var.is_beneficial}\n"
                f"\t\tIs visible: {var.is_visible}\n"
                f"\t\tPhysician knowledege: {var.physician_knowledge}"
                f"\t\tCodes: {var.codes}")

    # randomly generate medications which directly affect health
    meds_all = []
    meds_health = []
    for i in range(n_meds_health):
        new_target_var_index = None
        new_impact_coeff = np.random.uniform(-1 * max_med_health_impact, max_med_health_impact)
        new_impact_delay = np.random.randint(max_med_effect_delay + 1)
        new_is_beneficial = new_impact_coeff >= 0
        new_physician_knowledge = np.random.uniform(0, 1)
        new_codes = [n_bins_meds * i + j for j in range(n_bins_meds)]

        # TODO temporary modification to hard-code simple medication specs
        print("Warning! You're manually specifying the medication parameters instead of randomly generating them!")
        new_impact_coeff = -1.0
        new_is_beneficial = False
        # TODO remove this later

        new_med = MedSpec(
            target_var_index=new_target_var_index,
            impact_coeff=new_impact_coeff,
            impact_delay=new_impact_delay,
            is_beneficial=new_is_beneficial,
            physician_knowledge=new_physician_knowledge,
            codes=new_codes)
        meds_all.append(new_med)
        meds_health.append(new_med)

    meds_vars = []
    for i in range(n_meds_vars):
        new_target_var_index = np.random.randint(n_vars)
        new_impact_coeff = np.random.uniform(-1 * max_med_var_impact, max_med_var_impact)
        new_impact_delay = np.random.randint(max_med_effect_delay + 1)
        new_is_beneficial = (
            new_impact_coeff >= 0
            if vars_all[new_target_var_index].is_beneficial
            else new_impact_coeff <= 0)
        new_physician_knowledge = np.random.uniform(0, 1)
        new_codes = [n_bins_meds * i + j for j in range(n_bins_meds)]
        new_med = MedSpec(
            target_var_index=new_target_var_index,
            impact_coeff=new_impact_coeff,
            impact_delay=new_impact_delay,
            is_beneficial=new_is_beneficial,
            physician_knowledge=new_physician_knowledge,
            codes=new_codes)
        meds_all.append(new_med)
        meds_vars.append(new_med)

    # printing
    if verbose >= 1:
        print("\nRandomly generated medications: ")
        for i, var in enumerate(meds_health):
            print(
                f"\tMedication m_{i}\n"
                f"\t\tTarget: Health\n"
                f"\t\tImpact coefficient: {var.impact_coeff}\n"
                f"\t\tImpact delay: {var.impact_delay}\n"
                f"\t\tIs beneficial: {var.is_beneficial}\n"
                f"\t\tPhysician knowledege: {var.physician_knowledge}")
        for i, var in enumerate(meds_vars):
            print(
                f"\tMedication m_{i + n_meds_health}\n"
                f"\t\tTarget: Variable v_{var.target_var_index}\n"
                f"\t\tImpact coefficient: {var.impact_coeff}\n"
                f"\t\tImpact delay: {var.impact_delay}\n"
                f"\t\tIs beneficial: {var.is_beneficial}\n"
                f"\t\tPhysician knowledege: {var.physician_knowledge}")

    # randomly generate interactions between medications
    n_meds = n_meds_health + n_meds_vars
    possible_interaction_pairs = []
    for i in range(n_meds):
        for j in range(i + 1, n_meds):
            possible_interaction_pairs.append((i, j))
    np.random.shuffle(possible_interaction_pairs)
    assert n_interactions <= len(possible_interaction_pairs)
    interactions = []
    np_interaction_grid = np.ones((n_meds, n_meds, 2), dtype=int)
    for _ in range(n_interactions):
        interaction_pair = possible_interaction_pairs.pop()
        if meds_all[interaction_pair[0]].is_beneficial == meds_all[interaction_pair[1]].is_beneficial:
            possible_interaction_coeffs = interaction_coeffs_aligned
        else:
            possible_interaction_coeffs = interaction_coeffs_opposing
        interaction_coeffs = possible_interaction_coeffs[np.random.randint(len(possible_interaction_coeffs))]

        # TODO temporary modification to hard-code simple medication specs
        print("Warning! You're manually specifying the interactions instead of randomly generating them!")
        interaction_coeffs = (-1, -1)
        # TODO remove this later

        new_interaction = InteractionSpec(
            med_index_a=interaction_pair[0],
            coeff_a=interaction_coeffs[0],
            med_index_b=interaction_pair[1],
            coeff_b=interaction_coeffs[1])
        interactions.append(new_interaction)
        np_interaction_grid[interaction_pair] = interaction_coeffs

    # printing
    if verbose >= 1:
        print("\nRandomly generated interactions: ")
        print("\t\t\t", end="")
        for i in range(n_meds):
            print(f"m_{i}({'+' if meds_all[i].is_beneficial else '-'}) ".rjust(9), end="")
        print("")
        for i in range(np_interaction_grid.shape[0]):
            print(f"\tm_{i}({'+' if meds_all[i].is_beneficial else '-'})\t", end="")
            for j in range(np_interaction_grid.shape[1]):
                if np.array_equal(np_interaction_grid[i, j], [1, 1]):
                    if i >= j:
                        print("         ", end="")
                    else:
                        print("(------) ", end="")
                else:
                    print(
                        f"("
                        f"{str(np_interaction_grid[i, j, 0]).rjust(2)}, "
                        f"{str(np_interaction_grid[i, j, 1]).rjust(2)}) ", end="")
            print("")

    # ==========================================================================
    # per-patient simulation
    # ==========================================================================

    patients = []
    patient_morts = []
    mort_times = []

    # non-diag elmeents list outcomes for all patients recieving both m_i and m_j
    # diag elements list outcomes for all patients recieving m_i
    med_morts_pair = [[[] for _ in range(n_meds)] for _ in range(n_meds)]

    while len(patients) < n_patients:
        patient_index = len(patients)

        # fixed patient parameter generation
        patient_var_bounds = []
        for _ in range(n_vars):
            patient_var_bounds.append((np.random.uniform(-1, 0), np.random.uniform(0, 1)))

        # patient state setup
        patient_health = np.random.normal() + patient_health_offset
        patient_vars = [0] * n_vars
        patient_delay_line = deque(
            [
                [0, [0] * n_vars]
                for _
                in range(max_med_effect_delay + 1)
            ],
            maxlen=(max_med_effect_delay + 1))

        # randomly determine which medications can be administered
        if med_subset_per_patient:
            found = False
            patient_med_mask = None
            while not found:
                patient_med_mask = np.random.choice([True, False], size=n_meds)
                found = np.any(patient_med_mask)
        else:
            patient_med_mask = np.full(shape=n_meds, fill_value=True)

        # calculate a mask to make appending to med_morts_pair convenient
        patient_med_mask_pair = np.logical_and(
            patient_med_mask.reshape((1, n_meds)), patient_med_mask.reshape((n_meds, 1)))

        # printing
        if verbose >= 2:
            print(f"\nPatient {patient_index}")

        # ======================================================================
        # iterating over timesteps
        # ======================================================================

        patient = []
        patient_mort = False
        for t in range(n_timesteps_total):
            visit = []

            # make physiological observations
            visible_var_index = -1
            for var_index, var in enumerate(vars_all):
                if var.is_visible:
                    visible_var_index += 1
                    discretized_var = discretize(patient_vars[var_index], -1, 1, n_bins_vars)
                    visit.append(discretized_var + visible_var_index * n_bins_vars)
                    if verbose >= 2:
                        print(
                            f"\tT={t}: variable v_{var_index} observation: \n"
                            f"\t\tvalue: {patient_vars[var_index]}\n"
                            f"\t\tdiscretized into bin: {discretized_var + 1}/{n_bins_vars}")

            # determine chances of administrating each medication
            med_dosage_chances = [0.5] * n_meds
            for med_index, med in enumerate(meds_all):
                if med.target_var_index is None:
                    continue
                target_var = vars_all[med.target_var_index]
                if med.target_var_index is not None and target_var.is_visible:
                    target_var_unhealthiness = (patient_vars[med.target_var_index] + 1) / 2
                    if target_var.is_beneficial:
                        target_var_unhealthiness = 1 - target_var_unhealthiness
                    delta = (
                            physician_knowledge_sf
                            * 0.5
                            * target_var_unhealthiness
                            * med.physician_knowledge
                            * target_var.physician_knowledge)
                    if not med.is_beneficial:
                        delta *= -1
                    med_dosage_chances[med_index] += delta

            # randomly select medications to administer, and their dosages
            med_dosages = [0] * n_meds
            meds_admined = set()
            for med_index, med_dosage_chance in enumerate(med_dosage_chances):
                if not patient_med_mask[med_index]:
                    continue
                if np.random.uniform(0, 1) < med_dosage_chance:
                    med_dosages[med_index] = np.random.uniform(0, 1)
                    meds_admined.add(med_index)
                    discretized_dosage = discretize(med_dosages[med_index], 0, 1, n_bins_meds)
                    visit.append(discretized_dosage + med_index * n_bins_meds + n_vars_visible * n_bins_vars)

            # add medical codes to patient's history (no more medical codes until next timestep)
            patient.append(visit)

            # determine if any interactions occur, and apply their coefficients
            med_interaction_factors = [1] * n_meds
            for interaction in interactions:
                if (interaction.med_index_a in meds_admined) and (interaction.med_index_b in meds_admined):
                    med_interaction_factors[interaction.med_index_a] *= interaction.coeff_a
                    med_interaction_factors[interaction.med_index_b] *= interaction.coeff_b

            # apply health updates due to first-order variables
            # and variable updates due to second-order variables
            for var_index, var in enumerate(vars_all):
                var_impact = var.impact_coeff * patient_vars[var_index]
                if var.target_var_index is None:
                    patient_health += var_impact
                else:
                    patient_vars[var.target_var_index] += var_impact

            # apply each medication to the patient's state
            for med_index in range(n_meds):
                med = meds_all[med_index]
                med_impact = (
                        med.impact_coeff
                        * med_dosages[med_index]
                        * med_interaction_factors[med_index]
                        * (1 + med_impact_randomness * np.random.normal()))

                discretized_dosage = discretize(med_dosages[med_index], 0, 1, n_bins_meds)
                if verbose >= 2:
                    print(
                        f"\tT={t}: medication m_{med_index}: \n"
                        f"\t\tdosage chance: {med_dosage_chances[med_index]}\n"
                        f"\t\tdosage administered: "
                        f"{'None' if med_dosages[med_index] == 0 else med_dosages[med_index]}\n"
                        f"\t\tdiscretized into bin: "
                        f"{'None' if med_dosages[med_index] == 0 else f'{discretized_dosage + 1}/{n_bins_meds}'}\n"
                        f"\t\timpact coeff: {med.impact_coeff}\n"
                        f"\t\tinteraction factor: {med_interaction_factors[med_index]}\n"
                        f"\t\tnet effect: "
                        f"{'health' if med.target_var_index is None else f'variable {med.target_var_index}'} "
                        f"increased by {med_impact} in {med.impact_delay} timesteps")

                if med.impact_delay == 0:
                    if med.target_var_index is None:
                        patient_health += med_impact
                    else:
                        patient_vars[med.target_var_index] += med_impact
                else:
                    if med.target_var_index is None:
                        patient_delay_line[med.impact_delay][0] += med_impact
                    else:
                        patient_delay_line[med.impact_delay][1][med.target_var_index] += med_impact

            # apply effects of delayed medications
            delta_health, delta_vars = patient_delay_line.popleft()
            patient_delay_line.append([0, [0] * n_vars])
            patient_health += delta_health
            for i in range(n_vars):
                patient_vars[i] += delta_vars[i]

            # printing
            if verbose >= 2:
                print(
                    f"\tT={t}: patient vars before var clamping: \n"
                    f"\t\tvars: {patient_vars}")

            # clamp variable values
            for var_index in range(n_vars):
                patient_vars[var_index] = min(
                    max(patient_var_bounds[var_index][0], patient_vars[var_index]), patient_var_bounds[var_index][1])

            # printing
            if verbose >= 2:
                print(
                    f"\tT={t}: patient state after var clamping: \n"
                    f"\t\thealth: {patient_health}\n"
                    f"\t\tvars: {patient_vars}\n"
                    f"\t\tdelay line:")
                for i in range(max_med_effect_delay):
                    print(f"\t\t\t{patient_delay_line[i]}")

            # determine patient mortality
            mort_prob = 1 / (1 + math.exp(patient_health))
            patient_mort = np.random.binomial(1, mort_prob)

            # printing
            if verbose >= 2:
                print(f"\tT={t}: patient mortality: \n"
                      f"\t\tchance: {mort_prob}\n"
                      f"\t\toutcome: {'mortality' if patient_mort else 'survival'}")

            # break in case of patient mortality
            if patient_mort:
                mort_times.append(t)
                break

        # check to make sure there's at least one medical code
        if max([len(visit) for visit in patient]) == 0:
            if verbose >= 2:
                print("Patient doesn't have any medical codes. Discarding.")
            continue

        # check to make sure there's at least one medical code
        if max([len(visit) for visit in patient[:n_timesteps_trunc]]) == 0:
            if verbose >= 2:
                print(f"Patient doesn't have any medical codes in first {n_timesteps_trunc} timesteps. Discarding.")
            continue

        # record patient
        patients.append(patient[:n_timesteps_trunc])
        patient_morts.append(patient_mort)
        for med_index_i in range(n_meds):
            for med_index_j in range(n_meds):
                if patient_med_mask_pair[med_index_i, med_index_j]:
                    med_morts_pair[med_index_i][med_index_j].append(patient_mort)

        # printing
        if verbose >= 2:
            print(f"\tPatient history: {patient}")

    # printing
    if verbose >= 1:
        print(f"\nOverall mortality rate: {np.mean(patient_morts)} ({np.sum(patient_morts)} out of {n_patients})")
        print(f"\nTotal number of medical codes: {n_vars_visible * n_bins_vars + n_meds * n_bins_meds}")
        # plt.hist(mort_times)
        # plt.show()
        unique, counts = np.unique(mort_times, return_counts=True)
        mort_counts = dict(zip(unique, counts))
        for i in range(n_timesteps_trunc):
            print(f"\nMortalities on day {i}: {mort_counts[i]}/{n_patients} ({mort_counts[i]/n_patients})")
        if n_patients <= 30:
            print("")
            print("All patients:")
            for patient in patients:
                print(patient)
            print("All targets:")
            for target in patient_morts:
                print(target)
        np_med_mort_rates_pair = np.zeros((n_meds, n_meds))
        for med_index_i in range(n_meds):
            for med_index_j in range(n_meds):
                np_med_mort_rates_pair[med_index_i, med_index_j] = np.mean(
                    med_morts_pair[med_index_i][med_index_j])
        print("")
        print(pd.DataFrame(
            data=np_med_mort_rates_pair,
            columns=[f"m_{i}" for i in range(n_meds)],
            index=[f"m_{i}" for i in range(n_meds)]))

    # for RNN
    all_data = pd.DataFrame(data={'codes': patients}, columns=['codes']).reset_index()
    all_targets = pd.DataFrame(data={'target': patient_morts}, columns=['target']).reset_index()
    data_train, data_val_test = train_test_split(all_data, train_size=train_proportion, random_state=12345)
    target_train, target_val_test = train_test_split(all_targets, train_size=train_proportion, random_state=12345)
    val_proportion_adjusted = val_proportion / (1 - train_proportion)
    data_val, data_test = train_test_split(data_val_test, train_size=val_proportion_adjusted, random_state=12345)
    target_val, target_test = train_test_split(target_val_test, train_size=val_proportion_adjusted, random_state=12345)
    data_train.sort_index().to_pickle(out_directory + '/data_train.pkl')
    data_val.sort_index().to_pickle(out_directory + '/data_val.pkl')
    data_test.sort_index().to_pickle(out_directory + '/data_test.pkl')
    target_train.sort_index().to_pickle(out_directory + '/target_train.pkl')
    target_val.sort_index().to_pickle(out_directory + '/target_val.pkl')
    target_test.sort_index().to_pickle(out_directory + '/target_test.pkl')

    # parameters and information pickles
    with open(out_directory + '/interact_params.pkl', "wb") as writefile:
        pickle.dump(interact_params, writefile)
    with open(out_directory + '/vars_all.pkl', "wb") as writefile:
        pickle.dump(vars_all, writefile)
    with open(out_directory + '/meds_all.pkl', "wb") as writefile:
        pickle.dump(meds_all, writefile)
    with open(out_directory + '/interactions.pkl', "wb") as writefile:
        pickle.dump(interactions, writefile)

    # pickled dictionary of string lookup table for medical codes
    dictionary = dict()
    medical_code = 0
    for var_index in range(n_vars_visible):
        for level in range(n_bins_vars):
            dictionary.update({medical_code: f"v_{var_index} at level {level + 1}/{n_bins_vars}"})
            medical_code += 1
    for med_index in range(n_meds):
        for dosage in range(n_bins_meds):
            dictionary.update({medical_code: f"m_{med_index} at dosage {dosage + 1}/{n_bins_meds}"})
            medical_code += 1
    with open(out_directory + '/dictionary.pkl', "wb") as writefile:
        pickle.dump(dictionary, writefile)


if __name__ == '__main__':
    main(
        out_directory=sys.argv[1],
        n_patients=int(sys.argv[2]),
        n_timesteps_total=int(sys.argv[3]),
        n_timesteps_trunc=int(sys.argv[4]),
        train_proportion=float(sys.argv[5]),
        val_proportion=float(sys.argv[6]),
        verbose=int(sys.argv[7]))
