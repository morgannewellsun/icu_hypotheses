
# Outputs ne dataframe (saved as CSV), where each row represets a medical event
# - diagnosis, procedure, or medication prescription. Each type of event -
# diagnoses, procedures, and medications - is chronologically sorted w.r.t
# other events of the same type. However, there is no information regarding
# the relative order of these events. For example, it is possible to determine
# that procedure B was done after procedure A. But we do not know if procedure
# B was done before or after diagnosis C.


import argparse
from datetime import datetime
import os
import warnings

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def convert_to_icd9(dx_str):
    if dx_str.startswith('E'):
        if len(dx_str) > 4:
            return dx_str[:4] + '.' + dx_str[4:]
        else:
            return dx_str
    else:
        if len(dx_str) > 3:
            return dx_str[:3] + '.' + dx_str[3:]
        else:
            return dx_str


def convert_to_3digit_icd9(dx_str):
    if dx_str.startswith('E'):
        if len(dx_str) > 4:
            return dx_str[:4]
        else:
            return dx_str
    else:
        if len(dx_str) > 3:
            return dx_str[:3]
        else:
            return dx_str


def build_patient_maps(patients_filepath):
    patient_gender_map = {}
    patient_dob_map = {}
    patient_mortality_map = {}
    with open(patients_filepath, 'r') as file:
        file.readline()
        for line in file:
            tokens = line.strip().split(',')
            patient_id = int(tokens[1])
            patient_gender_map[patient_id] = tokens[2].replace('"', "")
            patient_dob_map[patient_id] = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
            patient_mortality_map[patient_id] = True if (len(tokens[5]) > 0) else False
    return patient_gender_map, patient_dob_map, patient_mortality_map


def build_admission_maps(admission_filepath):
    patient_admissions_unsorted_map = {}
    admission_datetime_map = {}
    with open(admission_filepath, 'r') as file:
        file.readline()
        for line in file:
            tokens = line.strip().split(',')
            patient_id = int(tokens[1])
            admission_id = int(tokens[2])
            admission_datetime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
            admission_datetime_map[admission_id] = admission_datetime
            if patient_id in patient_admissions_unsorted_map:
                patient_admissions_unsorted_map[patient_id].append(admission_id)
            else:
                patient_admissions_unsorted_map[patient_id] = [admission_id]
    patient_admissions_map = {}
    for patient_id, admission_ids in patient_admissions_unsorted_map.items():
        admission_ids_sorted = sorted(admission_ids, key=(lambda admission_id: admission_datetime_map[admission_id]))
        patient_admissions_map.update({patient_id: admission_ids_sorted})
    return patient_admissions_map, admission_datetime_map


def build_diagnoses_maps(diagnoses_filepath):
    admission_diagnoses_map = {}
    admission_diagnoses_trunc_map = {}
    with open(diagnoses_filepath, 'r') as file:
        file.readline()
        for line in file:  # note that diagnoses are already sorted by order
            tokens = line.strip().split(',')
            adm_id = int(tokens[2])
            dx_str = 'D_' + convert_to_icd9(tokens[4][1:-1])
            dx_str_3digit = 'D_' + convert_to_3digit_icd9(tokens[4][1:-1])
            if adm_id in admission_diagnoses_map:
                admission_diagnoses_map[adm_id].append(dx_str)
            else:
                admission_diagnoses_map[adm_id] = [dx_str]
            if adm_id in admission_diagnoses_trunc_map:
                admission_diagnoses_trunc_map[adm_id].append(dx_str_3digit)
            else:
                admission_diagnoses_trunc_map[adm_id] = [dx_str_3digit]
    return admission_diagnoses_map, admission_diagnoses_trunc_map


def build_procedures_maps(procedures_filepath):
    admission_procedures_map = {}
    admission_procedures_trunc_map = {}
    with open(procedures_filepath, 'r') as file:
        file.readline()
        for line in file:  # note that procedures are already sorted by order
            tokens = line.strip().split(',')
            adm_id = int(tokens[2])
            dx_str = 'P_' + convert_to_icd9(tokens[4][1:-1])
            dx_str_3digit = 'P_' + convert_to_3digit_icd9(tokens[4][1:-1])
            if adm_id in admission_procedures_map:
                admission_procedures_map[adm_id].append(dx_str)
            else:
                admission_procedures_map[adm_id] = [dx_str]
            if adm_id in admission_procedures_trunc_map:
                admission_procedures_trunc_map[adm_id].append(dx_str_3digit)
            else:
                admission_procedures_trunc_map[adm_id] = [dx_str_3digit]
    return admission_procedures_map, admission_procedures_trunc_map


def build_medications_maps(prescriptions_filepath):
    admission_medications_map = {}
    admission_medications_trunc_map = {}
    prescriptions_df = pd.read_csv(
        prescriptions_filepath, usecols=["HADM_ID", "STARTDATE", "ENDDATE", "NDC"], dtype={"NDC": str})
    prescriptions_df = prescriptions_df[prescriptions_df["NDC"].str.len() == 11]  # 11-character NDC codes
    for admission_id, admission_df in prescriptions_df.groupby("HADM_ID", sort=False):
        admission_df = admission_df.sort_values("ENDDATE", kind="mergesort")  # stable sort not available for >1 key
        admission_df = admission_df.sort_values("STARTDATE", kind="mergesort")
        admission_ndc_11_digit = "M_" + admission_df["NDC"]
        admission_ndc_9_digit = admission_ndc_11_digit.str.slice(0, 11)
        admission_medications_map.update({admission_id: list(admission_ndc_11_digit)})
        admission_medications_trunc_map.update({admission_id: list(admission_ndc_9_digit)})
    return admission_medications_map, admission_medications_trunc_map


def main(
        patients_filepath,
        admissions_filepath,
        diagnoses_filepath,
        procedures_filepath,
        prescriptions_filepath,
        out_directory):

    # -------------------------------------------------------------------------
    # build raw data maps
    # -------------------------------------------------------------------------
    print("[INFO] Building raw data maps")
    patient_gender_map, patient_dob_map, patient_mortality_map = build_patient_maps(patients_filepath)
    patient_admissions_map, admission_datetime_map = build_admission_maps(admissions_filepath)
    admission_diagnoses_map, admission_diagnoses_trunc_map = build_diagnoses_maps(diagnoses_filepath)
    admission_procedures_map, admission_procedures_trunc_map = build_procedures_maps(procedures_filepath)
    admission_medications_map, admission_medications_trunc_map = build_medications_maps(prescriptions_filepath)

    # -------------------------------------------------------------------------
    # construct event-per-row dataframe
    # -------------------------------------------------------------------------

    print("[INFO] Collecting event-level data")

    # patient-level data
    patient_ids = sorted(list(patient_mortality_map.keys()))
    patient_dobs = []
    patient_genders = []
    patient_mortalities = []
    patient_event_counts = []
    patient_diagnoses_counts = []
    patient_procedures_counts = []
    patient_medications_counts = []
    patient_admission_counts = []

    # admission-level data
    admission_ids = []
    admission_datetimes = []
    admission_event_counts = []
    admission_diagnoses_counts = []
    admission_procedures_counts = []
    admission_medications_counts = []

    # event-level data
    event_types = []
    event_codes_full = []
    event_codes_trunc = []

    # collect data
    for patient_id in patient_ids:
        patient_dobs.append(patient_dob_map[patient_id])
        patient_genders.append(patient_gender_map[patient_id])
        patient_mortalities.append([patient_mortality_map[patient_id]])
        patient_admission_ids = patient_admissions_map[patient_id]
        patient_admission_counts.append(len(patient_admission_ids))
        patient_event_count = 0
        patient_diagnoses_count = 0
        patient_procedures_count = 0
        patient_medications_count = 0
        for admission_id in patient_admission_ids:
            admission_ids.append(admission_id)
            admission_datetimes.append(admission_datetime_map[admission_id])
            admission_diagnoses = admission_diagnoses_map.get(admission_id, [])
            admission_procedures = admission_procedures_map.get(admission_id, [])
            admission_medications = admission_medications_map.get(admission_id, [])
            admission_diagnoses_trunc = admission_diagnoses_trunc_map.get(admission_id, [])
            admission_procedures_trunc = admission_procedures_trunc_map.get(admission_id, [])
            admission_medications_trunc = admission_medications_trunc_map.get(admission_id, [])
            admission_event_count = len(admission_diagnoses) + len(admission_procedures) + len(admission_medications)
            if admission_event_count == 0:
                warnings.warn(
                    f"[WARNING] Patient ID {patient_id} with admission ID {admission_id} contains zero events.")
            admission_event_counts.append(admission_event_count)
            patient_event_count += admission_event_count
            admission_diagnoses_counts.append(len(admission_diagnoses))
            patient_diagnoses_count += len(admission_diagnoses)
            admission_procedures_counts.append(len(admission_procedures))
            patient_procedures_count += len(admission_procedures)
            admission_medications_counts.append(len(admission_medications))
            patient_medications_count += len(admission_medications)
            event_types.extend(
                ['D'] * len(admission_diagnoses)
                + ['P'] * len(admission_procedures)
                + ['M'] * len(admission_medications))
            event_codes_full.extend(
                admission_diagnoses + admission_procedures + admission_medications)
            event_codes_trunc.extend(
                admission_diagnoses_trunc + admission_procedures_trunc + admission_medications_trunc)
        patient_event_counts.append(patient_event_count)
        patient_diagnoses_counts.append(patient_diagnoses_count)
        patient_procedures_counts.append(patient_procedures_count)
        patient_medications_counts.append(patient_medications_count)

    # repeat patient-level data for each admission
    patient_ids = np.repeat(patient_ids, patient_admission_counts)
    patient_dobs = np.repeat(patient_dobs, patient_admission_counts)
    patient_genders = np.repeat(patient_genders, patient_admission_counts)
    patient_mortalities = np.repeat(patient_mortalities, patient_admission_counts)
    patient_admission_counts_repeated = np.repeat(patient_admission_counts, patient_admission_counts)
    patient_event_counts_repeated = np.repeat(patient_event_counts, patient_admission_counts)
    patient_diagnoses_counts_repeated = np.repeat(patient_diagnoses_counts, patient_admission_counts)
    patient_procedures_counts_repeated = np.repeat(patient_procedures_counts, patient_admission_counts)
    patient_medications_counts_repeated = np.repeat(patient_medications_counts, patient_admission_counts)

    # repeat patient- and admission-level data for each event
    patient_ids = np.repeat(patient_ids, admission_event_counts)
    patient_dobs = np.repeat(patient_dobs, admission_event_counts)
    patient_genders = np.repeat(patient_genders, admission_event_counts)
    patient_mortalities = np.repeat(patient_mortalities, admission_event_counts)
    patient_admission_counts_repeated = np.repeat(patient_admission_counts_repeated, admission_event_counts)
    patient_event_counts_repeated = np.repeat(patient_event_counts_repeated, admission_event_counts)
    patient_diagnoses_counts_repeated = np.repeat(patient_diagnoses_counts_repeated, admission_event_counts)
    patient_procedures_counts_repeated = np.repeat(patient_procedures_counts_repeated, admission_event_counts)
    patient_medications_counts_repeated = np.repeat(patient_medications_counts_repeated, admission_event_counts)
    admission_ids = np.repeat(admission_ids, admission_event_counts)
    admission_datetimes = np.repeat(admission_datetimes, admission_event_counts)
    admission_event_counts_repeated = np.repeat(admission_event_counts, admission_event_counts)
    admission_diagnoses_counts_repeated = np.repeat(admission_diagnoses_counts, admission_event_counts)
    admission_procedures_counts_repeated = np.repeat(admission_procedures_counts, admission_event_counts)
    admission_medications_counts_repeated = np.repeat(admission_medications_counts, admission_event_counts)

    # construct the dataframe
    mimic_df = pd.DataFrame(data={
        "patient_id": patient_ids,  # int
        "patient_dob": patient_dobs,  # datetime
        "patient_gender": patient_genders,  # str, 'M' or 'F'
        "patient_mortality": patient_mortalities,  # bool
        "patient_admission_count": patient_admission_counts_repeated,  # int
        "patient_event_count": patient_event_counts_repeated,  # int
        "patient_diagnoses_count": patient_diagnoses_counts_repeated,  # int
        "patient_procedures_count": patient_procedures_counts_repeated,  # int
        "patient_medications_count": patient_medications_counts_repeated,  # int
        "admission_id": admission_ids,  # int
        "admission_datetime": admission_datetimes,  # datetime
        "admission_event_count": admission_event_counts_repeated,  # int
        "admission_diagnoses_count": admission_diagnoses_counts_repeated,  # int
        "admission_procedures_count": admission_procedures_counts_repeated,  # int
        "admission_medications_count": admission_medications_counts_repeated,  # int
        "event_type": event_types,  # str, 'D' or 'P' or 'M'
        "event_code_full": event_codes_full,  # str
        "event_code_trunc": event_codes_trunc,  # str
    })

    # -------------------------------------------------------------------------
    # build additional features
    # -------------------------------------------------------------------------

    print("[INFO] Building additional features")

    # calculate time since previous admission
    admission_timedeltas_days_since_prev = []
    for patient_id, patient_df in mimic_df.groupby("patient_id", sort=False):
        patient_admission_datetimes = patient_df["admission_datetime"].unique()
        patient_admission_timedeltas_days_since_prev = (
            (patient_admission_datetimes[1:] - patient_admission_datetimes[:-1]) / np.timedelta64(1, 'D'))
        patient_admission_timedeltas_days_since_prev = np.concatenate(
            [[np.inf], patient_admission_timedeltas_days_since_prev])
        admission_timedeltas_days_since_prev.append(patient_admission_timedeltas_days_since_prev)
    admission_timedeltas_days_since_prev = np.concatenate(admission_timedeltas_days_since_prev)
    admission_timedeltas_days_since_prev = np.repeat(admission_timedeltas_days_since_prev, admission_event_counts)
    mimic_df["admission_timedelta_days_since_prev"] = admission_timedeltas_days_since_prev

    # count number of occurences for all events
    event_count_mapper_full = mimic_df["event_code_full"].value_counts().to_dict()
    mimic_df["event_code_full_count"] = mimic_df["event_code_full"].map(event_count_mapper_full)
    event_count_mapper_trunc = mimic_df["event_code_trunc"].value_counts().to_dict()
    mimic_df["event_code_trunc_count"] = mimic_df["event_code_trunc"].map(event_count_mapper_trunc)

    # -------------------------------------------------------------------------
    # profile data characteristics and generate graphics
    # -------------------------------------------------------------------------

    print("[INFO] Profiling data")

    # distribution of event counts
    event_counts_diagnoses_full = []
    event_counts_procedures_full = []
    event_counts_medications_full = []
    for event_code, event_count in event_count_mapper_full.items():
        if event_code[0] == "D":
            event_counts_diagnoses_full.append(event_count)
        elif event_code[0] == "P":
            event_counts_procedures_full.append(event_count)
        elif event_code[0] == "M":
            event_counts_medications_full.append(event_count)
        else:
            warnings.warn(f"[WARNING] Unknown event type {event_code[0]} encountered in event code {event_code}.")
    event_counts_diagnoses_trunc = []
    event_counts_procedures_trunc = []
    event_counts_medications_trunc = []
    for event_code, event_count in event_count_mapper_trunc.items():
        if event_code[0] == "D":
            event_counts_diagnoses_trunc.append(event_count)
        elif event_code[0] == "P":
            event_counts_procedures_trunc.append(event_count)
        elif event_code[0] == "M":
            event_counts_medications_trunc.append(event_count)
        else:
            warnings.warn(f"[WARNING] Unknown event type {event_code[0]} encountered in event code {event_code}.")
    for name, event_counts in [
        ("diagnoses_full", event_counts_diagnoses_full),
        ("procedures_full", event_counts_procedures_full),
        ("medications_full", event_counts_medications_full),
        ("diagnoses_trunc", event_counts_diagnoses_trunc),
        ("procedures_trunc", event_counts_procedures_trunc),
        ("medications_trunc", event_counts_medications_trunc),
    ]:
        cumulative_event_distribution = np.cumsum(sorted(event_counts, reverse=True))
        cumulative_event_distribution = cumulative_event_distribution / cumulative_event_distribution[-1]
        plt.figure(figsize=(20, 10))
        plt.plot(range(1, len(cumulative_event_distribution) + 1), cumulative_event_distribution)
        plt.xlim((1, len(cumulative_event_distribution) + 1))
        plt.ylim((0, 1))
        thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
        plt.xticks(np.searchsorted(cumulative_event_distribution, thresholds) + 1)
        plt.yticks(thresholds)
        plt.grid(visible=True, which="major")
        plt.xlabel("number of codes used")
        plt.ylabel("% of total event instances represented")
        title = f"cumulative_event_count_{name}"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_directory, f"{title}.png"))

    # count number of patients with at least some minimum number of admissions, diagnoses, procedures
    for name, counts in [
        ("admissions", patient_admission_counts),
        ("diagnoses", patient_diagnoses_counts),
        ("procedures", patient_procedures_counts),
        ("medications", patient_medications_counts),
    ]:
        fig, ax = plt.subplots(figsize=(10, 10))
        hist, bin_edges = np.histogram(
            counts,
            bins=np.arange(0.5, max(counts) + 0.51))
        data_x = range(1, int(max(counts)) + 1)
        data_y = np.cumsum(hist[::-1])[::-1]
        ax.plot(data_x, data_y, marker="o", linestyle="none")
        ax.set_xlim(0, max(counts) + 1)
        ax.set_yscale("log")
        xticks = [data_x[0]]
        yticks = [data_y[0]]
        for xvalue, yvalue in zip(data_x[1:], data_y[1:]):
            if yvalue < 0.85 * yticks[-1]:
                yticks.append(yvalue)
                if xvalue > (max(counts) / 50) + xticks[-1]:
                    xticks.append(xvalue)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.tick_params(axis='x', labelrotation=90)
        ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.grid(visible=True, which="major")
        ax.set_xlabel(f"minimum (inclusive) {name} per patient")
        ax.set_ylabel("number of patients (log scale)")
        title = f"cumulative_patient_counts_min_{name}"
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(out_directory, f"{title}.png"))

    # -------------------------------------------------------------------------
    # save dataframe
    # -------------------------------------------------------------------------

    print("[INFO] Saving dataframe to .csv file")
    patient_columns = []
    admission_columns = []
    event_columns = []
    for column_name in mimic_df.columns:
        if column_name[:7] == "patient":
            patient_columns.append(column_name)
        elif column_name[:9] == "admission":
            admission_columns.append(column_name)
        elif column_name[:5] == "event":
            event_columns.append(column_name)
        else:
            warnings.warn(
                f"[WARNING] Column name '{column_name}' does not start with 'patient', 'admission', or 'event'")
    patient_columns = sorted(patient_columns)
    admission_columns = sorted(admission_columns)
    event_columns = sorted(event_columns)
    all_columns = patient_columns + admission_columns + event_columns
    mimic_df[all_columns].to_csv(os.path.join(out_directory, "mimic_df.csv"))


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--patients_filepath', type=str, required=True)
    parser.add_argument('--admissions_filepath', type=str, required=True)
    parser.add_argument('--diagnoses_filepath', type=str, required=True)
    parser.add_argument('--procedures_filepath', type=str, required=True)
    parser.add_argument('--prescriptions_filepath', type=str, required=True)
    parser.add_argument('--out_directory', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(
        args.patients_filepath,
        args.admissions_filepath,
        args.diagnoses_filepath,
        args.procedures_filepath,
        args.prescriptions_filepath,
        args.out_directory)
