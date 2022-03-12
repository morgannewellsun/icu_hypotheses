
"""
This script links DrugBank codes to corresponding NDC codes by comparing names of drugs.
The output is a CSV file mapping DrugBank codes to NDC codes.
Most DrugBank codes are not able to be mapped.
The mapping may be many-to-one.
"""

import argparse
import os

import numpy as np
import pandas as pd


def build_ndc_names_maps(prescriptions_filepath):
    ndc_codes = []
    ndc_names = []
    prescriptions_df = pd.read_csv(
        prescriptions_filepath, usecols=["DRUG", "DRUG_NAME_POE", "DRUG_NAME_GENERIC", "NDC"], dtype={"NDC": str})
    prescriptions_df = prescriptions_df[prescriptions_df["NDC"].str.len() == 11]  # only keep 11-character NDC codes
    for col_name in ["DRUG", "DRUG_NAME_POE", "DRUG_NAME_GENERIC"]:
        prescriptions_df[col_name] = prescriptions_df[col_name].str.lower()
        prescriptions_df[col_name] = prescriptions_df[col_name].str.replace(r'[^a-zA-Z\s]', '', regex=True)
        prescriptions_df[col_name] = prescriptions_df[col_name].str.replace(r'\s+', ' ', regex=True)
        prescriptions_df[col_name] = prescriptions_df[col_name].str.replace(r'\s$', '', regex=True)
        prescriptions_df[col_name] = prescriptions_df[col_name].str.replace(r'^\s', '', regex=True)
    for ndc, ndc_df in sorted(  # sort by descending frequency of code
            prescriptions_df.groupby("NDC"),
            key=(lambda ndc_group: len(ndc_group[1])),
            reverse=True):
        names = []
        for col_name in ["DRUG", "DRUG_NAME_POE", "DRUG_NAME_GENERIC"]:
            name_value_counts = ndc_df[col_name][~ndc_df[col_name].isnull()].value_counts()
            if len(name_value_counts) > 0:
                names.append(name_value_counts.index[0])  # choose most commonly used name
        ndc_codes.append(ndc)
        ndc_names.append(names)
    return ndc_codes, ndc_names


def build_drugbank_names_maps(drugbank_vocabulary_filepath):
    drugbank_ids = []
    drugbank_names = []
    drugbank_vocab_df = pd.read_csv(
        drugbank_vocabulary_filepath, usecols=["DrugBank ID", "Common name", "Synonyms"])
    for col_name in ["Common name", "Synonyms"]:
        drugbank_vocab_df[col_name] = drugbank_vocab_df[col_name].str.lower()
        drugbank_vocab_df[col_name] = drugbank_vocab_df[col_name].str.replace(r'[^a-zA-Z\s\|]', '', regex=True)
        drugbank_vocab_df[col_name] = drugbank_vocab_df[col_name].str.replace(r'\s+', ' ', regex=True)
        drugbank_vocab_df[col_name] = drugbank_vocab_df[col_name].str.replace(r'\s$', '', regex=True)
        drugbank_vocab_df[col_name] = drugbank_vocab_df[col_name].str.replace(r'^\s', '', regex=True)
    for row in drugbank_vocab_df.iterrows():
        drugbank_id, common_name, synonyms = row[1]
        names = [common_name]
        if synonyms is not np.nan:
            names.extend(synonyms.split(" | "))
        drugbank_ids.append(drugbank_id)
        drugbank_names.append(names)
    return drugbank_ids, drugbank_names


def main(
        prescriptions_filepath,
        drugbank_vocabulary_filepath,
        drugbank_interactions_filepath,
        out_directory):

    # -------------------------------------------------------------------------
    # build raw data maps
    # -------------------------------------------------------------------------
    print("[INFO] Building raw data lists")
    ndc_codes, ndc_name_lists = build_ndc_names_maps(prescriptions_filepath)
    drugbank_ids, drugbank_name_lists = build_drugbank_names_maps(drugbank_vocabulary_filepath)

    # -------------------------------------------------------------------------
    # map drugbank to ndc
    # -------------------------------------------------------------------------
    print("[INFO] Mapping DrugBank IDs to NDC codes")
    drugbank_matched_ndc_codes = []
    drugbank_matched_ndc_name_lists = []
    for drugbank_idx, drugbank_names in enumerate(drugbank_name_lists):
        if (drugbank_idx > 0) and (drugbank_idx % 1000 == 0):
            print(f"[INFO] Progress: {drugbank_idx}/{len(drugbank_ids)} DrugBank IDs mapped to NDC codes")
        matched_ndc_code = None
        matched_ndc_name_list = None
        found = False
        for ndc_idx, ndc_names in enumerate(ndc_name_lists):
            for ndc_name in ndc_names:
                for drugbank_name in drugbank_names:
                    if ndc_name == drugbank_name:
                        matched_ndc_code = ndc_codes[ndc_idx]
                        matched_ndc_name_list = ndc_names
                        found = True
                        break
                if found:
                    break
            if found:
                break
        drugbank_matched_ndc_codes.append(matched_ndc_code)
        drugbank_matched_ndc_name_lists.append(matched_ndc_name_list)

    # -------------------------------------------------------------------------
    # build drugbank-to-ndc map dataframe and save as csv
    # -------------------------------------------------------------------------
    print("[INFO] Constructing map dataframe and saving as .csv file")
    drugbank_ndc_map_df = pd.DataFrame({
        "drugbank_id": drugbank_ids,
        "drugbank_names": drugbank_name_lists,
        "ndc_code": drugbank_matched_ndc_codes,
        "ndc_names": drugbank_matched_ndc_name_lists})
    drugbank_ndc_map_df.to_csv(os.path.join(out_directory, "drugbank_ndc_map_df.csv"))

    # -------------------------------------------------------------------------
    # translate drugbank interactions list to NDC codes
    # -------------------------------------------------------------------------
    print("[INFO] Translating DrugBank interactions to NDC codes and saving as .csv file")
    drugbank_ndc_map = dict(zip(drugbank_ids, drugbank_matched_ndc_codes))
    interactions_db = pd.read_csv(drugbank_interactions_filepath, sep="\t", names=["drugbank_id_a", "drugbank_id_b"])
    n_interactions = len(interactions_db)
    interactions_db["ndc_a"] = interactions_db["drugbank_id_a"].map(drugbank_ndc_map)
    interactions_db["ndc_b"] = interactions_db["drugbank_id_b"].map(drugbank_ndc_map)
    interactions_db["mapping_successful"] = ~interactions_db[["ndc_a", "ndc_b"]].isnull().any(axis=1)
    interactions_db.to_csv(os.path.join(out_directory, "ndc_interactions.csv"))
    n_mapped_interactions = np.sum(interactions_db["mapping_successful"])
    print(f"[INFO] {n_mapped_interactions} out of {n_interactions} "
          f"({(100 * n_mapped_interactions / n_interactions):.2f}%) "
          f"interactions successfully mapped")





def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--prescriptions_filepath', type=str, required=True)
    parser.add_argument('--drugbank_vocabulary_filepath', type=str, required=True)
    parser.add_argument('--drugbank_interactions_filepath', type=str, required=True)
    parser.add_argument('--out_directory', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(
        args.prescriptions_filepath,
        args.drugbank_vocabulary_filepath,
        args.drugbank_interactions_filepath,
        args.out_directory)
