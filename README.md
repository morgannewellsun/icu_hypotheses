This repository contains code for an undergraduate thesis research project. The aim of the project is to apply machine learning to emergency health record data in order to identify adverse drug-drug interactions. More information about the project can be found in the thesis PDF [here](https://github.com/morgannewellsun/icu_hypotheses/blob/master/Sun%2C%20Morgan%20-%20Thesis%20Final%20Report%20(Virtual%20Experiments).pdf). The following setup information assumes knowledge about the project.

### Setup

To setup the prerequisites for the code, create an Anaconda environment using the icu_hypotheses.yaml file provided in this repo.

The data required to run the code consists of the following:

- MIMIC-III dataset (specifically, the files PATIENTS.csv, ADMISSIONS.csv, PRESCRIPTIONS.csv, PROCEDURES_ICD.csv, and DIAGNOSES_ICD.csv)
- BIOSNAP Chemical-Chemical dataset (a TSV file)
- Drugbank vocabulary (a CSV file)

### Code overview

All of the scripts used for the thesis can be found in the `refactored` directory. Due to the large number of hyperparameters for these scripts, hyperparameters are hard-coded inside the script at the start of each main() function; the function of these hyperparameters are commented there as well. Command-line arguments are, for the most part, only used to specify input and output filepaths and directories (refer to the --help for information on these arguments).

### Preparing the data

`preprocess_and_profile_mimic.py` will load the separate CSV files from the MIMIC-III data and join them together. The output is a file with one medical event per row (a medical event is either a prescription, procedure, or diagnosis). Indices are provided to identify distinct patients, patient-admissions, and patient-admission-events. The output file is used as input for all downstream scripts which use MIMIC-III data. This script additionally generates some plots which help understand various distributions in the MIMIC-III data.

`map_drugbank_to_ndc.py` will map the BIOSNAP drug-drug interaction pairs, which are specified using Drugbank identifier codes, to National Drug Codes. This is necessary to allow these pairs to be corresponded with the prescription events in MIMIC-III, which are specified as NDC codes (in addition to other coding schemes which are less relevant to this project). The output file is used by any downstream scripts which use known interaction data.

### Training the LSTMs

`train_lstm_on_mimic.py` will construct the forecaster LSTM, further preprocess the MIMIC-III data, and train the forecaster LSTM on MIMIC-III. The forecaster's predictive accuracy can be then evaluated using `evaluate_lstm_topk_accuracy.py`.

`train_retain_on_mimic.py` will similarly construct the RETAIN LSTM architecture and train it on MIMIC-III.

### Generating synthetic patients

`run_virtual_experiments.py` will use the trained forecaster LSTM to generate synthetic patients. `sample_and_swap.py` generates synthetic patients using the sample-and-swap methodology.

### Identifying DDIs

`generate_retain_interpretations.py` runs RETAIN inference on the synthetic patient data. This should be run once for the virtual-experiments synthetic patients, and once for the sample-and-swap synthetic patients. The output of this file is given to the various `find_interpretations_*.py` scripts, each of which will first use some methodology to attempt to identify adverse DDIs, then evaluate this against the list of known DDIs to calculate a precision-recall curve and AP score.
