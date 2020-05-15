# ICU Hypothesis Project

This is the repository for generating and testing complex hypotheses using the RETAIN model ([Choi et al. 2017](https://arxiv.org/abs/1608.05745)) and a custom attention-based LSTM generator.
## Table of Contents
1. [Requirements](requirements)
    1. [Package requirements](#package-requirements)
    2. [Hardware requirement](#hardware-requirement)
2. [Run code](#run-code)
    1. [Data](#data)
    2. [RETAIN Predictor](#retain-predictor)
    3. [LSTM Generator](#lstm-generator)
3. [Completed Runs](#completed-runs)
4. [Contributors](#contributors)

## Requirements
### Package requirements
- Python 3.x
- Tensorflow 2.x
- Keras 2.x
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

### Hardware requirement
The code was run using tigergpu. Training RETAIN locally is not recommended. Training the generator is feasible on the local machine.

## Run code
### Data
You can use the [MIMIC-III](https://mimic.physionet.org/) dataset after getting approval and going through training. This must be parsed by running the ```process_mimic_modified.py``` file in the [retain-keras folder](retain-keras). Running this file is explained in the commented header of the file itself.

In the mean time, you could use the fake dataset. There are ```*interact*.py``` files in the [fake data folder](fake_data).

```python generate_fake_interact1.py /output/directory/foldername NUM TIME PROP```

where ```NUM``` is the number of samples to make with maximum number of visits being ```TIME```. ```PROP``` is the training proportion.

### RETAIN Predictor
The RETAIN model used is an adaptation of the [reimplementation of RETAIN on Keras](https://github.com/Optum/retain-keras). To see more detailed description of each field, you can refer to that repo. Below are the commands used for the Independent Work Project specifically. The relevant files can be found in the [retain-keras folder](retain-keras)

1. **Train**: 

```python retain_train.py --num_codes=N --epochs=M --path_data_train=/path/to/train/data.pkl --path_data_test=/path/to/test/data.pkl --path_target_train=/path/to/train/target.pkl --path_target_test=/path/to/test/target.pkl --directory=/output/directory```

where ```N``` is the number of codes (medical or fake) in the dataset, ```M``` is the number of epochs to train. There are other fields that could be specified. This is

2. **Evaluate**

First, check the directory of the training and look at ```output.txt```, which has all the losses per Epoch. Use the best model, which has weights stored in ```weight-xx.h5```.

```python retain_evaluation.py --path_model=/path/to/model/weight-xx.h5 --path_data=/path/to/test/data.pkl --path_target=/path/to/test/target.pkl```

3. **Interpret**

For now, interpretation will give you mortality probability, and visit and feature importance weights by patient/sample. Make sure you have a ```dictionary.pkl``` that maps each unique code to what it actually is in a string type. For instance, medical code 0 could be 'low medicine 1 dosage'

```python retain_interpretation.py --path_model=/path/to/model/weight-xx.h5 --path_data=/path/to/test/data.pkl --path_dictionary=dictionary.pkl```

### LSTM Generator
These are the commands used to run the generator code. The files can be found in the [generator folder here](generator)

1. **Train**
The code takes data and converts them into sequences of codes. A sentinel code is used for "nothing happening" status. This is padded in the front of the list in the case when there are not enough codes in one patient data (< maxlen).

```python code_generator.py --num_codes=N --epochs=M --emb_size=P --maxlen=Q --path_data_train=/path/to/train/data.pkl --path_target_train=/path/to/train/target.pkl --directory=/output/directory --simple=True/False```

Maxlen ```Q``` is the maximum length of an input sequence. For the study, we used ```Q=15```, embedding size of ```M=10```. Simple should be set to ```--simple=True``` if using ```interact3```. If using ```interact2```, use ```--simple=False``` (default). 

2. **Generate**
Now, we take the best trained model and generate data. The folder has 2 files. ```code_generator_evaluation.py``` and ```code_generator_evaluation_med1.py```. The two have the same file inputs. The latter is data generated when patients receive 2 dosages of med1. The former is receiving 2 dosage of med2, or 1 of med1 and 1 of med2 (mini-experiment).

```python code_generator_evaluation.py --path_model=/path/to/model/weight-xx.h5 --directory=/output/of/data/folder --maxlen=Q --num_generate=N --max_visits=M```

where ```Q``` is the same maxlen as before during training. ```N``` is the number of samples to make with up to ```M``` codes predicted per patient (less if patient expired before ```M``` codes).

## Completed Runs
The RETAIN evaluations of various experiments have been completed and stored in the [images folder](images). [example_attempt2](images/example_attempt2),[example_mod1](images/example_mod1), and [example_modrev](images/example_modrev) are the RETAIN evaluations on MIMIC-III data using Bidirectional, Forward, and Reverse of input. ```fake_interactx``` has the run for RETAIN on ```interactx```. The ```weight``` folders are the runs of RETAIN with specific weighting (you shouldn't need to worry about this). Lastly, the [images/hypothesize_true3 folder](images/hypothesize_true3) has the runs of using generated data (with different number of samples used to train the generator) with ```interact3```. 

## Contributors
- Daniel Chae (Princeton Class of 2020), advised by Michael Guerzhoy
