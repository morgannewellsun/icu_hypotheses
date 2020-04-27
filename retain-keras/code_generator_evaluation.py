import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from keras.models import load_model

def import_model(path):
    """Import model from given path and assign it to appropriate devices"""
    K.clear_session()
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    tfsess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(tfsess)
    model = load_model(path)
    return model

def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

def main(ARGS):
    model = import_model(ARGS.path_model)
    patients = []


    # experiment 1: Give med2 only then give 1 med1.
    # Start off with either 3 med2's, or 2 med2's and both.

    temperature = 1.0

    if ARGS.decodify:
        med2_codes = [16, 32, 48]
        
        for i in range(ARGS.num_generate):
            med2_patient = np.random.choice(med2_codes, ARGS.maxlen).reshape((1, ARGS.maxlen))
            both_patient = np.copy(med2_patient)
            both_patient[0,-1] += np.random.randint(1,4)

            med2_list = med2_patient.copy().tolist()[0]
            both_list = both_patient.copy().tolist()[0]

            for n in range(ARGS.max_visits):
                preds = model.predict(med2_patient, verbose = 0)[0]
                next_code = sample(preds, temperature)
                med2_list.append(next_code)
                if next_code == 193 or next_code == 194:
                    break
                med2_patient[0, :ARGS.maxlen-1] = med2_patient[0, 1:]
                med2_patient[0, ARGS.maxlen-1] = next_code

            for n in range(ARGS.max_visits):
                preds = model.predict(both_patient, verbose = 0)[0]
                next_code = sample(preds, temperature)
                both_list.append(next_code)
                if next_code == 193 or next_code == 194:
                    break
                both_patient[0, :ARGS.maxlen-1] = both_patient[0, 1:]
                both_patient[0, ARGS.maxlen-1] = next_code

            if med2_list[-1] != 193:
                med2_list.append(194)
            if both_list[-1] != 193:
                both_list.append(194)


            patients.append(med2_list)
            patients.append(both_list)
        
    else:
        if ARGS.simple:
            med2_codes = np.arange(3, 6)
            med1_codes = np.arange(0, 3)
            termination = [12, 13]
        else:
            med2_codes = np.arange(18, 36)
            med1_codes = np.aranage(0, 18)
            termination = [63, 64]

        for i in range(ARGS.num_generate):
            exp_num = int(ARGS.maxlen * 3 / 5) # 30
            live_num = int((ARGS.maxlen - exp_num)/2) # 10x2 for total 50


            med2_sequence = np.random.choice(med2_codes, ARGS.maxlen)
            both_sequence = np.zeros(ARGS.maxlen)
            both_sequence[:exp_num] = med2_sequence[:exp_num]

            med1_both = np.random.choice(med1_codes, live_num)
            med2_both = np.random.choice(med2_codes, live_num)
            # place lattice
            for n in range(live_num):
                both_sequence[2*n+exp_num] = med1_both[n]
                both_sequence[2*n+1+exp_num] = med2_both[n]

            med2_list = med2_sequence.copy().tolist()[0]
            both_list = both_sequence.copy().tolist()[0]

            for n in range(ARGS.max_visits):
                preds = model.predict(med2_patient, verbose = 0)[0]
                next_code = sample(preds, temperature)
                med2_list.append(next_code)
                if next_code in termination:
                    break
                med2_patient[0, :ARGS.maxlen-1] = med2_patient[0, 1:]
                med2_patient[0, ARGS.maxlen-1] = next_code

            for n in range(ARGS.max_visits):
                preds = model.predict(both_patient, verbose = 0)[0]
                next_code = sample(preds, temperature)
                both_list.append(next_code)
                if next_code in termination:
                    break
                both_patient[0, :ARGS.maxlen-1] = both_patient[0, 1:]
                both_patient[0, ARGS.maxlen-1] = next_code

            if med2_list[-1] != termination[0]:
                med2_list.append(termination[1])
            if both_list[-1] != termination[0]:
                both_list.append(termination[1])


            patients.append(med2_list)
            patients.append(both_list)
        
    for p in patients:
        print(p)


def parse_arguments(parser):
    """Read user arguments"""
    parser.add_argument('--path_model', type=str, default='data/data_train.pkl',
                        help='Path to train data')
    parser.add_argument('--directory', type=str, default='./',
                        help='Path to output models')
    parser.add_argument('--maxlen', type=int, default=3,
                        help='Maximum length of LSTM')
    parser.add_argument('--num_generate', type=int, default=100,
                        help='Number of samples to generate * 2')
    parser.add_argument('--max_visits', type=int, default=30,
                        help='Number of visits to generate')
    parser.add_argument('--decodify', type=bool, default=False,
                        help='Decode if codified already')
    parser.add_argument('--simple', type=bool, default=False,
                        help='If simple, then process differently')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = parse_arguments(PARSER)
    main(ARGS)