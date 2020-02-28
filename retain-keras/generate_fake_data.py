import pickle
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    out_directory = sys.argv[1]
    N_str = sys.argv[2]
    train_proportion = float(sys.argv[3])

    # data point t
    TIME = 30
    OFFSET = 4
    DOSAGE = 5

    N = int(N_str)

    code_placeholder=np.zeros((N, TIME, 4))

    patients = []
    morts = []
    # for every patient
    for n in range(N):
        health = []

        health.append(np.random.normal() - OFFSET) # come in with lower health status

        patient = []
        # generate the latent stats from medication
        # one visit
        for i in range(1, TIME):
            visit = []
            med1 = max(np.random.normal(DOSAGE, 1), 0)
            med2 = max(np.random.normal(DOSAGE, 1), 0)
            visit.append(min(max(0, int(med1*2)-5), 9))
            visit.append(min(max(10, int(med2*2)+5), 19))
            
            health.append(np.random.normal(health[i-1] + 0.12 * med1 - 0.1 * med2, 0.1))

            patient.append(visit)

            # sigmoid on the latent health status
            mort = np.random.binomial(1, 1/(1+math.exp(health[i]/2+OFFSET+1)))
            if mort:
                morts.append[1]
                break
        
        patients.append(patient)
        if not mort:
            morts.append[0]

    all_data = pd.DataFrame(data={'codes': patients}, columns=['codes']).reset_index()
    all_targets = pd.DataFrame(data={'target': morts},columns=['target']).reset_index()

    data_train,data_test = train_test_split(all_data, train_size=train_proportion, random_state=12345)
    target_train,target_test = train_test_split(all_targets, train_size=train_proportion, random_state=12345)

    data_train.sort_index().to_pickle(out_directory+'/data_train.pkl')
    data_test.sort_index().to_pickle(out_directory+'/data_test.pkl')
    target_train.sort_index().to_pickle(out_directory+'/target_train.pkl')
    target_test.sort_index().to_pickle(out_directory+'/target_test.pkl')