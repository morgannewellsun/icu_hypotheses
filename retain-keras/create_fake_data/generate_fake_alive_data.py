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
    train_proportion = 0

    # data point t
    TIME = 30
    OFFSET = 4
    DOSAGE = 5

    N = int(N_str)

    code_placeholder=np.zeros((N, TIME, 4))

    patients = []
    morts = []
    types = {}
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
            #med2 = max(np.random.normal(DOSAGE, 1), 0)
            med2 = 0
            code1 = min(max(0, int(med1*4)-11), 17)
            code2 = min(max(18, int(med2*4)+7), 36)
            visit.append(code1)
            visit.append(code2)
            if code1 not in types:
                types[code1] = code1
            if code2 not in types:
                types[code2] = code2
            
            health.append(np.random.normal(health[i-1] + 0.12 * med1 - 0.1 * med2, 0.1))

            patient.append(visit)

            # sigmoid on the latent health status
            mort = np.random.binomial(1, 1/(1+math.exp(health[i]/2+OFFSET+1)))
            if mort:
                morts.append(1)
                break
        
        patients.append(patient)
        if not mort:
            morts.append(0)
    print(np.sum(np.array(morts)))

    all_data = pd.DataFrame(data={'codes': patients}, columns=['codes']).reset_index()
    all_targets = pd.DataFrame(data={'target': morts},columns=['target']).reset_index()

    data_train,data_test = train_test_split(all_data, train_size=train_proportion, random_state=12345)
    target_train,target_test = train_test_split(all_targets, train_size=train_proportion, random_state=12345)

    data_test.sort_index().to_pickle(out_directory+'/data_test_death.pkl')
    target_test.sort_index().to_pickle(out_directory+'/target_test_death.pkl')

    pickle.dump(types, open(out_directory+'/dictionary.pkl', 'wb'), -1)