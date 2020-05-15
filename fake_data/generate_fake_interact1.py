import pickle
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    out_directory = sys.argv[1]
    N = int(sys.argv[2])
    TIME = int(sys.argv[3])
    train_proportion = float(sys.argv[4])

    OFFSET = 4
    DOSAGE = 5

    patients = []
    morts = []
    logreg_patients = np.zeros((N, TIME))
    types = {}
    # for every patient
    for n in range(N):
        health = []

        health.append(np.random.normal() - OFFSET) # come in with lower health status

        patient = []
        logreg_patient = np.zeros(TIME*2)
        # generate the latent stats from medication
        # one visit
        for i in range(1, TIME):
            visit = []
            admin = np.random.binomial(1, 0.5, (2,1))[:,0]
            med1 = max(np.random.normal(DOSAGE, 1), 0)
            med2 = max(np.random.normal(DOSAGE, 1), 0)
            avgmed = (med1 + med2) / 2
            code1 = min(max(0, int(med1*4)-11), 17)
            code2 = min(max(18, int(med2*4)+7), 35)

            if admin[0] and admin[1]:
                health.append(np.random.normal(health[i-1] + 0.3 * avgmed, 0.1))
                visit.append(code1)
                visit.append(code2)
                logreg_patient[2*i] = med1
                logreg_patient[2*i+1] = med2
            elif admin[0]:
                health.append(np.random.normal(health[i-1] + 0.1 * med1, 0.1))
                visit.append(code1)
                logreg_patient[2*i] = med1
            elif admin[1]:
                health.append(np.random.normal(health[i-1] - 0.1 * med2, 0.1))
                visit.append(code2)
                logreg_patient[2*i] = 0.
            else:
                health.append(health[-1])
            
            if code1 not in types:
                types[code1] = code1
            if code2 not in types:
                types[code2] = code2
            
            patient.append(visit)

            # sigmoid on the latent health status
            mort = np.random.binomial(1, 1/(1+math.exp(health[i]/2+OFFSET+1)))
            if mort:
                morts.append(1)
                break
        
        patients.append(patient)
        logreg_patients[n, :] = logreg_patient[:TIME]
        if not mort:
            morts.append(0)

    
    # for logistic regression
    logreg_train, logreg_test = train_test_split(logreg_patients, train_size=train_proportion, random_state=12345)
    logreg_train_label, logreg_test_label = train_test_split(morts, train_size=train_proportion, random_state=12345)
    print('Record the following information:')
    print('%d expired out of %d'%(np.sum(morts), N))
    print('%d expired out of %d (test set)'%(np.sum(logreg_test_label), np.shape(logreg_test_label)[0]))
    logreg_model = LogisticRegression().fit(logreg_train, logreg_train_label)
    mean_accuracy = logreg_model.score(logreg_test, logreg_test_label)
    print('Mean accuracy: %f' % mean_accuracy)

    # for RNN
    all_data = pd.DataFrame(data={'codes': patients}, columns=['codes']).reset_index()
    all_targets = pd.DataFrame(data={'target': morts},columns=['target']).reset_index()

    data_train,data_test = train_test_split(all_data, train_size=train_proportion, random_state=12345)
    target_train,target_test = train_test_split(all_targets, train_size=train_proportion, random_state=12345)

    data_train.sort_index().to_pickle(out_directory+'/data_train.pkl')
    data_test.sort_index().to_pickle(out_directory+'/data_test.pkl')
    target_train.sort_index().to_pickle(out_directory+'/target_train.pkl')
    target_test.sort_index().to_pickle(out_directory+'/target_test.pkl')

    pickle.dump(types, open(out_directory+'/dictionary.pkl', 'wb'), -1)