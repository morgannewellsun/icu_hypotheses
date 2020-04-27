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

    # data point t
    OFFSET = 4
    DOSAGE = 5

    patients = np.zeros((N, TIME*2))

    # see if med3 will accumulate
    genetics = np.random.binomial(1, 0.5, (N,1))[:,0]

    logreg_patients = np.zeros((N, TIME*2))
    patients = []
    morts = []
    types = {}

    time_death = []

    # for every patient
    for n in range(N):
        health = [np.random.normal() - OFFSET]
        heartrate = 0.

        logreg_patient = np.zeros(TIME*4)
        patient = []
        for i in range(1, TIME):
            visit = []
            admin = np.random.binomial(1, 0.5, (3,1))[:,0]
            med1 = max(np.random.normal(DOSAGE, 1), 0)
            med2 = max(np.random.normal(DOSAGE, 1), 0)
            med3 = max(np.random.normal(DOSAGE, 1), 0)
            avgmed = (med1 + med2) / 2

            '''
            The code system works as follows: discretize the same way as the method in 1 and 2
            But this time, each 6 bins previously make up 1 bin
            Meaning, -INF to 4.25, 4.25-5.75, 5.75-INF
            '''
            code1 = min(max(0, int(((med1*4)-11)/6)), 2) # 0-2
            code2 = min(max(3, int(((med2*4)+7)/6)), 5) # 3-5
            code3 = min(max(6, int(((med3*4)+25)/6)), 8) # 6-8


            health_delta = 0.

            # interactive of both medications
            if admin[0] and admin[1]:
                health_delta += np.random.normal(0.3 * avgmed, 0.1)
                visit.append(code1)
                visit.append(code2)
                logreg_patient[4*i] = med1
                logreg_patient[4*i+1] = med2
            elif admin[0]:
                health_delta += np.random.normal(0.1 * med1, 0.1)
                visit.append(code1)
                logreg_patient[4*i] = med1
            elif admin[1]:
                health_delta += np.random.normal(-0.1 * med2, 0.1)
                visit.append(code2)
                logreg_patient[4*i+1] = med2

            # heart rate of medication 2 assigned genetically
            if admin[2]:
                visit.append(code3)
                logreg_patient[4*i+2] = med3
                if genetics[n]:
                    delta = -med3/50. # change is med3/50
                    heartrate += delta
                    logreg_patient[4*i+3] = delta
                    health_delta += np.random.normal(heartrate, 0.025)
                    visit.append(min(max(10, 10+int(-delta * 10)), 11))
                else:
                    health_delta += np.random.normal(0.1 * med3, 0.1)
                    visit.append(9) # no heart change
                
            patient.append(visit)
            health.append(health[-1] + health_delta)
            # sigmoid on the latent health status
            mort = np.random.binomial(1, 1/(1+math.exp(health[i]/2+OFFSET+1)))
            if mort:
                morts.append(1)
                time_death.append(i)
                break
        
        patients.append(patient)
        logreg_patients[n, :] = logreg_patient[:TIME*2]
        if not mort:
            morts.append(0)

    plt.hist(time_death)
    plt.show()
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