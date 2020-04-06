import numpy as np
import math
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    N_str = sys.argv[1]
    train_proportion = float(sys.argv[2])

    # data point t
    TIME = 30
    OFFSET = 4
    DOSAGE = 5

    N = int(N_str)

    code_placeholder=np.zeros((N, TIME, 4))

    patients = np.zeros((N, TIME*2))

    # see if med3 will accumulate
    genetics = np.random.binomial(1, 0.5, (N,1))[:,0]

    morts = np.zeros(N)
    # for every patient
    for n in range(N):
        health = [np.random.normal() - OFFSET]
        heartrate = 0.

        patient = np.zeros(TIME*4)
        for i in range(1, TIME):
            admin = np.random.binomial(1, 0.5, (3,1))[:,0]
            med1 = max(np.random.normal(DOSAGE, 1), 0)
            med2 = max(np.random.normal(DOSAGE, 1), 0)
            med3 = max(np.random.normal(DOSAGE, 1), 0)
            avgmed = (med1 + med2) / 2

            health_delta = 0.

            # interactive of both medications
            if admin[0] and admin[1]:
                health_delta += np.random.normal(0.3 * avgmed, 0.1)
                patient[4*i] = med1
                patient[4*i+1] = med2
            elif admin[0]:
                health_delta += np.random.normal(0.1 * med1, 0.1)
                patient[4*i] = med1
            elif admin[1]:
                health_delta += np.random.normal(-0.1 * med2, 0.1)
                patient[4*i+1] = med2

            # heart rate of medication 2 assigned genetically
            if admin[2]:
                patient[4*i+2] = med3
                if genetics[n]:
                    delta = -med3/50.
                    heartrate += delta
                    patient[4*i+3] = delta
                    health_delta += np.random.normal(heartrate, 0.05)
                else:
                    health_delta += np.random.normal(0.1 * med3, 0.1)
                
            
            health.append(health[-1] + health_delta)
            # sigmoid on the latent health status
            mort = np.random.binomial(1, 1/(1+math.exp(health[i]/2+OFFSET+1)))
            if mort:
                morts[n] = 1
                break
        
        patients[n, :] = patient[:TIME*2]
        if not mort:
            morts[n] = 0

    print(np.sum(morts))

    data_train,data_test = train_test_split(patients, train_size=train_proportion, random_state=12345)
    target_train,target_test = train_test_split(morts, train_size=train_proportion, random_state=12345)

    logreg_model = LogisticRegression().fit(data_train, target_train)
    mean_accuracy = logreg_model.score(data_test, target_test)
    print(mean_accuracy)