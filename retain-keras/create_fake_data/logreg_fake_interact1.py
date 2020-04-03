import numpy as np
import math
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

    patients = np.zeros((N, TIME*2))
    morts = np.zeros(N)
    # for every patient
    for n in range(N):
        health = []

        health.append(np.random.normal() - OFFSET) # come in with lower health status

        patient = np.zeros(TIME*2)
        for i in range(1, TIME):
            admin = np.random.binomial(1, 0.5, (2,1))[:,0]
            med1 = max(np.random.normal(DOSAGE, 1), 0)
            med2 = max(np.random.normal(DOSAGE, 1), 0)
            avgmed = (med1 + med2) / 2

            if admin[0] and admin[1]:
                health.append(np.random.normal(health[i-1] + 0.3 * avgmed, 0.1))
                patient[2*i] = med1
                patient[2*i+1] = med2
            elif admin[0]:
                health.append(np.random.normal(health[i-1] + 0.1 * med1, 0.1))
                patient[2*i] = med1
                patient[2*i+1] = 0.
            elif admin[1]:
                health.append(np.random.normal(health[i-1] - 0.1 * med2, 0.1))
                patient[2*i] = 0.
                patient[2*i+1] = med2
            else:
                health.append(health[-1])
                patient[2*i] = 0.
                patient[2*i+1] = 0.
            
            # sigmoid on the latent health status
            mort = np.random.binomial(1, 1/(1+math.exp(health[i]/2+OFFSET+1)))
            if mort:
                morts[i] = 1
                break
        
        patients[n, :] = patient
        if not mort:
            morts[i] = 0

    data_train,data_test = train_test_split(patients, train_size=train_proportion, random_state=12345)
    target_train,target_test = train_test_split(morts, train_size=train_proportion, random_state=12345)

    logreg_model = LogisticRegression().fit(data_train, target_train)
    mean_accuracy = logreg_model.score(data_test, target_test)
    print(mean_accuracy)