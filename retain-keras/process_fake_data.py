import pickle
import numpy as np
import pandas as pd
import math
import sys
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    out_directory = sys.argv[1]
    N_str = sys.argv[2]
    train_proportion = float(sys.argv[3])


    # data point t
    TIME = 100
    OFFSET = 4
    DOSAGE = 5

    N = int(N_str)

    mort = np.zeros(N)
    data=[]
    code_placeholder=np.zeros((N, TIME, 4))
    for n in range(N):
        health = np.zeros(TIME)
        med1 = np.zeros(TIME)
        med2 = np.zeros(TIME)

        health[0] = np.random.normal() - OFFSET # come in with lower health status


        med1[0] = np.random.normal(DOSAGE) # correct meds
        med2[0] = np.random.normal(DOSAGE) # wrong meds

        # generate the latent stats from medication
        for i in range(1, TIME):
            med1[i] = max(np.random.normal(DOSAGE, 1), 0)
            med2[i] = max(np.random.normal(DOSAGE, 1), 0)
            health[i] = np.random.normal(health[i-1] + 0.11 * med1[i] - 0.1 * med2[i], 0.1)
        # sigmoid on the latent health status
        mort[n] = np.random.binomial(1, 1/(1+math.exp(-1*(health[TIME-1]))))

        data.append(np.transpose(np.array([med1, med2, health])).tolist())
        print('%d: %f %f %d' % (n, health[1], health[TIME-1], mort[n]))

    print(np.sum(mort))
    #all_data = pd.DataFrame(data={'codes': code_placeholder.tolist(), 'numerics':data}, columns=['codes', 'numerics']).reset_index()
    #all_targets = pd.DataFrame(data={'target': mort.tolist()},columns=['target']).reset_index()

    #data_train,data_test = train_test_split(all_data, train_size=train_proportion, random_state=12345)
    #target_train,target_test = train_test_split(all_targets, train_size=train_proportion, random_state=12345)

    #data_train.sort_index().to_pickle(out_directory+'/data_train.pkl')
    #data_test.sort_index().to_pickle(out_directory+'/data_test.pkl')
    #target_train.sort_index().to_pickle(out_directory+'/target_train.pkl')
    #target_test.sort_index().to_pickle(out_directory+'/target_test.pkl')