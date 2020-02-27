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

    data=[]

    health_track = []
    track_death_health = []
    track_alive_health = []
    track_death_timestamp = []

    code_placeholder=np.zeros((N, TIME, 4))

    all_seq = []
    for n in range(N):
        health_hold = 0.
        health = []
        med1 = []
        med2 = []

        health.append(np.random.normal() - OFFSET) # come in with lower health status
        med1.append(np.random.normal(DOSAGE)) # correct meds
        med2.append(np.random.normal(DOSAGE)) # wrong meds

        # generate the latent stats from medication
        for i in range(1, TIME):
            # randomly pick which medicine to give (binary equal chance)
            if np.random.binomial(1, 0.5):
                med1.append(max(np.random.normal(DOSAGE, 1), 0))
                health.append(np.random.normal(health[i-1] + 0.12 * med1[-1], 0.1))
            else:
                med2.append(max(np.random.normal(DOSAGE, 1), 0))
                health.append(np.random.normal(health[i-1] - 0.1 * med2[-1], 0.1))

            #health.append(np.random.normal(health[i-1] + 0.12 * med1[i] - 0.1 * med2[i], 0.1))

            # sigmoid on the latent health status
            mort = np.random.binomial(1, 1/(1+math.exp(health[i]/2+OFFSET+1)))
            if mort:
                track_death_health.append(health[i])
                track_death_timestamp.append(i)
                health_track.append(health[i])
                break
            
        if not mort:
            track_alive_health.append(health[TIME-1])
            health_track.append(health[TIME-1])

    #all_data = pd.DataFrame(data={'codes': code_placeholder.tolist(), 'numerics':data}, columns=['codes', 'numerics']).reset_index()
    #all_targets = pd.DataFrame(data={'target': mort.tolist()},columns=['target']).reset_index()

    #data_train,data_test = train_test_split(all_data, train_size=train_proportion, random_state=12345)
    #target_train,target_test = train_test_split(all_targets, train_size=train_proportion, random_state=12345)

    #data_train.sort_index().to_pickle(out_directory+'/data_train.pkl')
    #data_test.sort_index().to_pickle(out_directory+'/data_test.pkl')
    #target_train.sort_index().to_pickle(out_directory+'/target_train.pkl')
    #target_test.sort_index().to_pickle(out_directory+'/target_test.pkl')