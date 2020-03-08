import pickle
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    N_str = sys.argv[1]

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
    for n in range(N):
        health = []
        med1 = []
        med2 = []

        health.append(np.random.normal() - OFFSET) # come in with lower health status
        med1.append(np.random.normal(DOSAGE)) # correct meds
        med2.append(np.random.normal(DOSAGE)) # wrong meds

        # generate the latent stats from medication
        for i in range(1, TIME):

            med1.append(max(np.random.normal(DOSAGE, 1), 0))
            med2.append(max(np.random.normal(DOSAGE, 1), 0))

            admin = np.random.binomial(1, 0.5, (2,1))[:,0]
            avgmed = (med1[-1] + med2[-1]) / 2

            if admin[0] and admin[1]:
                health.append(np.random.normal(health[i-1] + 0.15 * avgmed, 0.1))
            elif admin[0]:
                health.append(np.random.normal(health[i-1] + 0.1 * med1[-1], 0.1))
            elif admin[1]:
                health.append(np.random.normal(health[i-1] - 0.1 * med2[-1], 0.1))
            else:
                health.append(health[-1])
            
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

    print(med1)
    print(med2)

    print(len(track_death_health))
    plt.figure()
    colors = ['orange', 'black']
    plt.hist([track_death_health, track_alive_health], bins='auto', label=colors)
    plt.title("Health of patients expired vs not expired")
    plt.show()

    plt.figure()
    plt.hist(health_track, bins='auto')
    plt.title("Health of all patients")
    plt.show()

    plt.figure()
    plt.hist(track_death_timestamp, bins='auto')
    plt.title("Time of death of expired patients")
    plt.show()
