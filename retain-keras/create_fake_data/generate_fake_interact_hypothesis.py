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
        for i in range(1, TIME*4):
            visit = []
            med1 = max(np.random.normal(DOSAGE, 1), 0)
            med2 = max(np.random.normal(DOSAGE, 1), 0)
            avgmed = (med1 + med2) / 2
            code1 = min(max(0, int(med1*4)-11), 17)
            code2 = min(max(18, int(med2*4)+7), 36)

            if i < TIME:
                health.append(np.random.normal(health[i-1] + 0.3 * avgmed, 0.1))
                visit.append(code1)
                visit.append(code2)
            elif i < TIME*2:
                health.append(np.random.normal(health[i-1] + 0.1 * med1, 0.1))
                visit.append(code1)
            elif i < TIME*3:
                health.append(np.random.normal(health[i-1] - 0.1 * med2, 0.1))
                visit.append(code2)
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
        if not mort:
            morts.append(0)

    interact_data = pd.DataFrame(data={'codes': patients[:TIME]}, columns=['codes']).reset_index()
    interact_target = pd.DataFrame(data={'target': morts[:TIME]},columns=['target']).reset_index()
    interact_data.sort_index().to_pickle(out_directory+'/interact_data.pkl')
    interact_target.sort_index().to_pickle(out_directory+'/interact_target.pkl')

    med1_data = pd.DataFrame(data={'codes': patients[TIME:TIME*2]}, columns=['codes']).reset_index()
    med1_target = pd.DataFrame(data={'target': morts[TIME:TIME*2]},columns=['target']).reset_index()
    med1_data.sort_index().to_pickle(out_directory+'/med1_data.pkl')
    med1_target.sort_index().to_pickle(out_directory+'/med1_target.pkl')

    med2_data = pd.DataFrame(data={'codes': patients[TIME*2:TIME*3]}, columns=['codes']).reset_index()
    med2_target = pd.DataFrame(data={'target': morts[TIME*2:TIME*3]},columns=['target']).reset_index()
    med2_data.sort_index().to_pickle(out_directory+'/med2_data.pkl')
    med2_target.sort_index().to_pickle(out_directory+'/med2_target.pkl')

    control_data = pd.DataFrame(data={'codes': patients[TIME*3:]}, columns=['codes']).reset_index()
    control_target = pd.DataFrame(data={'target': morts[TIME*3]},columns=['target']).reset_index()
    control_data.sort_index().to_pickle(out_directory+'/control_data.pkl')
    control_target.sort_index().to_pickle(out_directory+'/control_target.pkl')

    pickle.dump(types, open(out_directory+'/dictionary.pkl', 'wb'), -1)