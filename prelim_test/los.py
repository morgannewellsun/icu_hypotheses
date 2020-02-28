import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt

data_path = './mimic_data/'

with open('deadaticu_subj_stay.json') as jf:
	dead_patients = json.load(jf)

patients = pd.read_csv(data_path + 'PATIENTS.csv.gz', compression = 'gzip', index_col = False)
icustays = pd.read_csv(data_path + 'ICUSTAYS.csv.gz', compression = 'gzip', index_col = False)

gender_flag = 0

if not gender_flag:
	ind = dead_patients.values()
	'''
	DEAD PATIENTS
	'''
	icustays_dead_ind = [i for i, x in enumerate(icustays['ICUSTAY_ID']) if x in ind]

	icustays_dead = icustays.loc[icustays_dead_ind]

	icu_types = icustays_dead.LAST_CAREUNIT.unique()
	icu_types = np.delete(icu_types, np.where(icu_types == 'NICU')) # remove NICU
	counts_dead = np.zeros(len(icu_types))
	counts_total = np.zeros(len(icu_types))

	plt.figure()
	hist_list = []
	for i, icu_type in enumerate(icu_types):
		icu_ind = [i for i, x in enumerate(icustays_dead.LAST_CAREUNIT) if str(x) == str(icu_type)]
		count = len(icu_ind)
		counts_dead[i] = count

		icutype_dead = icustays_dead.iloc[icu_ind]
		hist_list.append(icutype_dead.LOS[icutype_dead.LOS < 30])
		print('%s has %d, %f' % (icu_type, count, count/icustays_dead.shape[0]))
	plt.hist(hist_list, bins = 30, stacked = True)
	plt.legend(icu_types)
	plt.xlabel('Days at unit')
	plt.ylabel('Frequency')
	plt.show()

	'''
	LIVE PATIENTS
	'''

	icustays_live_ind = [i for i, x in enumerate(icustays['ICUSTAY_ID']) if x not in ind]

	icustays_live = icustays.loc[icustays_live_ind]

	counts_live = np.zeros(len(icu_types))
	counts_total = np.zeros(len(icu_types))

	plt.figure()
	hist_list = []
	for i, icu_type in enumerate(icu_types):
		icu_ind = [i for i, x in enumerate(icustays_live.LAST_CAREUNIT) if str(x) == str(icu_type)]
		count = len(icu_ind)
		counts_live[i] = count

		icutype_live = icustays_live.iloc[icu_ind]
		hist_list.append(icutype_live.LOS[icutype_live.LOS < 30])

		print('%s has %d, %f' % (icu_type, count, count/icustays_live.shape[0]))
	plt.hist(hist_list, bins = 30, stacked = True)
	plt.legend(icu_types)
	plt.xlabel('Days at unit')
	plt.ylabel('Frequency')
	plt.show()

else:
	'''
	GENDER DEAD PATIENTS
	'''
	dead_patients_int = [int(x) for x in dead_patients.keys()]
	filtered_patients = patients.iloc[[i for i, x in enumerate(patients.SUBJECT_ID) if x in dead_patients_int]]

	dead_females = filtered_patients.iloc[[i for i, x in enumerate(filtered_patients.GENDER) if x == 'F']]
	dead_males = filtered_patients.iloc[[i for i, x in enumerate(filtered_patients.GENDER) if x == 'M']]

	dead_female_icu = [dead_patients[str(k)] for k in dead_females.SUBJECT_ID]
	dead_male_icu = [dead_patients[str(k)] for k in dead_males.SUBJECT_ID]



	icustays_dead_ind = [i for i, x in enumerate(icustays['ICUSTAY_ID']) if x in dead_female_icu]

	icustays_dead = icustays.loc[icustays_dead_ind]

	icu_types = icustays_dead.LAST_CAREUNIT.unique()
	icu_types = np.delete(icu_types, np.where(icu_types == 'NICU')) # remove NICU
	counts_dead = np.zeros(len(icu_types))
	counts_total = np.zeros(len(icu_types))

	plt.figure()
	hist_list = []
	for i, icu_type in enumerate(icu_types):
		icu_ind = [i for i, x in enumerate(icustays_dead.LAST_CAREUNIT) if str(x) == str(icu_type)]
		count = len(icu_ind)
		counts_dead[i] = count

		icutype_dead = icustays_dead.iloc[icu_ind]
		hist_list.append(icutype_dead.LOS[icutype_dead.LOS < 30])
		print('%s has %d, %f' % (icu_type, count, count/icustays_dead.shape[0]))
	plt.hist(hist_list, bins = 30, stacked = True)
	plt.legend(icu_types)
	plt.xlabel('Days at unit')
	plt.ylabel('Frequency')
	plt.show()

	icustays_dead_ind = [i for i, x in enumerate(icustays['ICUSTAY_ID']) if x in dead_male_icu]

	icustays_dead = icustays.loc[icustays_dead_ind]

	plt.figure()
	hist_list = []
	for i, icu_type in enumerate(icu_types):
		icu_ind = [i for i, x in enumerate(icustays_dead.LAST_CAREUNIT) if str(x) == str(icu_type)]
		count = len(icu_ind)
		counts_dead[i] = count

		icutype_dead = icustays_dead.iloc[icu_ind]
		hist_list.append(icutype_dead.LOS[icutype_dead.LOS < 30])
		print('%s has %d, %f' % (icu_type, count, count/icustays_dead.shape[0]))
	plt.hist(hist_list, bins = 30, stacked = True)
	plt.legend(icu_types)
	plt.xlabel('Days at unit')
	plt.ylabel('Frequency')
	plt.show()
	'''
	GENDER LIVE PATIENTS
	'''