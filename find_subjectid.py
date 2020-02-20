import pandas as pd
import numpy as np
import datetime
import json

data_path = './mimic_data/'

patients = pd.read_csv(data_path + 'PATIENTS.csv.gz', compression = 'gzip', index_col = False)

dead_patients_ind = [i for i, x in enumerate(patients['DOD_HOSP']) if len(str(x)) > 3]
dead_patients = patients.loc[dead_patients_ind]
# get the datetime of death at a hospital
dead_date = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dead_patients['DOD_HOSP']]
dead_subjectid = dead_patients['SUBJECT_ID']
dead_dict = dict(zip(dead_subjectid, dead_date))

# find the last ICU admission for patients who died at the hospital
subj_icurange = {}
icustays = pd.read_csv(data_path + 'ICUSTAYS.csv.gz', compression = 'gzip', index_col = False)
for i, subj in enumerate(dead_subjectid):
	if i % 1000 == 0:
		print('%d out of %d completed' % (i, dead_subjectid.size))
	subj_admits_ind = [i for i, x in enumerate(icustays['SUBJECT_ID']) if x == subj]
	if len(subj_admits_ind) > 0:
		subj_admit = icustays.loc[subj_admits_ind[-1]]
		if type(subj_admit['INTIME']) == str and type(subj_admit['OUTTIME']) == str:
			time_in = datetime.datetime.strptime(subj_admit['INTIME'], '%Y-%m-%d %H:%M:%S') - datetime.timedelta(hours = 6)
			time_out = datetime.datetime.strptime(subj_admit['OUTTIME'], '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours = 6)
			subj_icurange[subj] = {'in': time_in, 'out': time_out, 'id': subj_admit['ICUSTAY_ID']}

# check the time interval
print('Now finding ICU deaths...')
icu_mortality = {}
for k in subj_icurange.keys():
	time_in = subj_icurange[k]['in']
	time_out = subj_icurange[k]['out']
	if dead_dict[k] >= time_in and dead_dict[k] <= time_out:
		icu_mortality[int(k)] = int(subj_icurange[k]['id'])
print(len(icu_mortality))

print('Writing to icumortality.json...')
with open('deadaticu_subj_stay.json', 'w') as writer:
	json.dump(icu_mortality, writer)

print('DONE!')