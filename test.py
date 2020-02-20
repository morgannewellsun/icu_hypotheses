import pandas as pd
import numpy as np

data_path = './mimic_data/'

d_items = pd.read_csv(data_path + 'DIAGNOSES_ICD.csv.gz', compression = 'gzip', index_col = False)

print(d_items.shape)
# above line tells you that ITEMID 211 is carevue HR and 220045 is the metavision HR
# braden scale for carevue is ITEM_ID 87
