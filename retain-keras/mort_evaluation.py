import pandas as pd
import numpy as np

y_train = pd.read_pickle(ARGS.path_target_train)['target'].values
print('%d expired out of %d' % (np.sum(y_train), len(y_train)))