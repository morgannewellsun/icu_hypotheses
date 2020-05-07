import pandas as pd
import numpy as np
import argparse

def main(ARGS):

	y_train = pd.read_pickle(ARGS.path_target)['target'].values
	total_mort = np.sum(y_train)
	if ARGS.doubled:
		y_train = y_train[::2]
		print('Case 1: %d expired out of %d' % (np.sum(y_train), int(len(y_train)/2)))
		print('Case 2: %d expired out of %d' % (total_mort - np.sum(y_train), int(len(y_train)/2)))
	else:
		print('%d expired out of %d' % (total_mort, len(y_train)))

def parse_arguments(parser):
    """Read user arguments"""
    parser.add_argument('--path_target', type=str, default='data/target_train.pkl',
                        help='Path to target')
    parser.add_argument('--doubled', type=bool, default=False, help='Split double-ups (med2 and both case)')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = parse_arguments(PARSER)
    main(ARGS)