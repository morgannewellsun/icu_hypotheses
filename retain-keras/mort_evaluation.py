import pandas as pd
import numpy as np

def main(ARGS):

	y_train = pd.read_pickle(ARGS.path_target)['target'].values
	print('%d expired out of %d' % (np.sum(y_train), len(y_train)))

def parse_arguments(parser):
    """Read user arguments"""
    parser.add_argument('--path_target', type=str, default='data/target_train.pkl',
                        help='Path to target')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = parse_arguments(PARSER)
    main(ARGS)