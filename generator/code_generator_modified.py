
import argparse

# import keras
from keras.models import Model
from keras import layers
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, CSVLogger
import numpy as np
import pandas as pd


def read_data(path_data, path_target):
    """Read the data from provided paths and assign it into lists"""
    x_train = pd.read_pickle(path_data)['codes'].values
    y_train = pd.read_pickle(path_target)['target'].values
    return x_train, y_train


def process(x_train, y_train, num_codes, maxlen):
    """Generate all truncated subsequences of all patient medical histories"""
    num_sequences = 0
    all_sequences = []
    for i, patient in enumerate(x_train):

        # initialize patient sequence with zeros for padding
        # subsequences of length maxlen will be extracted from the sequence
        # first subsequence will consist of (maxlen-1) zeros and first code
        patient_sequence = [0] * (maxlen - 1)
        for visit in patient:
            for code in visit:
                # increase all codes by 1 to allow zero to represent padding
                patient_sequence.append(code + 1)

        # finish sequences with patient outcome
        # mortality represented by (num_codes+1)
        # release represented by (num_codes+2)
        if y_train[i] == 1:
            patient_sequence.append(num_codes + 1)
        else:
            patient_sequence.append(num_codes + 2)

        num_sequences += len(patient_sequence) - maxlen
        all_sequences.append(patient_sequence)

    # x is an array of subsequences
    # y contains the next code for each subsequence in x
    x = np.zeros((num_sequences, maxlen))
    y = np.zeros((num_sequences, num_codes + 3))
    count = 0
    for patient in all_sequences:
        for i in range(len(patient) - maxlen):
            x[count, :] = patient[i: i+maxlen]
            y[count, int(patient[i+maxlen])] = 1
            count += 1

    print('Found %d sequences of length %d among %d patients' % (num_sequences, maxlen, len(all_sequences)))
    return x, y


def main(args):

    maxlen = args.maxlen
    embed_size = args.emb_size
    num_codes = args.num_codes
    epochs = args.epochs
    batch_size = args.batch_size

    print('Reading Data...')
    x_train, y_train = read_data(args.path_data_train, args.path_target_train)
    x_val, y_val = read_data(args.path_data_val, args.path_target_val)

    print('Processing Data...')
    x_train, y_train = process(x_train, y_train, num_codes, maxlen)
    x_val, y_val = process(x_val, y_val, num_codes, maxlen)

    print('Creating Model...')
    input_layer = layers.Input((maxlen,), name='time_input')
    embedding = layers.Embedding(input_dim=num_codes+3, output_dim=embed_size)(input_layer)
    activations = layers.LSTM(128, input_shape=(maxlen, embed_size), return_sequences=True)(embedding)
    attention = layers.Dense(1, activation='tanh')(activations)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(embed_size)(attention)
    attention = layers.Permute([2, 1])(attention)
    sent_representation = layers.Multiply()([attention, embedding])
    attention_activations = layers.LSTM(128, input_shape=(maxlen, embed_size))(sent_representation)
    predictions = layers.Dense(num_codes+3, activation='softmax')(attention_activations)
    model = Model(input=input_layer, output=predictions)
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    print('Training Model...')
    checkpoint = ModelCheckpoint(filepath=args.directory + '/weight-{epoch:02d}.h5')
    csv_logger = CSVLogger(filename=args.directory + '/logs.csv')
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint, csv_logger])


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_codes', type=int, required=True, help='Number of medical codes')
    parser.add_argument('--emb_size', type=int, default=40, help='Size of the embedding layer')
    parser.add_argument('--maxlen', type=int, default=3, help='Maximum size of LSTM')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--path_data_train', type=str, default='data/data_train.pkl', help='Path to train data')
    parser.add_argument('--path_target_train', type=str, default='data/target_train.pkl', help='Path to train target')
    parser.add_argument('--path_data_val', type=str, default='data/data_test.pkl', help='Path to val data')
    parser.add_argument('--path_target_val', type=str, default='data/target_test.pkl', help='Path to val target')
    parser.add_argument('--directory', type=str, default='./', help='Path to output models')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
