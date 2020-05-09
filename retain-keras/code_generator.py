import os
import argparse
import keras
import numpy as np
import pandas as pd
from keras.models import Model
from keras import layers
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

def read_data(ARGS):
    """Read the data from provided paths and assign it into lists"""
    x_train = pd.read_pickle(ARGS.path_data_train)['codes'].values
    y_train = pd.read_pickle(ARGS.path_target_train)['target'].values

    return (x_train, y_train)

def process(data_train, y_train, num_codes, maxlen, simple):
    
    num_sequences = 0
    all_sequences = []
    for i, patient in enumerate(data_train):

        # get the code sequence into an interable list
        patient_sequence = [0] * (maxlen - 1) # zero pad
        for visit in patient:
            for code in visit:
                patient_sequence.append(code+1)

        if y_train[i] == 1:
            if simple:
                patient_sequence.append(12+1) # expired
            else:
                patient_sequence.append(63)
        else:
            if simple:
                patient_sequence.append(13+1)
            else:
                patient_sequence.append(64) # released

        num_sequences += len(patient_sequence) - maxlen

        all_sequences.append(patient_sequence)

    x = np.zeros((num_sequences, maxlen))
    y = np.zeros((num_sequences, num_codes))
    count = 0
    for patient in all_sequences:
        for i in range(len(patient)-maxlen):
            x[count,:] = patient[i:i+maxlen]
            y[count, int(patient[i+maxlen])] = 1
            count += 1

    print('Found %d sequences of length %d among %d patients' % (num_sequences, maxlen, len(all_sequences)))
    return x, y



def main(ARGS):

    maxlen = ARGS.maxlen
    embed_size = ARGS.emb_size
    num_codes = ARGS.num_codes
    epochs = ARGS.epochs
    batch_size = ARGS.batch_size

    print('Reading Data...')
    data_train, y_train = read_data(ARGS)

    print('Processing Data...')
    x, y = process(data_train, y_train, num_codes, maxlen, ARGS.simple)
    if ARGS.simple:
        termination = [12, 13]
        nothing = 14
    else:
        termination = [63, 64]
        nothing = 65

    print('Creating Model...')
    input_layer = layers.Input((maxlen,), name='time_input')
    #print(input_layer)
    embedding = layers.Embedding(input_dim=num_codes, output_dim=embed_size)(input_layer)
    #print(embedding)
    activations = layers.LSTM(128, input_shape=(maxlen, embed_size), return_sequences=True)(embedding)

    # compute importance for each step
    attention = layers.Dense(1, activation='tanh')(activations)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(embed_size)(attention)
    attention = layers.Permute([2, 1])(attention)

    sent_representation = layers.Multiply()([attention, embedding])

    attention_activations = layers.LSTM(128, input_shape=(maxlen, embed_size))(sent_representation)
    predictions = layers.Dense(num_codes, activation='softmax')(attention_activations)

    model = Model(input=input_layer, output=predictions)
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    

    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    print('Training Model...')

    temperature = 1.0 # hyperparam

    checkpoint = ModelCheckpoint(filepath=ARGS.directory+'/weight-{epoch:02d}.h5')

    if not ARGS.diagnosis:
        model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])
    else:
        for i in range(epochs):
            print('Epoch %d' % i)
            # Fit the model for 1 epoch on the available training data
            model.fit(x, y,
                      batch_size=batch_size,
                      epochs=1,
                      callbacks=[checkpoint])

            # get some random set of maxlen   
            example_sample = []
            sampled = np.zeros((1, maxlen))
            sampled[0, :] = np.random.randint(0, termination[0], maxlen)
            for s in sampled[0, :]:
                example_sample.append(int(s))
            
            # generate 27 characters or if termination hits
            for j in range(20):
                preds = model.predict(sampled, verbose = 0)[0]
                next_code = sample(preds, temperature)
                example_sample.append(next_code)
                if next_code in termination:
                    break
                sampled[0, :maxlen-1] = sampled[0, 1:]
                sampled[0, maxlen-1] = next_code
            print(example_sample)

            # see training error at the end of epoch
            total_loss = 0
            
            for m in range(x.shape[0]):
                train_preds = model.predict(x[m].reshape((1, maxlen)), verbose = 0)[0]
                total_loss += sample(train_preds, temperature) != np.argmax(y[m,:])

            print('Total loss %d out of %d' % (total_loss, x.shape[0]))

def parse_arguments(parser):
    """Read user arguments"""
    parser.add_argument('--num_codes', type=int, required=True,
                        help='Number of medical codes')
    parser.add_argument('--emb_size', type=int, default=40,
                        help='Size of the embedding layer')
    parser.add_argument('--maxlen', type=int, default=3,
                        help='Maximum size of LSTM')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--path_data_train', type=str, default='data/data_train.pkl',
                        help='Path to train data')
    parser.add_argument('--path_target_train', type=str, default='data/target_train.pkl',
                        help='Path to train target')
    parser.add_argument('--directory', type=str, default='./',
                        help='Path to output models')
    parser.add_argument('--diagnosis', type=bool, default=False,
                        help='Print for each epoch if True, but no checkpoint')
    parser.add_argument('--simple', type=bool, default=False,
                        help='If simple, then process differently')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = parse_arguments(PARSER)
    main(ARGS)