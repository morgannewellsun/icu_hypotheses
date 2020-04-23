import os
import argparse
import keras
import numpy as np
import pdb
from keras.models import Model
from keras import layers
from keras.optimizers import RMSprop

'''path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('Corpus length:', len(text))

# Length of extracted character sequences
maxlen = 60
embed_size = 100

# We sample a new sequence every `step` characters
step = 3

# This holds our extracted sequences
sentences = []

# This holds the targets (the follow-up characters)
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('Number of sequences:', len(sentences))

# List of unique characters in the corpus
chars = sorted(list(set(text)))
print('Unique characters:', len(chars))
# Dictionary mapping unique characters to their index in `chars`
char_indices = dict((char, chars.index(char)) for char in chars)

# Next, one-hot encode the characters into binary arrays.
print('Vectorization...')


x = np.zeros((len(sentences), maxlen))
y = np.zeros((len(sentences), len(chars)))
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t] = char_indices[char]
    y[i, char_indices[next_chars[i]]] = 1'''


def read_data(ARGS):
    """Read the data from provided paths and assign it into lists"""
    data_train_df = pd.read_pickle(ARGS.path_data_train)
    y_train = pd.read_pickle(ARGS.path_target_train)['target'].values
    data_output_train = [data_train_df['codes'].values]

    return (data_output_train, y_train)

def codify(data_train, y_train, num_codes, maxlen = 3):
    # preprocess
    processed_patients = []
    sample_size = 0
    for i, patient in enumerate(data_train):
        if len(patient) < maxlen:
            continue
        sample_size += len(patient) - maxlen
        processed_patient = []
        for visit in patient:
            if len(visit) == 0:
                processed_patient.append(82) # nothing happened

            single_code = 0
            for code in visit:
                if code < 3:
                    single_code += code
                elif code < 6:
                    single_code += 3 * code
                elif code < 9:
                    single_code += 9 * code
                elif code == 10:
                    single_code += 27
                elif code == 11:
                    single_code += 27 * 2
            processed_patient.append(single_code)
        # put in 1 more code for how it ended
        if y_train[i] == 1:
            processed_patient.append(83)
        else:
            processed_patient.append(84)
        processed_patients.append(processed_patient)

    # process and output
    x = np.zeros((sample_size, maxlen))
    y = np.zeros((sample_size, num_codes))
    for patient in processed_patients:
        for i in range(len(patient)-maxlen):
            x[i,:] = patient[i:i+maxlen]
            y[i, patient[i+maxlen]] = 1
    return x, y


def main(ARGS):

    maxlen = ARGS.maxlen
    embed_size = ARGS.emb_size
    num_codes = ARGS.num_codes

    print('Reading Data...')
    data_train, y_train = read_data(ARGS)

    print('Codifying Data...')
    x, y = codify(data_train, y_train, num_codes, maxlen)

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

    import random
    import sys

    print('Training Model...')
    for epoch in range(1, 60):
        print('epoch', epoch)
        # Fit the model for 1 epoch on the available training data
        model.fit(x, y,
                  batch_size=128,
                  epochs=1)

        # Select a text seed at random
        start_index = random.randint(0, len(text) - maxlen - 1)
        generated_text = text[start_index: start_index + maxlen]
        print('--- Generating with seed: "' + generated_text + '"')

        for temperature in [0.2, 0.5, 1.0, 1.2]:
            print('------ temperature:', temperature)
            sys.stdout.write(generated_text)

            # We generate 400 characters
            for i in range(400):
                sampled = np.zeros((1, maxlen))
                for t, char in enumerate(generated_text):
                    sampled[0, t] = char_indices[char]

                preds = model.predict(sampled, verbose=0)[0]
                next_index = sample(preds, temperature)
                next_char = chars[next_index]

                generated_text += next_char
                generated_text = generated_text[1:]

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

def parse_arguments(parser):
    """Read user arguments"""
    parser.add_argument('--num_codes', type=int, required=True,
                        help='Number of medical codes')
    parser.add_argument('--emb_size', type=int, default=10,
                        help='Size of the embedding layer')
    parser.add_argument('--maxlen', type=int, default=3,
                        help='Size of the embedding layer')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--path_data_train', type=str, default='data/data_train.pkl',
                        help='Path to train data')
    parser.add_argument('--path_target_train', type=str, default='data/target_train.pkl',
                        help='Path to train target')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = parse_arguments(PARSER)
    main(ARGS)