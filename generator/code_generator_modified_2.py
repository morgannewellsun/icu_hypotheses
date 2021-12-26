"""
This script:

1. Preprocesses data for sequence-to-sequence training as opposed to Daniel's
    script, which prepares data for sequence-to-timestep training. The data
    is sequentialized in the same way as Daniel's script.
2. Instantiates the attention LSTM from Daniel's paper
3. Trains the model
"""



import argparse
import pickle

import keras as k
import numpy as np
import pandas as pd
import tensorflow as tf


def read_data(path_patients, path_outcomes):
    """
    Read and return data from path.

    Arguments:
    path_patients: path to pickled patient medical history
    path_outcomes: path to pickled patient outcomes

    Returns:
    patients: list of patients; each is a list of visits; each is a list of medical codes
    outcomes: list of patient outcomes
    """
    patients = pd.read_pickle(path_patients)['codes'].values
    outcomes = pd.read_pickle(path_outcomes)['target'].values
    return patients, outcomes


def read_dictionary(path_dictionary):
    """
    Read and return medical code definition dictionary from path.
    
    Arguments:
    path_dictionary: path to medical code definition dictionary

    Returns:
    dictionary: dictionary of medical code definitions
    """
    with open(path_dictionary, "rb") as readfile:
        dictionary = pickle.load(readfile)
    return dictionary


def offset_medical_codes(patients, dictionary, offset):
    """
    Takes in a list of patient me as a list of visits.
    Add a fixed offset to all medical codes, and update the dictionary.

    Arguments:
    patients: list of patients; each is a list of visits; each is a list of medical codes
    dictionary (optional): dictionary of medical code definitions, or None
    offset: integer offset to add to all medical codes

    Returns:
    patients_offset: list of patients; each is a list of visits; each is a list of offset medical codes
    dictionary_offset: dictionary of offset medical code definitions, if input dictionary was provided
    """
    patients_offset = []
    for patient in patients:
        patient_offset = []
        for visit in patient:
            visit_offset = []
            for code in visit:
                visit_offset.append(code + offset)
            patient_offset.append(visit_offset)
        patients_offset.append(patient_offset)
    if dictionary is None:
        return patients_offset, None
    else:
        dictionary_offset = dict()
        for code, value in dictionary.items():
            dictionary_offset.update({code+offset: value})
        return patients_offset, dictionary_offset


def sequentialize_naive(patients, outcomes, dictionary):
    """
    Unnests patients into a flat sequences, and applies outcomes as medical codes.
    Does so in a manner which loses separation between visits.
    Does NOT pad or take subsequences.

    Arguments:
    patients: list of patients; each is a list of visits; each is a list of medical codes
    outcomes: list of patient outcomes
    dictionary (optional): dictionary of medical code definitions, or None

    Returns:
    patients_seq: list of patients; each is a list of medical codes including outcome codes
    dictionary_offset: dictionary of offset medical code definitions, if input dictionary was provided
    """

    # Offset all medical codes by 3 (0 = pad, 1 = survival, 2 = mortality)
    patients_offset, dictionary_offset = offset_medical_codes(patients, dictionary, offset=3)
    if dictionary is not None:
        dictionary_offset.update({
            0: "padding",
            1: "outcome_survival",
            2: "outcome_mortality"
        })

    # Flatten each patient into a sequence
    patients_seq = []
    for patient in patients_offset:
        patient_seq = []
        for visit in patient:
            patient_seq.extend(visit)
        patients_seq.append(patient_seq)

    # Append outcomes as medical codes
    for patient_seq, mortality in zip(patients_seq, outcomes):
        patient_seq.append(2 if mortality else 1)

    # Return
    return patients_seq, dictionary_offset


def pad_to_max_length(patients_seq, max_length):
    """
    Pads the input patient sequences to a length of max_length.
    Generates the appropriate corresponding masks.
    Outputs numpy arrays.

    Arguments:
    patients_seq: list of patients; each is a list of medical codes including outcome codes
    max_length: integer length, must be no less than length of longest list in patients_seq

    Returns:
    patients_padded: ndarray of shape (n_patients, max_length) containing padded patients_seq data
    masks: ndarray of shape (n_patients, max_length) containing int mask values (0 = masked, 1 = unmasked)
    """
    patients_padded = np.zeros(shape=(len(patients_seq), max_length), dtype=float)
    masks = np.zeros_like(patients_padded, dtype=float)
    for patient_index, patient_seq in enumerate(patients_seq):
        amount_to_pad = max_length - len(patient_seq)
        assert amount_to_pad >= 0
        patients_padded[patient_index, :len(patient_seq)] = patient_seq
        masks[patient_index, :len(patient_seq)] = 1
    return patients_padded, masks


def mask_initial_n(masks, initial_n):
    """
    Masks the initial n codes of each patient.
    The value of initial_n should match the length of probe data which will be used.

    Arguments:
    masks: ndarray of shape (n_patients, max_length) containing boolean mask values (False = mask)
    initial_n: number of codes to mask per patient

    Returns:
    masks: ndarray of shape (n_patients, max_length) containing int mask values (0 = masked, 1 = unmasked)
    """
    masks = np.copy(masks)
    masks[:, :initial_n] = 0
    return masks


def extract_xy_seq2seq(patients_padded, masks):
    """
    Extracts (x, y) pairs for sequence-to-sequence training.

    Arguments:
    patients_padded: ndarray of shape (n_patients, max_length) containing padded patients_seq data
    masks: ndarray of shape (n_patients, max_length) containing int mask values (0 = masked, 1 = unmasked)

    Returns:
    x: ndarray of shape (n_patients, max_length-1), containing input sequences
    y: ndarray of shape (n_patients, max_length-1), containing target sequences offset by 1 from x
    y_mask: ndarray of shape(n_patients, max_length-1), containing int masks for y (0 = masked, 1 = unmasked)
    """
    x = patients_padded[:, :-1]
    y = patients_padded[:, 1:]
    y_mask = masks[:, 1:]
    return x, y, y_mask


def masked_sparse_categorical_crossentropy(y_true_and_mask, y_pred):
    """
    A masked version of keras.losses.sparse_categorical_crossentropy.

    Arguments:
    y_true_and_mask: tensor of shape (batch_size, timesteps, 2) containing [y_true, y_mask]
    y_pred: tensor of shape (batch_size, timesteps, num_classes) predicted by the model

    Returns:
    losses_masked: tensor of shape (batch_size,), with masked losses zeroed out.
    """
    y_true = y_true_and_mask[:, :, 0]
    y_mask = y_true_and_mask[:, :, 1]
    losses = k.losses.sparse_categorical_crossentropy(y_true, y_pred)
    losses_masked = losses * y_mask
    return losses_masked


class MaskedSparseCategoricalAccuracy(k.metrics.Metric):
    """
    A masked version of keras.metrics.SparseCategoricalAccuracy.

    Arguments:
    y_true_and_mask: tensor of shape (batch_size, timesteps, 2) containing [y_true, y_mask]
    y_pred: tensor of shape (batch_size, timesteps, num_classes) predicted by the model
    sample_weight: tensor of shape (batch_size, timesteps) with 0 = masked, 1 = unmasked
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_total = tf.Variable(
            initial_value=tf.zeros(shape=(), dtype=tf.float32), trainable=False, name="n_total")
        self.n_correct = tf.Variable(
            initial_value=tf.zeros(shape=(), dtype=tf.float32), trainable=False, name="n_correct")

    def update_state(self, y_true_and_mask, y_pred, sample_weight=None):
        y_true = y_true_and_mask[:, :, 0]
        y_mask = y_true_and_mask[:, :, 1]
        self.n_total = self.n_total.assign_add(tf.reduce_sum(y_mask))
        self.n_correct = self.n_correct.assign_add(
            tf.reduce_sum(k.metrics.sparse_categorical_accuracy(y_true, y_pred) * y_mask))

    def result(self):
        return self.n_correct / self.n_total

    def reset_states(self):
        self.n_total.assign(tf.zeros(shape=(), dtype=tf.float32))
        self.n_correct.assign(tf.zeros(shape=(), dtype=tf.float32))


def main(
        *,
        path_patients_train, 
        path_outcomes_train, 
        path_patients_val, 
        path_outcomes_val,
        path_dictionary,
        path_output,
        max_length,
        probe_length,
        embed_size,
        attention_lstm_size,
        predictor_lstm_size,
        epochs, 
        batch_size):
    
    # Load data and dictionary
    patients_train, outcomes_train = read_data(path_patients_train, path_outcomes_train)
    patients_val, outcomes_val = read_data(path_patients_val, path_outcomes_val)
    dictionary = read_dictionary(path_dictionary)

    # Preprocess data
    patients_seq_train, dictionary = sequentialize_naive(patients_train, outcomes_train, dictionary)
    patients_seq_val, _ = sequentialize_naive(patients_val, outcomes_val, None)
    patients_padded_train, masks_train = pad_to_max_length(patients_seq_train, max_length)
    patients_padded_val, masks_val = pad_to_max_length(patients_seq_val, max_length)
    masks_train = mask_initial_n(masks_train, probe_length)
    masks_val = mask_initial_n(masks_val, probe_length)
    x_train, y_train, y_mask_train = extract_xy_seq2seq(patients_padded_train, masks_train)
    x_val, y_val, y_mask_val = extract_xy_seq2seq(patients_padded_val, masks_val)

    # Count number of medical codes after modifying dictionary
    n_codes = len(dictionary)

    # Determine length of input
    input_length = max_length - 1

    # Create model
    input_layer = k.layers.Input((input_length,), name='time_input')
    embedding = k.layers.Embedding(input_dim=n_codes, output_dim=embed_size)(input_layer)
    activations = k.layers.LSTM(
        attention_lstm_size, input_shape=(input_length, embed_size), return_sequences=True)(embedding)
    attention = k.layers.Dense(1, activation='tanh')(activations)
    attention = k.layers.Flatten()(attention)
    attention = k.layers.Activation('softmax')(attention)
    attention = k.layers.RepeatVector(embed_size)(attention)
    attention = k.layers.Permute([2, 1])(attention)
    sent_representation = k.layers.Multiply()([attention, embedding])
    attention_activations = k.layers.LSTM(
        predictor_lstm_size, input_shape=(input_length, embed_size), return_sequences=True)(sent_representation)
    predictions = k.layers.Dense(n_codes, activation='softmax')(attention_activations)
    model = k.models.Model(input=input_layer, output=predictions)

    # Compile and train model
    '''
    # Workaround for sparse loss functions and metrics in old tf/keras version
    # https://github.com/tensorflow/tensorflow/issues/17150
    # Note that this is only needed when using built-in sparse loss fns
    y_train = np.expand_dims(y_train, axis=-1)
    y_val = np.expand_dims(y_val, axis=-1)
    model.compile(
        # loss=masked_sparse_categorical_crossentropy,
        loss=k.losses.SparseCategoricalCrossentropy(),
        optimizer=k.optimizers.RMSprop(lr=0.01),  # TODO why not adam?,
        sample_weight_mode="temporal")
    model.fit(
        x=x_train, y=y_train, sample_weight=y_mask_train,
        validation_data=(x_val, y_val, y_mask_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            k.callbacks.ModelCheckpoint(filepath=path_output + '/weight-{epoch:02d}.h5'),
            k.callbacks.CSVLogger(filename=path_output + '/logs.csv')])
    '''

    model.compile(
        # loss=masked_sparse_categorical_crossentropy,
        loss=masked_sparse_categorical_crossentropy,
        metrics=[MaskedSparseCategoricalAccuracy(name="acc")],
        optimizer=k.optimizers.RMSprop(lr=0.01))  # TODO why not adam?
    model.fit(
        x=x_train, y=np.stack([y_train, y_mask_train], axis=-1),
        validation_data=(x_val, np.stack([y_val, y_mask_val], axis=-1)),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            k.callbacks.ModelCheckpoint(filepath=path_output + '/weight-{epoch:02d}.h5'),
            k.callbacks.CSVLogger(filename=path_output + '/logs.csv')])


def parse_arguments():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--path_patients_train', type=str, required=True)
    parser.add_argument('--path_outcomes_train', type=str, required=True)
    parser.add_argument('--path_patients_val', type=str, required=True)
    parser.add_argument('--path_outcomes_val', type=str, required=True)
    parser.add_argument('--path_dictionary', type=str, required=True)
    parser.add_argument('--path_output', type=str, required=True)

    parser.add_argument('--max_length', type=int, required=True)
    parser.add_argument('--probe_length', type=int, required=True)

    parser.add_argument('--embed_size', type=int, required=True)
    parser.add_argument('--attention_lstm_size', type=int, required=True)
    parser.add_argument('--predictor_lstm_size', type=int, required=True)

    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(

        path_patients_train=args.path_patients_train,
        path_outcomes_train=args.path_outcomes_train,
        path_patients_val=args.path_patients_val,
        path_outcomes_val=args.path_outcomes_val,
        path_dictionary=args.path_dictionary,
        path_output=args.path_output,

        max_length=args.max_length,
        probe_length=args.probe_length,

        embed_size=args.embed_size,
        attention_lstm_size=args.attention_lstm_size,
        predictor_lstm_size=args.predictor_lstm_size,

        epochs=args.epochs,
        batch_size=args.batch_size)
