"""
This script:

1. Preprocesses data for sequence-to-sequence training as opposed to Daniel's
    script, which prepares data for sequence-to-timestep training. The data
    is sequentialized in the same way as Daniel's script.
2. Instantiates the attention LSTM from Daniel's paper.
3. Trains the model using a seq2seq method.
4. Uses the model to complete some probe sequences generated from test data.
5. Performs some analysis specific to interact6 to quantify the quality of the
    model's reconstruction of probe sequences.
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


def visits_to_seq_naive(patients, outcomes, dictionary):
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


def pad_to_length(patients_seq, max_length):
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
    initial_n: number of initial codes to mask per patient

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
    This is used instead of the sample_weights exposed by keras's API because
    of bugs and inconsistent behavior in this version of keras.

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
    This is used instead of the sample_weights exposed by keras's API because
    of bugs and inconsistent behavior in this version of keras.

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


def seq_to_visits_naive(predicted_seqs):
    """
    Restructures flat medical code sequences into lists of visits.
    Additionally undoes the medical code offset from visits_to_seq_naive.

    Arguments:
    predicted_seqs: ndarray of shape (n_patients, max_length)

    Returns:
    patients: list of patients; each is a list of visits; each is a list of medical codes
    outcomes: list of patient outcomes
    """
    outcome_codes = {1, 2}
    patients = []
    outcomes = []
    for predicted_seq in predicted_seqs:
        patient = []
        visit = []
        outcome = None
        for code in predicted_seq:
            if code in outcome_codes:
                outcome = True if code == 2 else False
                break
            elif (len(visit) == 0) or (code > visit[-1] + 3):
                visit.append(int(code - 3))
            else:
                patient.append(visit)
                visit = [int(code - 3)]
        if len(visit) > 0:
            patient.append(visit)
        patients.append(patient)
        outcomes.append(outcome)
    return patients, outcomes


def visits_to_multihot(patients, n_codes):
    """
    Restructures list-of-visits patient data format to multihot ndarray format.

    Arguments:
    patients: list of patients; each is a list of visits; each is a list of medical codes
    n_codes: number of medical codes to expect

    Returns:
    patients_multihot: ndarray of size (n_patients, n_visits, n_codes)
    """
    max_n_visits = max([len(visits) for visits in patients])
    patients_multihot = np.zeros(shape=(len(patients), max_n_visits, n_codes), dtype=int)
    for patient_index, patient in enumerate(patients):
        for visit_index, visit in enumerate(patient):
            for code in visit:
                patients_multihot[patient_index, visit_index, code] = 1
    return patients_multihot


def analyze_run_length(patients, periods, n_codes):
    """
    Quantifies the deviation from expected interact6 data.

    Arguments:
    patients: list of patients; each is a list of visits; each is a list of medical codes
    periods: list of expected periods for each medical code
    n_codes: number of medical codes to expect
    initial_n: number of initial codes to mask

    Returns:
    rms_deviations: RMS of deviations from expected run lengths
    """

    # Convert patient visits to multihot per visit
    patients_multihot = visits_to_multihot(patients, n_codes)

    # Count run lengths, and mask the first and last runs with zeros. Example:
    # patients_multihot[0, :, 0] = [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]
    #       run_lengths[0, :, 0] = [0, 0, 0, 4, 4, 4, 4, 2, 2, 1, 1, 0, 0, 0]
    run_bounds_bool = patients_multihot[:, 1:] != patients_multihot[:, :-1]
    run_lengths = np.zeros_like(patients_multihot)
    for patient_index in range(len(patients)):
        for code_index in range(n_codes):
            patient_code_run_bounds_indices = np.concatenate([
                [-1],
                np.where(run_bounds_bool[patient_index, :, code_index])[0]])
            patient_code_run_lengths = patient_code_run_bounds_indices[1:] - patient_code_run_bounds_indices[:-1]
            patient_code_run_lengths_rep = np.repeat(patient_code_run_lengths, patient_code_run_lengths)
            run_lengths[patient_index, :len(patient_code_run_lengths_rep), code_index] = patient_code_run_lengths_rep
            run_lengths[patient_index, :patient_code_run_lengths_rep[0], code_index] = 0

    # for sanity check
    # print(patients_multihot[0])
    # print(run_lengths[0])

    # Compute masked average deviation from expected run_lengths
    for period in periods:
        assert period % 2 == 0  # too lazy to deal with odd periods
    expected_run_lengths = [period // 2 for period in periods]
    expected_run_lengths = np.array(expected_run_lengths)
    expected_run_lengths = expected_run_lengths.reshape((1, 1, -1))
    squared_deviations = np.square(run_lengths - expected_run_lengths)
    mask = (run_lengths != 0).astype(int)
    rms_deviations = np.sqrt(np.sum(squared_deviations * mask, axis=(0, 1)) / np.sum(mask, axis=(0, 1)))
    return rms_deviations


def main(
        *,
        path_patients_train, 
        path_outcomes_train, 
        path_patients_val, 
        path_outcomes_val,
        path_patients_test,
        path_outcomes_test,
        path_dictionary,
        path_output,
        max_length,
        probe_length,
        embed_size,
        attention_lstm_size,
        predictor_lstm_size,
        epochs, 
        batch_size,
        verbose):
    
    # Load data, and dictionary
    patients_train, outcomes_train = read_data(path_patients_train, path_outcomes_train)
    patients_val, outcomes_val = read_data(path_patients_val, path_outcomes_val)
    patients_test, outcomes_test = read_data(path_patients_test, path_outcomes_test)
    dictionary = read_dictionary(path_dictionary)

    # Count number of medical codes before modifying dictionary
    n_codes_original = len(dictionary)

    # Preprocess data
    patients_seq_train, dictionary = visits_to_seq_naive(patients_train, outcomes_train, dictionary)
    patients_seq_val, _ = visits_to_seq_naive(patients_val, outcomes_val, None)
    patients_seq_test, _ = visits_to_seq_naive(patients_test, outcomes_test, None)
    if verbose >= 1:
        max_length_in_data = max(
            [len(s) for s in patients_seq_train]
            + [len(s) for s in patients_seq_val]
            + [len(s) for s in patients_seq_test])
        min_length_in_data = min(
            [len(s) for s in patients_seq_train]
            + [len(s) for s in patients_seq_val]
            + [len(s) for s in patients_seq_test])
        print(f"Maximum sequentialized length in all data: {max_length_in_data}.")
        print(f"Minimum sequentialized length in all data: {min_length_in_data}.")
        print(f"Training data contains {len(patients_seq_train)} sequences.")
        print(f"Validation data contains {len(patients_seq_val)} sequences.")
        print(f"Test data contains {len(patients_seq_test)} sequences.")
    patients_padded_train, masks_train = pad_to_length(patients_seq_train, max_length)
    patients_padded_val, masks_val = pad_to_length(patients_seq_val, max_length)
    patients_padded_test, masks_test = pad_to_length(patients_seq_test, max_length)
    masks_train = mask_initial_n(masks_train, probe_length)
    masks_val = mask_initial_n(masks_val, probe_length)
    masks_test = mask_initial_n(masks_train, probe_length)
    x_train, y_train, y_mask_train = extract_xy_seq2seq(patients_padded_train, masks_train)
    x_val, y_val, y_mask_val = extract_xy_seq2seq(patients_padded_val, masks_val)
    x_test, y_test, y_mask_test = extract_xy_seq2seq(patients_padded_test, masks_test)

    # Count number of medical codes after modifying dictionary
    n_codes_augmented = len(dictionary)

    # Determine length of input
    input_length = max_length - 1

    # Create model
    input_layer = k.layers.Input((input_length,), name='time_input')
    embedding = k.layers.Embedding(input_dim=n_codes_augmented, output_dim=embed_size)(input_layer)
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
    predictions = k.layers.Dense(n_codes_augmented, activation='softmax')(attention_activations)
    model = k.models.Model(input=input_layer, output=predictions)

    # Compile and train model
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
            k.callbacks.ModelCheckpoint(
                filepath=path_output + '/weight-best.h5',
                monitor='val_acc',
                verbose=1,
                save_best_only=True,
                mode='max'),
            k.callbacks.CSVLogger(filename=path_output + '/logs.csv')])

    # Reload the best weights
    model.load_weights(path_output + '/weight-best.h5')

    # Reconstruct sequences data from probe sequences derived from test data
    probe_seqs = x_test[:, :probe_length]
    probe_seqs, _ = pad_to_length(probe_seqs, input_length)
    assert len(probe_seqs) % batch_size == 0  # too lazy to deal with partial batches rn
    probe_seq_batches = np.split(probe_seqs, int(len(probe_seqs) / batch_size), axis=0)
    predicted_seq_batches = []
    for probe_seq_batch in probe_seq_batches:
        for timestep in range(probe_length, input_length+1):
            pred_onehot_batch = model.predict(probe_seq_batch, verbose=0)
            pred_onehot_batch[:, :, 0] = np.NINF  # don't predict padding
            pred_indices_batch = np.argmax(pred_onehot_batch, axis=-1)
            if timestep != input_length:
                probe_seq_batch[:, timestep] = pred_indices_batch[:, timestep-1]
            else:
                predicted_seq_batches.append(
                    np.concatenate([probe_seq_batch, pred_indices_batch[:, timestep-1:timestep]], axis=1))
    predicted_seqs = np.concatenate(predicted_seq_batches, axis=0)

    # Group predictions into visits
    predicted_patients, _ = seq_to_visits_naive(predicted_seqs)

    # Analyze run lengths in predictions
    rms_deviations = analyze_run_length(
        patients=predicted_patients, periods=[4, 6, 10, 14], n_codes=n_codes_original)
    print(rms_deviations)


def parse_arguments():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--path_patients_train', type=str, required=True)
    parser.add_argument('--path_outcomes_train', type=str, required=True)
    parser.add_argument('--path_patients_val', type=str, required=True)
    parser.add_argument('--path_outcomes_val', type=str, required=True)
    parser.add_argument('--path_patients_test', type=str, required=True)
    parser.add_argument('--path_outcomes_test', type=str, required=True)
    parser.add_argument('--path_dictionary', type=str, required=True)
    parser.add_argument('--path_output', type=str, required=True)

    parser.add_argument('--max_length', type=int, required=True)
    parser.add_argument('--probe_length', type=int, required=True)

    parser.add_argument('--embed_size', type=int, required=True)
    parser.add_argument('--attention_lstm_size', type=int, required=True)
    parser.add_argument('--predictor_lstm_size', type=int, required=True)

    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)

    parser.add_argument('--verbose', type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(

        path_patients_train=args.path_patients_train,
        path_outcomes_train=args.path_outcomes_train,
        path_patients_val=args.path_patients_val,
        path_outcomes_val=args.path_outcomes_val,
        path_patients_test=args.path_patients_test,
        path_outcomes_test=args.path_outcomes_test,
        path_dictionary=args.path_dictionary,
        path_output=args.path_output,

        max_length=args.max_length,
        probe_length=args.probe_length,

        embed_size=args.embed_size,
        attention_lstm_size=args.attention_lstm_size,
        predictor_lstm_size=args.predictor_lstm_size,

        epochs=args.epochs,
        batch_size=args.batch_size,

        verbose=args.verbose)
