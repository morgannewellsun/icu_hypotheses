
import argparse

import keras as k
import numpy as np
import pandas as pd
import tensorflow as tf


def masked_sparse_categorical_crossentropy(y_true_and_mask, y_pred):
    """
    A masked version of keras.losses.sparse_categorical_crossentropy.
    This is used instead of the sample_weights exposed by keras's API because
    of bugs and inconsistent behavior in this version of keras.

    Arguments:
    y_true_and_mask: tensor of shape (batch_size, timesteps, 2) containing [y_true, y_mask]
    y_pred: tensor of shape (batch_size, timesteps, num_classes) predicted by the model

    Returns:
    losses_masked: tensor of shape (batch_size, timesteps), with masked losses zeroed out.
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


def main(preprocessed_mimic_filepath, output_directory):

    # =========================================================================
    # Hyperparameters
    # =========================================================================

    # data preparation
    use_truncated_codes = True
    proportion_event_instances = 0.9  # {0.5, 0.8, 0.9, 0.95, 0.99}
    medications_per_patient_incl_min = 15
    use_separator_token = False
    probe_length = 3
    train_val_test_splits = (0.8, 0.2, 0.0)

    # model architecture and training
    input_length = 15  # number of medications
    truncate_whole_visits = False
    embed_size = 40
    attention_lstm_size = 128
    predictor_lstm_size = 128
    batch_size = 128
    n_epochs = 600

    # checks and derived hyperparameters
    event_code_col_name = "event_code_trunc" if use_truncated_codes else "event_code_full"
    event_code_count_col_name = "event_code_trunc_count" if use_truncated_codes else "event_code_full_count"
    assert proportion_event_instances in {0.5, 0.8, 0.9, 0.95, 0.99}
    selector_dict = {
        (True, 0.5): 8993,
        (True, 0.8): 2107,
        (True, 0.9): 825,
        (True, 0.95): 376,
        (True, 0.99): 65,
        (False, 0.5): 8474,
        (False, 0.8): 1931,
        (False, 0.9): 780,
        (False, 0.95): 350,
        (False, 0.99): 59}  # values derived from cumulative_event_count_medications plots
    event_code_count_incl_min = selector_dict[(use_truncated_codes, proportion_event_instances)]
    assert sum(train_val_test_splits) == 1

    # =========================================================================
    # Data loading and preparation
    # =========================================================================

    print("[INFO] Loading and preparing data")

    # load medication data
    mimic_df = pd.read_csv(preprocessed_mimic_filepath)
    mimic_df = mimic_df[mimic_df["event_type"] == "M"]

    # keep only most common medications
    mimic_df = mimic_df[mimic_df[event_code_count_col_name] >= event_code_count_incl_min]

    # keep only patients with enough medications
    mimic_df = mimic_df[mimic_df["patient_medications_count"] >= medications_per_patient_incl_min]

    # map string NDC codes to integer indices
    code_idx_to_str_map = ["PADDING", "OUTCOME_SURVIVAL", "OUTCOME_MORTALITY"]
    if use_separator_token:
        code_idx_to_str_map.append("SEPARATOR")
    code_idx_to_str_map.extend(list(mimic_df[event_code_col_name].value_counts().index))
    n_codes = len(code_idx_to_str_map)
    code_str_to_idx_map = dict([(code_str, code_idx) for code_idx, code_str in enumerate(code_idx_to_str_map)])
    mimic_df["event_code_idx"] = mimic_df[event_code_col_name].map(code_str_to_idx_map)

    # unpack dataframe into nested list: [patient_idx, admission_idx, event_idx]
    patients_nested = []
    patient_mortalities = []
    for patient_id, patient_df in mimic_df.groupby("patient_id", sort=False):
        patient_mortalities.append(patient_df["patient_mortality"].iloc[0])
        patient_admissions = []
        for admission_id, admission_df in patient_df.groupby("admission_id", sort=False):
            patient_admissions.append(list(admission_df["event_code_idx"]))
        patients_nested.append(patient_admissions)

    print(f"[INFO] Number of different medical codes: {n_codes}")
    print(f"[INFO] Number of patients: {len(patients_nested)}")
    print(f"[INFO] Average patient mortality rate: {np.mean(patient_mortalities)}")

    # truncate oldest visits from each patient so the history fits inside input_length+1
    # this block of code performs truncation in whole-visit chunks
    if truncate_whole_visits:
        patients_nested_truncated = []
        for patient_admissions in patients_nested:
            patient_admissions_truncated_reversed = []
            patient_num_tokens = 1  # all patients contain an outcome token
            for admission_events in patient_admissions[::-1]:
                patient_num_tokens += len(admission_events) + (1 if use_separator_token else 0)
                if patient_num_tokens <= input_length + 1:
                    patient_admissions_truncated_reversed.append(admission_events)
                else:
                    break
            patients_nested_truncated.append(patient_admissions_truncated_reversed[::-1])
    else:
        patients_nested_truncated = patients_nested

    # flatten each patient into a sequence: [patient_idx, event_idx]
    # include separator tokens between visits, as well as a token representing final outcome
    patients_seq = []
    for patient_admissions, patient_mortality in zip(patients_nested_truncated, patient_mortalities):
        patient_seq = []
        for admission_events in patient_admissions:
            patient_seq.extend(admission_events)
            if use_separator_token:
                patient_seq.append(code_str_to_idx_map["SEPARATOR"])
        patient_seq.append(
            code_str_to_idx_map["OUTCOME_MORTALITY"] if patient_mortality else code_str_to_idx_map["OUTCOME_SURVIVAL"])
        patients_seq.append(patient_seq)

    # truncate oldest visits from each patient so the history fits inside input_length+1
    # this block of code performs truncation down to the individual code
    if not truncate_whole_visits:
        patients_seq_truncated = []
        for patient_seq in patients_seq:
            patients_seq_truncated.append(patient_seq[-1 * (input_length + 1):])
        patients_seq = patients_seq_truncated

    # determine accuracy of a strategy where the most common code is always guessed
    patients_seq_flattened = []
    for patient_seq in patients_seq:
        patients_seq_flattened.extend(patient_seq[probe_length:])
    _, counts = np.unique(patients_seq_flattened, return_counts=True)
    naive_strategy_accuracy = np.max(counts) / len(patients_seq_flattened)
    print(f"[INFO] Accuracy of an 'always guess most common code' strategy: {naive_strategy_accuracy}")

    # pad flattened sequences to length input_length+1
    # the +1 is because x and y will be extracted from this array using [:-1] and [1:]
    patients_seq_np = np.full(
        shape=(len(patients_seq), input_length + 1),
        fill_value=code_str_to_idx_map["PADDING"],
        dtype=float)
    masks_np = np.zeros_like(patients_seq_np, dtype=float)
    for patient_idx, patient_seq in enumerate(patients_seq):
        patients_seq_np[patient_idx, :len(patient_seq)] = patient_seq
        masks_np[patient_idx, :len(patient_seq)] = 1

    # mask the first probe_length codes
    masks_np[:, :probe_length] = 0

    # extract x and y sequences of length input_length
    patients_seq_x_np = patients_seq_np[:, :-1]
    patients_seq_y_np = patients_seq_np[:, 1:]
    masks_y_np = masks_np[:, 1:]

    # split train/val/test
    n_train = int(len(patients_seq_x_np) * train_val_test_splits[0])
    n_val = int(len(patients_seq_x_np) * train_val_test_splits[1])
    n_test = int(len(patients_seq_x_np) - (n_train + n_val))
    rng = np.random.default_rng()
    train_val_test_indexer = np.array([0] * n_train + [1] * n_val + [2] * n_test)
    rng.shuffle(train_val_test_indexer)  # in-place
    patients_seq_x_train_np = patients_seq_x_np[train_val_test_indexer == 0]
    patients_seq_y_train_np = patients_seq_y_np[train_val_test_indexer == 0]
    masks_y_train_np = masks_y_np[train_val_test_indexer == 0]
    patients_seq_x_val_np = patients_seq_x_np[train_val_test_indexer == 1]
    patients_seq_y_val_np = patients_seq_y_np[train_val_test_indexer == 1]
    masks_y_val_np = masks_y_np[train_val_test_indexer == 1]
    patients_seq_x_test_np = patients_seq_x_np[train_val_test_indexer == 2]
    patients_seq_y_test_np = patients_seq_y_np[train_val_test_indexer == 2]
    masks_y_test_np = masks_y_np[train_val_test_indexer == 2]

    # =========================================================================
    # Model construction
    # =========================================================================

    print("[INFO] Constructing model")

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

    # =========================================================================
    # Model compilation and training
    # =========================================================================

    print("[INFO] Training model")

    # compile and train model
    model.compile(
        loss=masked_sparse_categorical_crossentropy,
        metrics=[MaskedSparseCategoricalAccuracy(name="acc")],
        optimizer=k.optimizers.RMSprop(lr=0.01))  # TODO why not adam?
    model.fit(
        x=patients_seq_x_train_np, y=np.stack([patients_seq_y_train_np, masks_y_train_np], axis=-1),
        validation_data=(patients_seq_x_val_np, np.stack([patients_seq_y_val_np, masks_y_val_np], axis=-1)),
        batch_size=batch_size,
        epochs=n_epochs,
        callbacks=[
            k.callbacks.ModelCheckpoint(filepath=output_directory + '/weight-{epoch:02d}.h5'),
            k.callbacks.ModelCheckpoint(
                filepath=output_directory + '/weight-best.h5',
                monitor='val_acc',
                verbose=1,
                save_best_only=True,
                mode='max'),
            k.callbacks.CSVLogger(filename=output_directory + '/logs.csv')])

    # reload the best weights
    model.load_weights(output_directory + '/weight-best.h5')


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--preprocessed_mimic_filepath', type=str, required=True)
    parser.add_argument('--output_directory', type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args.preprocessed_mimic_filepath, args.output_directory)
