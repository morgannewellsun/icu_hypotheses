import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.layers as L
from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.preprocessing import sequence
from keras.utils.data_utils import Sequence
from keras.regularizers import l2
from keras.constraints import non_neg, Constraint
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


# =============================================================================
# Original retain_train.py code from Edward Choi
# =============================================================================

class SequenceBuilder(Sequence):
    """Generate Batches of data"""

    def __init__(self, data, target, batch_size, ARGS, target_out=True):
        # Receive all appropriate data
        self.codes = data[0]
        index = 1
        if ARGS.numeric_size:
            self.numeric = data[index]
            index += 1

        if ARGS.use_time:
            self.time = data[index]

        self.num_codes = ARGS.num_codes
        self.target = target
        self.batch_size = batch_size
        self.target_out = target_out
        self.numeric_size = ARGS.numeric_size
        self.use_time = ARGS.use_time
        self.n_steps = ARGS.n_steps
        # self.balance = (1-(float(sum(target))/len(target)))/(float(sum(target))/len(target))

    def __len__(self):
        """Compute number of batches.
        Add extra batch if the data doesn't exactly divide into batches
        """
        if len(self.codes) % self.batch_size == 0:
            return len(self.codes) // self.batch_size
        return len(self.codes) // self.batch_size + 1

    def __getitem__(self, idx):
        """Get batch of specific index"""

        def pad_data(data, length_visits, length_codes, pad_value=0):
            """Pad data to desired number of visiits and codes inside each visit"""
            zeros = np.full((len(data), length_visits, length_codes), pad_value)
            for steps, mat in zip(data, zeros):
                if steps != [[-1]]:
                    for step, mhot in zip(steps, mat[-len(steps):]):
                        # Populate the data into the appropriate visit
                        mhot[:len(step)] = step

            return zeros

        # Compute reusable batch slice
        batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        x_codes = self.codes[batch_slice]
        # Max number of visits and codes inside the visit for this batch
        pad_length_visits = min(max(map(len, x_codes)), self.n_steps)
        pad_length_codes = max(map(lambda x: max(map(len, x)), x_codes))
        # Number of elements in a batch (useful in case of partial batches)
        length_batch = len(x_codes)
        # Pad data
        x_codes = pad_data(x_codes, pad_length_visits, pad_length_codes, self.num_codes)
        outputs = [x_codes]
        # Add numeric data if necessary
        if self.numeric_size:
            x_numeric = self.numeric[batch_slice]
            x_numeric = pad_data(x_numeric, pad_length_visits, self.numeric_size, -99.0)
            outputs.append(x_numeric)
        # Add time data if necessary
        if self.use_time:
            x_time = sequence.pad_sequences(self.time[batch_slice],
                                            dtype=np.float32, maxlen=pad_length_visits,
                                            value=+99).reshape(length_batch, pad_length_visits, 1)
            outputs.append(x_time)

        # Add target if necessary (training vs validation)
        if self.target_out:
            target = self.target[batch_slice].reshape(length_batch, 1, 1)
            # sample_weights = (target*(self.balance-1)+1).reshape(length_batch, 1)
            # In our experiments sample weights provided worse results
            return (outputs, target)

        return outputs


class FreezePadding_Non_Negative(Constraint):
    """Freezes the last weight to be near 0 and prevents non-negative embeddings"""

    def __call__(self, w):
        other_weights = K.cast(K.greater_equal(w, 0)[:-1], K.floatx())
        last_weight = K.cast(K.equal(K.reshape(w[-1, :], (1, K.shape(w)[1])), 0.), K.floatx())
        appended = K.concatenate([other_weights, last_weight], axis=0)
        w *= appended
        return w


class FreezePadding(Constraint):
    """Freezes the last weight to be near 0."""

    def __call__(self, w):
        other_weights = K.cast(K.ones(K.shape(w))[:-1], K.floatx())
        last_weight = K.cast(K.equal(K.reshape(w[-1, :], (1, K.shape(w)[1])), 0.), K.floatx())
        appended = K.concatenate([other_weights, last_weight], axis=0)
        w *= appended
        return w


def read_data(ARGS):
    """Read the data from provided paths and assign it into lists"""
    data_train_df = pd.read_pickle(ARGS.path_data_train)
    data_test_df = pd.read_pickle(ARGS.path_data_test)
    y_train = pd.read_pickle(ARGS.path_target_train)['target'].values
    y_test = pd.read_pickle(ARGS.path_target_test)['target'].values
    data_output_train = [data_train_df['codes'].values]
    data_output_test = [data_test_df['codes'].values]

    if ARGS.numeric_size:
        data_output_train.append(data_train_df['numerics'].values)
        data_output_test.append(data_test_df['numerics'].values)
    if ARGS.use_time:
        data_output_train.append(data_train_df['to_event'].values)
        data_output_test.append(data_test_df['to_event'].values)
    return (data_output_train, y_train, data_output_test, y_test)


def model_create(ARGS):
    """Create and Compile model and assign it to provided devices"""

    def retain(ARGS):
        """Create the model"""

        # Define the constant for model saving
        reshape_size = ARGS.emb_size + ARGS.numeric_size
        if ARGS.allow_negative:
            embeddings_constraint = FreezePadding()
            beta_activation = 'tanh'
            output_constraint = None
        else:
            embeddings_constraint = FreezePadding_Non_Negative()
            beta_activation = 'sigmoid'
            output_constraint = non_neg()

        def reshape(data):
            """Reshape the context vectors to 3D vector"""
            return K.reshape(x=data, shape=(K.shape(data)[0], 1, reshape_size))

        # Code Input
        codes = L.Input((None, None), name='codes_input')
        inputs_list = [codes]
        # Calculate embedding for each code and sum them to a visit level
        codes_embs_total = L.Embedding(ARGS.num_codes + 1,
                                       ARGS.emb_size,
                                       name='embedding',
                                       embeddings_constraint=embeddings_constraint)(codes)
        codes_embs = L.Lambda(lambda x: K.sum(x, axis=2))(codes_embs_total)
        # Numeric input if needed
        if ARGS.numeric_size:
            numerics = L.Input((None, ARGS.numeric_size), name='numeric_input')
            inputs_list.append(numerics)
            full_embs = L.concatenate([codes_embs, numerics], name='catInp')
        else:
            full_embs = codes_embs

        # Apply dropout on inputs
        full_embs = L.Dropout(ARGS.dropout_input)(full_embs)

        # Time input if needed
        if ARGS.use_time:
            time = L.Input((None, 1), name='time_input')
            inputs_list.append(time)
            time_embs = L.concatenate([full_embs, time], name='catInp2')
        else:
            time_embs = full_embs

        # Setup Layers
        # This implementation uses Bidirectional LSTM instead of reverse order
        #    (see https://github.com/mp2893/retain/issues/3 for more details)

        alpha = L.Bidirectional(L.LSTM(ARGS.recurrent_size,
                                       return_sequences=True, implementation=2),
                                name='alpha')
        beta = L.Bidirectional(L.LSTM(ARGS.recurrent_size,
                                      return_sequences=True, implementation=2),
                               name='beta')

        alpha_dense = L.Dense(1, kernel_regularizer=l2(ARGS.l2))
        beta_dense = L.Dense(ARGS.emb_size + ARGS.numeric_size,
                             activation=beta_activation, kernel_regularizer=l2(ARGS.l2))

        # Compute alpha, visit attention
        alpha_out = alpha(time_embs)
        alpha_out = L.TimeDistributed(alpha_dense, name='alpha_dense_0')(alpha_out)
        alpha_out = L.Softmax(axis=1)(alpha_out)
        # Compute beta, codes attention
        beta_out = beta(time_embs)
        beta_out = L.TimeDistributed(beta_dense, name='beta_dense_0')(beta_out)
        # Compute context vector based on attentions and embeddings
        c_t = L.Multiply()([alpha_out, beta_out, full_embs])
        c_t = L.Lambda(lambda x: K.sum(x, axis=1))(c_t)
        # Reshape to 3d vector for consistency between Many to Many and Many to One implementations
        contexts = L.Lambda(reshape)(c_t)

        # Make a prediction
        contexts = L.Dropout(ARGS.dropout_context)(contexts)
        output_layer = L.Dense(1, activation='sigmoid', name='dOut',
                               kernel_regularizer=l2(ARGS.l2), kernel_constraint=output_constraint)

        # TimeDistributed is used for consistency
        # between Many to Many and Many to One implementations
        output = L.TimeDistributed(output_layer, name='time_distributed_out')(contexts)
        # Define the model with appropriate inputs
        model = Model(inputs=inputs_list, outputs=[output])

        return model

    # Set Tensorflow to grow GPU memory consumption instead of grabbing all of it at once
    K.clear_session()
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    tfsess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(tfsess)
    # If there are multiple GPUs set up a multi-gpu model
    model_final = retain(ARGS)

    # Compile the model - adamax has produced best results in our experiments
    model_final.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'],
                        sample_weight_mode="temporal")

    return model_final


def create_callbacks(model, data, ARGS):
    """Create the checkpoint and logging callbacks"""

    class LogEval(Callback):
        """Logging Callback"""

        def __init__(self, filepath, model, data, ARGS, interval=1):

            super(Callback, self).__init__()
            self.filepath = filepath
            self.interval = interval
            self.data_test, self.y_test = data
            self.generator = SequenceBuilder(data=self.data_test, target=self.y_test,
                                             batch_size=ARGS.batch_size, ARGS=ARGS,
                                             target_out=False)
            self.model = model

        def on_epoch_end(self, epoch, logs={}):
            # Compute ROC-AUC and average precision the validation data every interval epochs
            if epoch % self.interval == 0:
                # Compute predictions of the model
                y_pred = [x[-1] for x in
                          self.model.predict_generator(self.generator,
                                                       verbose=0,
                                                       use_multiprocessing=False,
                                                       workers=ARGS.workers,
                                                       max_queue_size=5)]
                score_roc = roc_auc_score(self.y_test, y_pred)
                score_pr = average_precision_score(self.y_test, y_pred)
                # Create log files if it doesn't exist, otherwise write to it
                if os.path.exists(self.filepath):
                    append_write = 'a'
                else:
                    append_write = 'w'
                with open(self.filepath, append_write) as file_output:
                    file_output.write("\nEpoch: {:d}- ROC-AUC: {:.6f} ; PR-AUC: {:.6f}" \
                                      .format(epoch, score_roc, score_pr))

                print("\nEpoch: {:d} - ROC-AUC: {:.6f} PR-AUC: {:.6f}" \
                      .format(epoch, score_roc, score_pr))

    # Create callbacks
    checkpoint = ModelCheckpoint(filepath=ARGS.directory + '/weight-{epoch:02d}.h5')
    log = LogEval(ARGS.directory + '/log.txt', model, data, ARGS)
    return (checkpoint, log)


def train_model(model, data_train, y_train, data_test, y_test, ARGS):
    """Train the Model with appropriate callbacks and generator"""
    checkpoint, log = create_callbacks(model, (data_test, y_test), ARGS)
    train_generator = SequenceBuilder(data=data_train, target=y_train,
                                      batch_size=ARGS.batch_size, ARGS=ARGS)
    model.fit_generator(generator=train_generator, epochs=ARGS.epochs,
                        max_queue_size=15, use_multiprocessing=False,
                        callbacks=[checkpoint, log], verbose=1, workers=ARGS.workers, initial_epoch=0)


# =============================================================================
# Updated main code to use RETAIN with MIMIC-III prescriptions
# =============================================================================


def main(preprocessed_mimic_filepath, output_directory):

    # =========================================================================
    # Hyperparameters
    # =========================================================================

    # dataset filtering - these parameters MUST be identical across all scripts
    use_truncated_codes = True
    proportion_event_instances = 0.9  # {0.5, 0.8, 0.9, 0.95, 0.99}
    admissions_per_patient_incl_min = 1
    medications_per_patient_incl_min = 50  # patients with less will be excluded entirely
    medications_per_patient_incl_max = 100  # patients with more or equal will have early medications truncated

    # data processing
    train_val_test_splits = (0.8, 0.1, 0.1)  # MUST be identical across all scripts

    # model architecture and training
    embed_size = 256
    lstm_size = 256
    dropout_rate_input = 0.0  # should be zero for this dataset
    dropout_rate_context = 0.0
    l2_factor = 0.001
    batch_size = 128
    n_epochs = 40

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
    assert abs(1 - sum(train_val_test_splits)) < 0.00001

    # =========================================================================
    # Data loading and standardized filtering
    # =========================================================================

    print("[INFO] Loading and preparing data")

    # load medication data
    mimic_df = pd.read_csv(preprocessed_mimic_filepath)
    mimic_df = mimic_df[mimic_df["event_type"] == "M"]

    # keep only most common medications
    mimic_df = mimic_df[mimic_df[event_code_count_col_name] >= event_code_count_incl_min]

    # keep only patients with enough admissions
    mimic_df = mimic_df[mimic_df["patient_admission_count"] >= admissions_per_patient_incl_min]

    # keep only patients with enough medications
    mimic_df = mimic_df[mimic_df["patient_medications_count"] >= medications_per_patient_incl_min]

    # truncate earliest medications for patients with too many medications
    truncated_patient_dfs = []
    for patient_id, patient_df in mimic_df.groupby("patient_id", sort=False):
        truncated_patient_dfs.append(patient_df.iloc[-1 * medications_per_patient_incl_max:])
    mimic_df = pd.concat(truncated_patient_dfs, axis=0)

    # =========================================================================
    # Further script-specific data preparation
    # =========================================================================

    # map string NDC codes to integer indices
    # padding is represented by the categorical index n_codes (the last index)
    # note that the mapping between meaning and categorical index for RETAIN
    # differs from that used by the LSTM, so LSTM predictions should be first
    # translated to NDC codes before being input into RETAIN
    code_idx_to_str_map = ["OUTCOME_SURVIVAL", "OUTCOME_MORTALITY"]
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
        
    # determine average and stdev number of admissions/medications for surviving/mortality patients
    # it is desirable for the distributions to be similar for the two groups
    # otherwise RETAIN can overfit on the length of input sequence
    admission_counts_survival = []
    admission_counts_mortality = []
    medication_counts_survival = []
    medication_counts_mortality = []
    for patient_admissions, patient_mortality in zip(patients_nested, patient_mortalities):
        if patient_mortality:
            admission_counts_mortality.append(len(patient_admissions))
            medication_counts_mortality.append(sum([len(admission_events) for admission_events in patient_admissions]))
        else:
            admission_counts_survival.append(len(patient_admissions))
            medication_counts_survival.append(sum([len(admission_events) for admission_events in patient_admissions]))

    print(f"[INFO] Number of different medical codes: {n_codes}")
    print(f"[INFO] Number of patients: {len(patients_nested)}")
    print(f"[INFO] Average patient mortality rate: {np.mean(patient_mortalities)}")
    print(f"[INFO] Number of admissions for surviving patients: "
          f"mean {np.around(np.mean(admission_counts_survival), 2)}, "
          f"std {np.around(np.std(admission_counts_survival), 2)}")
    print(f"[INFO] Number of admissions for mortality patients: "
          f"mean {np.around(np.mean(admission_counts_mortality), 2)}, "
          f"std {np.around(np.std(admission_counts_mortality), 2)}")
    print(f"[INFO] Number of medications for surviving patients: "
          f"mean {np.around(np.mean(medication_counts_survival), 2)}, "
          f"std {np.around(np.std(medication_counts_survival), 2)}")
    print(f"[INFO] Number of medications for mortality patients: "
          f"mean {np.around(np.mean(medication_counts_mortality), 2)}, "
          f"std {np.around(np.std(medication_counts_mortality), 2)}")
    
    # split train/val/test
    patients_nested_np_ragged = np.array(patients_nested)
    patient_mortalities_np = np.array(patient_mortalities)
    n_train = int(len(patients_nested) * train_val_test_splits[0])
    n_val = int(len(patients_nested) * train_val_test_splits[1])
    n_test = int(len(patients_nested) - (n_train + n_val))
    rng = np.random.default_rng(seed=12345)
    train_val_test_indexer = np.array([0] * n_train + [1] * n_val + [2] * n_test)
    rng.shuffle(train_val_test_indexer)  # in-place
    patients_nested_train_np_ragged = patients_nested_np_ragged[train_val_test_indexer == 0]
    patient_mortalities_train_np = patient_mortalities_np[train_val_test_indexer == 0]
    patients_nested_val_np_ragged = patients_nested_np_ragged[train_val_test_indexer == 1]
    patient_mortalities_val_np = patient_mortalities_np[train_val_test_indexer == 1]
    # patients_nested_test_np_ragged = patients_nested_np_ragged[train_val_test_indexer == 2]
    # patient_mortalities_test_np = patient_mortalities_np[train_val_test_indexer == 2]

    # finagle data so that it works with the RETAIN code
    data_train = [patients_nested_train_np_ragged]
    y_train = patient_mortalities_train_np.astype(np.int64)
    data_val = [patients_nested_val_np_ragged]
    y_val = patient_mortalities_val_np.astype(np.int64)

    # delete old log file
    try:
        os.remove(os.path.join(output_directory, "log.txt"))
    except FileNotFoundError:
        pass

    # create argument namespace for RETAIN code
    ARGS = argparse.Namespace
    ARGS.num_codes = n_codes
    ARGS.numeric_size = 0  # don't use any numeric features
    ARGS.use_time = False  # don't use time as a feature
    ARGS.emb_size = embed_size
    ARGS.epochs = n_epochs
    ARGS.workers = 4
    ARGS.n_steps = 300  # truncate patients after 300 admissions (not applicable since no patients have that many)
    ARGS.recurrent_size = lstm_size
    ARGS.batch_size = batch_size
    ARGS.dropout_input = dropout_rate_input
    ARGS.dropout_context = dropout_rate_context
    ARGS.l2 = l2_factor
    ARGS.directory = output_directory
    ARGS.allow_negative = False  # Exception thrown if this is True for some reason, not sure why

    # train RETAIN
    print('Creating Model')
    model = model_create(ARGS)
    print('Training Model')
    train_model(
        model=model, data_train=data_train, y_train=y_train, data_test=data_val, y_test=y_val, ARGS=ARGS)

    # find and report epoch with best PR-AUC
    best_epoch = 0
    best_pr_auc = 0.0
    with open(os.path.join(output_directory, "log.txt"), "r") as readfile:
        readfile.readline()
        for epoch, line in enumerate(readfile.readlines()):
            splits = line.split("PR-AUC: ")
            pr_auc = float(splits[-1])
            if pr_auc > best_pr_auc:
                best_epoch = epoch
                best_pr_auc = pr_auc
    print(f"[INFO] Epoch number {best_epoch} had the highest PR-AUC of {best_pr_auc}")


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--preprocessed_mimic_filepath', type=str, required=True)
    parser.add_argument('--output_directory', type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args.preprocessed_mimic_filepath, args.output_directory)


"""
GRID SEARCH FOR RETAIN PARAMETERS:

embed_size, lstm_size, dropout_rate_context, l2_factor : best validation PR-AUC

512, 512, 0.0, 0.001 : 0.796986
512, 512, 0.2, 0.001 :
512, 512, 0.4, 0.001 :

512, 256, 0.0, 0.001 : 0.795547
512, 256, 0.2, 0.001 : 
512, 256, 0.4, 0.001 : 

256, 512, 0.0, 0.001 : 0.791528
256, 512, 0.2, 0.001 : 
256, 512, 0.4, 0.001 : 

256, 256, 0.0, 0.001 : 0.799791
256, 256, 0.2, 0.001 : 0.798606
256, 256, 0.4, 0.001 : 0.792274

256, 128, 0.0, 0.001 : 0.796881
256, 128, 0.2, 0.001 : 0.794085
256, 128, 0.4, 0.001 : 0.793953

128, 256, 0.0, 0.001 : 0.797035
128, 256, 0.2, 0.001 : 0.795079
128, 256, 0.4, 0.001 : 0.794217

128, 128, 0.0, 0.001 : 0.796271
128, 128, 0.2, 0.001 : 
128, 128, 0.4, 0.001 : 
"""
