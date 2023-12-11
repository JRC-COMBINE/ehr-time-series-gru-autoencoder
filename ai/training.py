# Logging
import functools
import logging
import os
import random
from datetime import datetime

# Utility
import copy
import string
from pathlib import Path
import json
from typing import List
import humanfriendly as hf

# Math and data
import numpy as np
from sklearn.preprocessing import StandardScaler

# Weights and Biases (wandb)
import wandb
from wandb.keras import WandbCallback

# NN
import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback, EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
from tensorflow.keras.losses import get as get_keras_loss
from tqdm import tqdm

# Baseline model
from sklearn.decomposition import PCA

from ai.model import DynModel, bottleneck_size_default
from evaluation import plot
from common import io
from data.preprocessor import static_cols_prefix


# Return keras.utils.Sequence that contains the batches
class DynSequence(Sequence):
    def __init__(self, admission_indices, dyn_charts, batch_size, batching_function, shuffle=True):
        self.adm_indices = admission_indices
        self.dyn_charts = dyn_charts
        self.batch_size = batch_size
        self.batching_fun = batching_function

        # Randomize order of admissions
        np.random.shuffle(self.adm_indices)

        # Order of batches within one epoch is given through this indexing array
        self.batch_ordering = np.arange(0, len(self))

        # Cache batches to speed up training at the expense of memory - only possible if not shuffling admissions
        self.shuffle = shuffle
        self.cache = {}

    def __getitem__(self, index):
        # Find out batch index w.r.t. current batch ordering
        batch_index = self.batch_ordering[index]

        if batch_index not in self.cache:
            # Determine slice of admission list that corresponds to this batch index
            batch_adm_indices = self.adm_indices[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

            # Form a batch out of the admissions
            batch_input = self.batching_fun(
                dyn_charts=self.dyn_charts,
                batch_adm_idxs=batch_adm_indices
            )

            if self.shuffle:
                return self._batch_from_input(batch_input)

            self.cache[batch_index] = batch_input

        batch_input = self.cache[batch_index]
        return self._batch_from_input(batch_input)

    @staticmethod
    def _batch_from_input(batch_input):
        return batch_input, batch_input

    def __len__(self):
        return int(np.ceil(len(self.adm_indices) / self.batch_size))

    def on_epoch_end(self):
        """
        Method called (by keras) at the end of every epoch.

        Shuffle batches so that their order within an epoch is changed. This has been observed to lead to faster
         convergence. ("Practical Recommendations for Gradient-Based Training of Deep Architectures", Bengio, 2012).
        """

        # Shuffle batch order
        np.random.shuffle(self.batch_ordering)

        # If shuffling admissions, also shuffle their order
        if self.shuffle:
            np.random.shuffle(self.adm_indices)


class SaveModelCallback(Callback):
    def __init__(self, model_saving_function, monitored_metric='val_loss'):
        super().__init__()
        self.model_saving_function = model_saving_function

        # Monitor metric to only save models that are better than what is already saved
        self.monitored_metric = monitored_metric
        self.smallest_yet = None
        self.epochs_without_improvement = 0

        # Save if metric was ever not valid
        self.metric_ever_invalid = False

    def on_epoch_end(self, epoch, logs=None):
        # Only save model if it's better
        metric = logs[self.monitored_metric]
        metric_valid = np.isfinite(metric)
        if metric_valid and (self.smallest_yet is None or metric < self.smallest_yet):
            self.model_saving_function(epoch=epoch)
            logging.info(f"Saved model! {self.monitored_metric} of {metric} is smaller than best"
                         f" yet ({self.smallest_yet})")
            self.smallest_yet = metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if not metric_valid:
                self.metric_ever_invalid = True
                logging.info(f"Not saving model - metric is invalid: {self.monitored_metric} = {metric}")
            else:
                logging.info(f"Not saving model - {self.monitored_metric} of {metric} is not smaller than best yet"
                             f" ({self.smallest_yet})")
            logging.info(f"{self.epochs_without_improvement} epochs without improvement.")


class EvalAfterEachBatchCallback(Callback):
    def __init__(self, training_data):
        super().__init__()

        # Training data: Used for evaluating batches in inference mode
        self.training_data = training_data

        # Collect lists of losses (after each batch)
        self.losses_training = []
        self.losses_eval_mode = []

    def on_train_batch_end(self, batch, logs=None):
        # Remember train loss after this batch
        self.losses_training.append(logs['loss'])

        # Eval the same batch using the current state of the model in eval mode
        batch_x, batch_y = self.training_data[batch]  # Retrieve batch
        eval_mode_loss = self.model.evaluate(
            x=batch_x,
            y=batch_y,
            verbose=0  # Don't show progress bar
        )
        self.losses_eval_mode.append(eval_mode_loss)


class TrainLossCallback(Callback):
    def __init__(self):
        super().__init__()

        # Losses: List of lists (one list per epoch)
        self.losses = [[]]
        self.epoch_loss_means = []

    def on_batch_end(self, batch, logs=None):
        batch_loss = logs.get('loss')
        self.losses[-1].append(batch_loss)

    def on_epoch_end(self, epoch, logs=None):
        epoch_losses = self.losses[-1]
        epoch_losses_mean = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1} mean train loss: {epoch_losses_mean}")

        # Save loss of this epoch
        self.epoch_loss_means.append(epoch_losses_mean)

        # Start new list for next epoch
        self.losses.append([])


class Training:
    """
    Training pipeline for neural network model.
    """

    def __init__(self, iom, prep, max_epochs=24, batch_size=4, loss='huber_loss', optimizer='Adam',
                 learning_rate=0.00075, max_batch_len=0, early_stopping_patience=8, clip_grad_norm=True, baseline=False,
                 shuffle_admissions=False, no_training=False, additional_model_args=None):

        # Enable mixed precision if training on GPU
        if len(tf.config.list_physical_devices('GPU')) > 0:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        # Store args for wandb
        self.pipeline_args = None

        # Baseline mode: If running in baseline mode, we don't run the NN model training but the baseline model training
        self.baseline_mode = baseline
        if self.baseline_mode:
            logging.info("Running in BASELINE mode.")
        else:
            logging.info("Running with NN model, not in baseline mode.")
        self.model_kind_name = {
            True: "baseline (PCA)",
            False: "dynamic (NN)"
        }[self.baseline_mode]
        self.baseline_eval = {}
        self._baseline_features = None
        self._baseline_model = None
        self._baseline_input_data_matrix = None  # Rows are indexed w.r.t. ordering of 'all' split
        # (i.e. first row = data for first admission in 'all' split); data is scaled.
        self._baseline_input_data_matrix_cols = None
        self._baseline_input_data_matrix_unscaled = None  # contains a version of the matrix where the scaling has been
        # reversed

        # Admission filters: Used for experimenting on subsets of the total population
        self.admission_filters = ["bed_and_breakfast"]  # this list only shows the *possible* admission filter. If
        # any admission filter is active, it will be the value of the split variable used in other modules (like
        # clustering)

        # Model paths
        self.iom = iom
        self._dyn_model_path = Path(iom.get_models_dir(), "dyn_model.h5")
        self._dyn_model_enc_path = Path(iom.get_models_dir(), "dyn_model_enc.h5")

        # Model instances
        self._dyn_model = None
        self._dyn_model_enc = None
        self._dyn_model_architecture = None

        # Preprocessor: Needed to get training data
        self.prep = prep
        self.prep.trainer = self

        # Model arguments
        if additional_model_args is None:
            additional_model_args = {}
        self.additional_model_args = additional_model_args

        # Training settings
        self.optimizer = optimizer
        self.shuffle_admissions = io.str_to_bool(shuffle_admissions)  # if True, admissions are shuffled after every
        # epoch during training
        self.lr = float(learning_rate)  # Applied to optimizer after model compilation
        self.loss = loss
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.max_batch_len = int(max_batch_len)  # if max batch length is set to 0, don't cut batches at all
        self.early_stopping_patience = int(early_stopping_patience)
        if self.max_epochs <= self.early_stopping_patience:
            logging.warning(f"Early stopping patience must be smaller than epochs per run"
                            f" (otherwise, early stopping would never be triggered).")
            self.max_epochs = int(1.25 * self.early_stopping_patience) + 1
            logging.warning(f"Increasing max epochs to {self.max_epochs}")
        self.use_positional_encoding = self.prep.use_positional_encoding
        self.positional_encoding_dims = self.prep.positional_encoding_dims
        self.clip_grad_norm = io.str_to_bool(clip_grad_norm)

        # If this flag is set, NO TRAINING will take place. Instead, untrained model will be evaluated on the
        # validation data. This process is useful for finding model architectures which are biased towards good
        # performance.
        self.no_training = io.str_to_bool(no_training)
        if self.no_training:
            logging.info("NOT TRAINING model - only testing val loss")

        # Training state
        self._training_meta_file_path = Path(iom.get_models_dir(), "training.json")
        self.training_state = {
            'epoch': 0,
            'done': False,
            'stats': None
        }

        # Feature file (which saves the features for admissions so that they only have to be calculated once)
        self._features_file_path = Path(iom.get_models_dir(), "features.pkl")
        self._features_loaded = {}
        self._new_features_computed = False

        # Split data into training and validation set
        val_split = 0.1
        unscaled_exams, _ = self.prep.get_supported_dyn_data_list()
        num_data = len(unscaled_exams)
        data_idxs = np.arange(num_data)

        # Shuffle the indices in a repeatable way, then restore randomness by setting a new, random seed
        new_random_seed = np.random.randint(1, 2**14)
        rng = np.random.default_rng(1)  # same, fixed seed every time
        rng.shuffle(data_idxs)
        rng = np.random.default_rng(new_random_seed)

        self._train_data_idxs = data_idxs[:np.floor((1 - val_split) * num_data).astype(int)]
        self._val_data_idxs = data_idxs[-(num_data - len(self._train_data_idxs)):]
        logging.info(f"Of {num_data} unscaled dyns, {len(self._train_data_idxs)} are train "
                     f"and {len(self._val_data_idxs)} are val")

        # Save training indices in prep - it needs it for fitting scaler for static and dynamic data
        self.prep.train_data_idxs = self._train_data_idxs

        # Extract static data
        if not self.prep.ignore_static_data:
            static_arr, static_categorical = self.prep.get_scaled_static_data_array()

        # Remove admissions with zero dynamic entries - this can sometimes happen when the raw data contained only
        # NaN values
        charts = self.prep.get_dyn_charts()
        logging.info(f"Received {len(charts)} dynamic charts.")
        self._train_data_idxs = [adm_idx for adm_idx in self._train_data_idxs if len(charts[adm_idx]) > 0]
        self._val_data_idxs = [adm_idx for adm_idx in self._val_data_idxs if len(charts[adm_idx]) > 0]
        logging.info(f"After removing zero-length dynamic charts, {len(self._train_data_idxs)} train charts remain.")
        logging.info(f"After removing zero-length dynamic charts, {len(self._val_data_idxs)} val charts remain.")

        # Save updated training indices in prep
        self.prep.train_data_idxs = self._train_data_idxs

        # Sanity check: Make sure that number (and thus, indices) of dynamic charts matches static data array size
        if not self.prep.ignore_static_data:
            assert len(static_arr) == len(charts),\
                f"Static array has {len(static_arr)} entries but there are {len(charts)} dyn charts!"
            assert len(static_categorical) == len(charts),\
                f"Static categorical has {len(static_categorical)} entries but there are {len(charts)} dyn charts!"

        # Masking: To put admissions with differing lengths into one batch, we need to mask the shorter admissions.
        # We do this using a value that does not occur in the data.
        self.masking_value = self.prep.masking_value
        # Training statistics
        self.training_start_time_utc = datetime.utcnow()
        self.training_end_time_utc = None
        self.losses_train = []
        self.losses_val = []
        self.trained_epochs = []

        # Diagnostics: Record training loss after each batch (for plotting)
        self.eval_after_every_batch = False
        self.eval_after_batch_callback = None

        # Features
        self._features_cache = {}

    @functools.lru_cache(maxsize=32)
    def get_split_indices(self, split):
        if split == io.split_name_val:
            return self._val_data_idxs
        elif split == io.split_name_train:
            return self._train_data_idxs
        elif split == io.split_name_all or split == io.split_name_unit_tests:
            return self._train_data_idxs + self._val_data_idxs
        elif split in self.admission_filters:
            return self._evaluate_admission_filter(split)
        else:
            assert False, f"Split '{split}' unknown!"

    def _evaluate_admission_filter(self, filter_name):
        """
        Evaluates (and caches) the admission filter with the specified name. This can be used to e.g. select admissions
        with a specific disease or other characteristics.
        :param filter_name:
        :return: list of admission indices
        """

        # Start with all the valid admissions
        adm_indices = self.get_split_indices(io.split_name_all)

        logging.info(f"Evaluating admission filter '{filter_name}' ...")

        # Filter based on criteria
        if filter_name == "bed_and_breakfast":
            # Filter to admissions where patients have spent at most 24 hours in ICU
            static_info = self.prep.extract_static_medical_data(
                adm_indices=adm_indices
            )
            adm_indices = [idx for (idx, days) in zip(adm_indices, static_info['FUTURE_days_in_care']) if days <= 1]
            # less or equal to a full day in ICU
        else:
            assert False, f"Admission filter with filter_name '{filter_name}' not known!"

        return adm_indices

    def sample_from_split(self, split, sample_num):
        split_indices = self.get_split_indices(split)
        if sample_num >= len(split_indices):
            return np.array(split_indices)
        else:
            return np.random.choice(split_indices, size=sample_num, replace=False)

    def find_train_indices_in_all(self) -> List[int]:
        """
        Find the training indices in the full index list. This is required when using the 'all' list to build a matrix
         (as is done for the baseline method) and then wanting to extract the training indices from that.
        :return: list of indices of training admissions w.r.t. the 'all' split
        """
        return self._find_split_indices_in_all(io.split_name_train)

    def _find_split_indices_in_all(self, split) -> List[int]:
        split_indices_all = self.get_split_indices(io.split_name_all)
        split_indices = self.get_split_indices(split)
        split_indices_wrt_all = [split_indices_all.index(idx) for idx in split_indices]
        return split_indices_wrt_all

    def get_trained_dyn_model(self):
        if self._dyn_model is None:
            # Model must be loaded or trained
            self.train_or_load()
        return self._dyn_model

    def get_trained_dyn_enc_model(self):
        if self._dyn_model_enc is None:
            # Model must be loaded or trained
            self.train_or_load()
        return self._dyn_model_enc

    def train_or_load(self):
        # Try to load trained models from disk
        models_loaded = self.load_model_checkpoints()

        if not models_loaded or not self.training_state['done']:
            # Models need to be trained
            self._train()

        # Check if loaded dynamic data columns match with the ones that are suitable for the model
        if not self.baseline_mode:
            self.ascertain_dyn_col_integrity()

    def ascertain_dyn_col_integrity(self):
        actually_loaded_dyn_cols = self.prep.dyn_data_columns
        loaded_dyn_data_info = io.read_json(self.iom.get_dyn_data_cols_path())
        assert loaded_dyn_data_info is not None, f"Dynamic column info could not be loaded from" \
                                                 f" '{self.iom.get_dyn_data_cols_path()}'!"
        model_trained_dyn_cols = loaded_dyn_data_info['cols']

        # Ignore static entries
        actually_loaded_dyn_cols = [c for c in actually_loaded_dyn_cols if not c.startswith(static_cols_prefix)]
        model_trained_dyn_cols = [c for c in model_trained_dyn_cols if not c.startswith(static_cols_prefix)]

        assert actually_loaded_dyn_cols == model_trained_dyn_cols, f"The loaded dynamic data columns \n" \
                                                                   f"{actually_loaded_dyn_cols} do not match with" \
                                                                   f" the columns that the model was trained on: \n" \
                                                                   f"{model_trained_dyn_cols}!"
        logging.info("Success: Loaded dynamic data columns match with those that the loaded model was trained with.")

    def get_model_architecture(self):
        if self._dyn_model_architecture is None:
            self._dyn_model_architecture = DynModel(
                prep=self.prep,
                trainer=self,
                **self.additional_model_args
            )
        return self._dyn_model_architecture

    def sequence_model_data(self, split_indices) -> Sequence:
        """
        Generates Sequence for training (and validation) data to be consumed by the model.

        All admissions that are part of the same batch are extended in the temporal dimension so that
        they all have the same length. Different batches can have different temporal length.

        :param split_indices: Admission indices (w.r.t. the global admission list, not w.r.t to the database)

        :return: Sequence with the batches as elements
        """
    
        logging.info(f"Building batch sequence for {len(split_indices)} admissions ...")

        # Instantiate sequence
        dyn_seq = DynSequence(
            dyn_charts=self.prep.get_dyn_charts(),  # All dynamic data (train but
            # also val).
            admission_indices=list(split_indices),  # these indices tell us which data to use
            batch_size=self.batch_size,
            batching_function=self.prepare_admissions_for_batch,
            shuffle=self.shuffle_admissions
        )

        return dyn_seq

    def prepare_admissions_for_batch(self, dyn_charts, batch_adm_idxs):
        """
        Prepares batches of training (or validation) data by bringing different admissions to the same temporal length.

        :param dyn_charts:
        :param batch_adm_idxs:
        :return:
        """

        # Retrieve dynamic charts for the batch admissions
        batch_dyn_charts = [dyn_charts[adm_idx] for adm_idx in batch_adm_idxs]

        # Cut a random temporal chunk out of each admission - this can greatly speed up training since extremely long
        # sequences lead to extremely long gradient paths in RNNs
        # (Note: Batches are only ever cut when training, not when performing reconstruction or computing features
        if self.max_batch_len != 0:
            seq_max_steps = min(max([len(df) for df in batch_dyn_charts]), self.max_batch_len)
            chunks_start_indices = [random.randint(0, max(0, len(df) - seq_max_steps)) for df in batch_dyn_charts]
            batch_dyn_charts = [df[idx:idx + seq_max_steps] for idx, df in zip(chunks_start_indices, batch_dyn_charts)]

        # The admissions may have different temporal lengths. We want to put them together in one batch, so we have to
        # pad all admissions that are shorter than the maximal length in the batch.
        batch_max_len = max([len(charts) for charts in batch_dyn_charts])

        # Produce batch data for each selected admission
        batch = []
        for inside_batch_idx, adm_idx in enumerate(batch_adm_idxs):
            # Convert to matrix representation suitable for NN model
            dyn_arr = batch_dyn_charts[inside_batch_idx].to_numpy()
            dyn_steps, dyn_features = dyn_arr.shape

            # Padding data: It must "fill up" the data in the temporal dimension
            padding_fill = batch_max_len - dyn_steps
            if padding_fill > 0:
                padding_data = np.zeros(shape=(padding_fill, dyn_features)) + self.masking_value
                dyn_arr = np.concatenate(
                    [dyn_arr, padding_data],
                    axis=0  # Temporal dimension
                )

            batch.append(dyn_arr)

        # Stack the admissions into a batch
        batch = np.stack(batch)
        # shape (batch_size, steps, features)
        
        
        
        # Output is the same as input
        return batch

    def _prepare_input_data(self, adm_idx):
        # Get dynamic training data
        dyn_chart = self.prep.get_dyn_charts()[adm_idx]

        # Convert to numpy and add a batch dimension
        dyn_arr = dyn_chart.to_numpy()
        dyn_arr = np.expand_dims(dyn_arr, axis=0)

        return dyn_arr

    def _train(self):
        # Train dynamic model (which uses features from list-like models)
        logging.info(f"Training {self.model_kind_name} model...")
        if not self.baseline_mode:
            self._train_dyn_model()
        else:
            self._train_baseline_model()
        logging.info(f"Training {self.model_kind_name} model done!")

        if not self.baseline_mode:
            # Plot fitting progress
            self.plot_fit_progress_after_training()

        # Remove callback for plotting model fit
        self.eval_after_batch_callback = None

    def write_dyn_data_col_file(self):
        # Check if file already exists (only continue writing if it doesn't)
        loaded_dyn_data_info = io.read_json(self.iom.get_dyn_data_cols_path())
        if loaded_dyn_data_info is not None:
            return

        # Compose file
        cols = self.prep.dyn_data_columns
        dyn_data_col_info = {
            'date': datetime.utcnow().isoformat(),
            'cols': cols,
            'col_labels': [self.prep.label_for_any_item_id(c) for c in cols],
            'num_cols': len(cols),
            'num_adms': len(self._train_data_idxs) + len(self._val_data_idxs)
        }

        # Save as JSON in model checkpoint directory
        io.write_json(dyn_data_col_info, self.iom.get_dyn_data_cols_path(), pretty=True)

    def plot_fit_progress_after_training(self):
        plotter = plot.Plotting(
            iom=self.iom,
            preprocessor=self.prep,
            trainer=self,
            evaluator=None,
            clustering=None,
            split=None
        )
        plotter.plot_fit_progress()

    def _train_baseline_model(self):
        # Check if training already performed
        if len(self._features_cache) > 0:
            return

        # Prepare training data for baseline (a big matrix with the static data and aggregated forms of the dynamic
        # data)
        data_matrix, col_names = self.prep.generate_baseline_data_matrix()
        # (shape: (n_samples, n_data_features))
        self._baseline_input_data_matrix = data_matrix
        self._baseline_input_data_matrix_cols = col_names

        # Also save an unscaled version of the data input matrix
        self._baseline_input_data_matrix_unscaled = self.prep.baseline_scaler.inverse_transform(data_matrix)

        # Train baseline method (PCA) on training admissions
        matrix_train_indices = self.find_train_indices_in_all()  # matrix was created using 'all' split, not original
        # admission order!
        matrix_train = data_matrix[matrix_train_indices]

        # Fit PCA
        logging.info(f"Fitting PCA for baseline on  training matrix "
                     f"(shape (n_samples, n_data_features) = {matrix_train.shape}) ...")
        pca = PCA(
            n_components=bottleneck_size_default,  # Have the baseline use the same number of features as the NN model
            svd_solver='full'
        )
        pca.fit(matrix_train)
        self._baseline_model = pca
        logging.info(f"Fitting PCA for baseline done!")

        self._new_features_computed = True  # Remember that new features were computed

        # Apply to full matrix
        features = pca.transform(data_matrix)

        # Save features
        self._baseline_features = features

        # Save fit info about PCA
        train_score = pca.score(matrix_train)
        val_score = pca.score(data_matrix[self._find_split_indices_in_all(io.split_name_val)])
        ln_to_log10 = np.log(10) / np.log(np.e)  # == np.log(10) / 1 => convert from base e to base 10 logarithm
        self.baseline_eval = {
            'explained_variance_cumsum': list(pca.explained_variance_ratio_.cumsum()),
            'explained_variance_sum': np.sum(pca.explained_variance_ratio_),
            'features_in': int(data_matrix.shape[1]),
            'features_out': int(pca.n_components_),
            'loglikelihood_train': float(train_score),
            'loglikelihood_val': float(val_score),
            'log10likelihood_train': float(train_score / ln_to_log10),
            'log10likelihood_val': float(val_score / ln_to_log10)
        }

        # Mark training as done
        self.training_state['done'] = True

    def _train_dyn_model(self):
        # Create early stopping callback
        loss_monitored = 'val_loss'
        early_stopping_kwargs = {}
        loss_baselines = {  # Early stopping patience is NOT renewed (when it runs out) if loss is above this
            # baseline. This serves as a way to terminate unpromising trainings early.
            'huber_loss': 4.0
        }
        if self.loss in loss_baselines:
            early_stopping_kwargs['baseline'] = loss_baselines[self.loss]
        early_stopping = EarlyStopping(
            monitor=loss_monitored,
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
            **early_stopping_kwargs
        )
        callbacks = [early_stopping]
        logging.info(f"Early stopping with patience {self.early_stopping_patience}")

        # Add callback that reduces the learning rate if learning plateaus
        reduce_lr_patience = max(self.early_stopping_patience // 2, 10)
        reduce_lr = ReduceLROnPlateau(
            monitor=loss_monitored,
            factor=0.5,
            patience=reduce_lr_patience,
            verbose=1
        )
        callbacks.append(reduce_lr)
        logging.info(f"Learning rate will be reduced after {loss_monitored} plateaus for {reduce_lr_patience} epochs.")

        # Add callback that stops training if NaN loss is encountered
        nan_stop_callback = TerminateOnNaN()
        callbacks.append(nan_stop_callback)

        # Build model architecture
        if self._dyn_model is None:
            self._dyn_model = self.get_model_architecture().get_dyn_model()
        if self._dyn_model_enc is None:
            self._dyn_model_enc = self.get_model_architecture().get_dyn_model_enc()

        # Compile model
        self._dyn_model.compile(optimizer=self.optimizer, loss=self.loss)

        # Apply learning rate
        keras_backend.set_value(self._dyn_model.optimizer.learning_rate, self.lr)

        # Clip gradient norm
        if self.clip_grad_norm:
            self._dyn_model.optimizer.clipnorm = 1.0

        # Print and save summary
        dyn_summary = self.get_summary(self._dyn_model)
        logging.info(dyn_summary)
        self.training_state['model_summary'] = dyn_summary

        logging.info(f"Using optimizer {self.optimizer} with LR {self.lr} and loss {self.loss}")

        # Create callback for saving model after every epoch
        save_model_callback = SaveModelCallback(
            self.save_model_checkpoints,
            monitored_metric=loss_monitored
        )
        callbacks.append(save_model_callback)

        # Callback for saving the training loss of every batch (by default, keras displays and saves an epoch's last
        # batch loss as the loss as that epoch's loss
        train_loss_callback = TrainLossCallback()
        callbacks.append(train_loss_callback)

        # Prepare data sequences
        train_data_seq = self.sequence_model_data(
            split_indices=self._train_data_idxs
        )
        
        val_data_seq = self.sequence_model_data(
            split_indices=self._val_data_idxs
        )
        
        # Find out initial validation loss (with model in an untrained state)
        if self.training_state['epoch'] == 0:
            logging.info("Finding out validation loss before training (using the model's random initialization) ...")
            self.training_state['val_loss_before_training'] = self._dyn_model.evaluate(val_data_seq, verbose=0)
            logging.info(f"Validation loss before training: {self.training_state['val_loss_before_training']}")

            # Exit if we are not training
            if self.no_training:
                self.save_model_checkpoints(epoch=0)
                return

        # Create callback for evaluating model performance after every single batch
        # (only used for diagnosing problems)
        if self.eval_after_every_batch:
            self.eval_after_batch_callback = EvalAfterEachBatchCallback(training_data=train_data_seq)
            callbacks.append(self.eval_after_batch_callback)

        # Write dynamic data columns file: This file stores the columns of dynamic data that were used for training
        # this model. It's important that future runs using the same model also use the same name (and index) of
        # each column of dynamic data (this could e.g. be a problem when using a model with a different number of
        # admissions than it was trained with).
        self.write_dyn_data_col_file()

        # Wandb callback
        def clean_up_val(v):
            if type(v) in [float, int, bool]:
                return v
            else:
                # None
                if v is None:
                    return v

                # Bool
                if v.lower() in ["true", "false"]:
                    return io.str_to_bool(v)

                # Int
                try:
                    v_int = int(v)
                    return v_int
                except ValueError:
                    pass

                # Float
                try:
                    v_float = float(v)
                    return v_float
                except ValueError:
                    pass

                return v

        random_id = "_".join(["".join([random.choice(string.ascii_uppercase) for _ in range(3)]) for _ in range(2)])
        wandb_dir = os.path.join(os.environ.get("HOME"), "wandb")
        io.makedirs(wandb_dir)
        wandb.init(
            dir=wandb_dir,
            project="dl-clus",
            config={k: clean_up_val(val) for (k, val) in self.pipeline_args.items()},
            name=self.pipeline_args['id'] + "_" + random_id,
            mode="disabled"
        )
        callbacks.append(
            WandbCallback(
                save_model=False
            )
        )

        # Fit the model
        time_pre_fit = datetime.now()
        train_history = self._dyn_model.fit(
            x=train_data_seq,
            # y as an argument is not necessary since target values are obtained from generator in x
            validation_data=val_data_seq,
            initial_epoch=self.training_state['epoch'],
            epochs=self.training_state['epoch'] + self.max_epochs,
            callbacks=callbacks
        )
        seconds_taken_actual = (datetime.now() - time_pre_fit).total_seconds()

        # Record losses to history
        self.losses_train += train_loss_callback.epoch_loss_means
        self.losses_val += train_history.history['val_loss']
        self.trained_epochs += [e + 1 for e in train_history.epoch]

        # Determine if NaN loss was triggered
        num_admissions = len(self.prep.encounter_ids)
        seconds_taken_projected_min = 0.05 * num_admissions
        nan_stop_triggered = seconds_taken_actual < seconds_taken_projected_min \
                             or save_model_callback.metric_ever_invalid

        # Determine is training is done
        early_stopping_triggered = early_stopping.stopped_epoch != 0
        training_done = self.training_state['epoch'] >= self.max_epochs \
                        or early_stopping_triggered \
                        or nan_stop_triggered
        self.training_state['done'] = training_done

        if self.training_state['done']:
            logging.info("Training is done!")

            # Record that training has finished
            self.training_end_time_utc = datetime.utcnow()

            # If NaN loss happened, reload models from disk
            if nan_stop_triggered:
                logging.info("Since NaN loss was recorded, reload working model checkpoints from disk.")

                # Load model checkpoints, but keep current training state
                state = copy.deepcopy(self.training_state)
                self.load_model_checkpoints()
                self.training_state = state

            # Save model checkpoints a final time (necessary if stopping training using max_epochs)
            self.save_model_checkpoints(self.training_state['epoch'] - 1)

            # Finish run in wandb
            wandb.run.finish()
        else:
            logging.info("Training is NOT done.")

    @staticmethod
    def get_summary(model) -> str:
        summary_lines = []
        model.summary(print_fn=lambda line: summary_lines.append(line))
        summary = "\n".join(summary_lines)
        return summary

    def reconstruct_time_series(self, adm_idx, evaluate: bool = False):
        """
        Reconstruct time series
        :param adm_idx:
        :param evaluate: If True, return (reconstructed_values, reconstruction_loss (float))
        :return: reconstructed_values (np.ndarray)
        """

        # Reconstruction works differently for baseline model (PCA) vs. NN model
        if not self.baseline_mode:

            # Get a trained autoencoder model
            autoenc_model = self.get_trained_dyn_model()

            # Prepare pre-bottleneck inputs: This is what the model has as information for reconstruction
            x_input = self._prepare_input_data(
                adm_idx=adm_idx
            )

            # Reconstruct the time series
            reconstructed_values = autoenc_model.predict(
                x=x_input
            )
            # shape (batch, steps, features) = (1, len(dyn_chart), len(self.prep.dyn_data_columns))
            reconstructed_values = reconstructed_values[0, :, :]  # Remove batch dimension

            # Convert to float64 - this is what the ground truth values use
            reconstructed_values = reconstructed_values.astype(np.float64)

            if evaluate:
                # Evaluate: Get the loss incurred by the reconstruction of the time series
                x_output = x_input
                reconstruction_loss = autoenc_model.evaluate(
                    x=x_input,
                    y=x_output
                )  # loss is a scalar

        else:
            # Reconstruct using the baseline model (PCA)

            # Get features
            features = self._load_or_calculate_features(adm_idx=adm_idx)

            # Calculate the inverse of the PCA in order to get its best approximation of the original inputs
            scaled_input_data = self._baseline_model.inverse_transform(features)

            # Reverse PCA scaling: This maps PCA input data back to original values
            # Shape for inverse transform must be (n_samples, n_features)
            rec_input_data = self.prep.baseline_scaler.inverse_transform(np.expand_dims(scaled_input_data, axis=0))[0]

            # Prepare keras loss function
            if evaluate:
                keras_loss = get_keras_loss(self.loss)

            # The baseline was trained using a matrix whose rows correspond to admissions and whose columns represent
            # statistics of the different time series attributes, e.g. there are columns each for the mean, minimum
            # and maximum of a time series for an admission. In order to reconstruct the time series with the baseline
            # model, we have to find the column where the mean approximation for each attribute is stored.

            col_idx_cache = {}

            def col_idx_for_attr_idx(tr_self, item_id):
                if item_id not in col_idx_cache:
                    # Find out how the preprocessor calls this dynamic attribute
                    prep_col_name = tr_self.prep.key_for_item_id(item_id=item_id)

                    # Find out the index of the respective mean column in the PCA data input matrix
                    col_idx_cache[item_id] = [idx for (idx, cn)
                                              in enumerate(tr_self._baseline_input_data_matrix_cols)
                                              if prep_col_name in cn and "mean" in cn.lower()
                                              ][0]

                return col_idx_cache[item_id]

            # Reconstruct in the same shape as the NN model: (time_steps, features)
            # (The columns must follow the order of dyn columns in self.prep.dyn_data_columns)
            reconstructed_means = []
            reconstruction_losses = []
            for col_name in self.prep.dyn_data_columns:
                # Skip over meta columns
                if col_name in self.prep.meta_columns:
                    continue

                # Retrieve the best mean approximation that the PCA model can create
                col_idx = col_idx_for_attr_idx(self, col_name)
                mean_approx = rec_input_data[col_idx]

                # Apply canonical scaling to the value - this is counterintuitive, but necessary since the PCA baseline
                # model uses inputs that are normalized in a slightly different fashion from the NN model. Other
                # functions calling this function expect values to be normalized like the NN model.
                mean_approx = self.prep.perform_scaling(np.array([mean_approx], dtype=np.float64),
                                                        column_name=col_name).item()
                reconstructed_means.append(mean_approx)

                # If requested, also calculate loss
                if evaluate:

                    # Find out the original value (which served as the input for the PCA) and scale it
                    adm_pos_in_baseline_features = self.get_split_indices(io.split_name_all).index(adm_idx)
                    gt_val = self._baseline_input_data_matrix_unscaled[adm_pos_in_baseline_features, col_idx]  # Note
                    # that we are using the data matrix with scaling reversed
                    gt_val = self.prep.perform_scaling(np.array([gt_val], dtype=np.float64),
                                                       column_name=col_name).item()

                    # Calculate loss between ground truth value and scaled approximation
                    # (Note that the loss is calculated with scaled values, just like for the NN model. This makes
                    # comparisons between the NN model and the baseline easier.)
                    val_loss = keras_loss(np.array([gt_val]), np.array([mean_approx])).numpy()
                    reconstruction_losses.append(val_loss)

            # Average losses over time series (since that is what happens for the NN model as well)
            if evaluate:
                reconstruction_loss = np.mean(reconstruction_losses)

            # Convert to numpy array with shape (time_steps, features) by repeating the mean approximations in time
            num_time_steps = len(self.prep.get_dyn_charts()[adm_idx])
            reconstructed_values = np.concatenate([np.array([reconstructed_means])] * num_time_steps, axis=0)
            reconstructed_values = reconstructed_values.astype(np.float64)  # like ground truth

        # Finally, return values (regardless of model used for calculating them)
        if not evaluate:
            return reconstructed_values
        else:
            return reconstructed_values, float(reconstruction_loss)

    @staticmethod
    def _normalize_feature_dimensions(features: np.ndarray) -> np.ndarray:
        # Bring each dimension of the features to zero mean, unit variance
        # (features.shape = (num_admissions, num_feature_dims))
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)

        return features_norm

    def compute_features(self, split, normalized_dimensions=False):

        # Compute features if not already done
        if split not in self._features_cache:
            # Get admission indices for the split
            adm_indices = self.get_split_indices(split)

            # Compute bottleneck features for all admissions
            # Instead of doing this computation in one big step, it is performed one by one, since the time series
            # length between the admissions is different
            all_adm_features = []

            for adm_idx in tqdm(adm_indices, desc=f"Computing features for {len(adm_indices)} admissions..."):
                adm_features = self._load_or_calculate_features(adm_idx=adm_idx)
                all_adm_features.append(adm_features)

            # Concatenate the features
            self._features_cache[split] = np.stack(all_adm_features, axis=0)

            # Save features file (if any new features were computed)
            if self._new_features_computed:
                self._write_features_file()

        # Get stored features
        features = self._features_cache[split]  # shape: (num_adm, num_features)

        # If requested, normalize feature dimensions to zero mean, unit variance
        if normalized_dimensions:
            features = self._normalize_feature_dimensions(features)

        return features

    def _load_or_calculate_features(self, adm_idx):
        # Try to load feature
        hadm_id = self.prep.get_hadm_id(adm_idx)
        if len(self._features_loaded) > 0 and hadm_id in self._features_loaded:
            return self._features_loaded[hadm_id]

        # Loading was impossible, so compute features for this admission
        self._new_features_computed = True  # Remember that new features were computed

        if not self.baseline_mode:
            # Get a trained encoder model
            enc_model = self.get_trained_dyn_enc_model()

            # Get input data
            x_inputs = self._prepare_input_data(adm_idx=adm_idx)

            # Alert if dynamic inputs are zero-length: This situation should never occur since zero-length admissions
            # are filtered out prior to training.
            _, dyn_len, _ = x_inputs.shape
            if dyn_len == 0:
                # Find out information about the offending admission
                logging.error(f"Admission with idx {adm_idx} and hadm_id {self.prep.get_hadm_id(adm_idx)}"
                              f" has zero-length dynamic data chart!")

            # Compute
            adm_features = enc_model.predict(
                x=x_inputs
            )

            # Select first entry from (trivial) batch dimension
            adm_features = adm_features[0]

            # Abort if features contain NaNs
            assert not any(np.isnan(adm_features)), f"Features for admission with index {adm_idx} and hadm_id " \
                                                    f"{self.prep.get_hadm_id(adm_idx)} contain NaNs: {adm_features}"
        else:
            # Retrieve baseline features
            adm_pos_in_baseline_features = self.get_split_indices(io.split_name_all).index(adm_idx)
            adm_features = self._baseline_features[adm_pos_in_baseline_features]

        # Save features in memory. They will be later saved to disk.
        # adm_features.shape: (num_features_dims,)
        self._features_loaded[hadm_id] = adm_features

        return adm_features

    def _write_features_file(self):
        # Load features saved in file
        if os.path.isfile(self._features_file_path):
            features_on_disk = io.read_pickle(self._features_file_path)
        else:
            features_on_disk = {}

        # Unify with features currently loaded into memory
        features_on_disk.update(self._features_loaded)  # (prefer memory features over disk features)
        self._features_loaded = features_on_disk

        # Save unified features
        io.write_pickle(self._features_loaded, self._features_file_path)

        logging.info(f"Saved features to {self._features_file_path}")

    def load_model_checkpoints(self):
        # Model can not be loaded if in baseline mode
        if self.baseline_mode:
            return False

        # Load dynamic model files
        dyn_loaded = False
        if os.path.exists(self._dyn_model_path) and os.path.exists(self._dyn_model_enc_path):

            self._dyn_model = self._load_model(self._dyn_model_path.as_posix())
            self._dyn_model.compile(optimizer=self.optimizer, loss=self.loss)
            logging.info(f"Loaded dynamic model from {self._dyn_model_path}")

            self._dyn_model_enc = self._load_model(self._dyn_model_enc_path.as_posix())

            dyn_loaded = True

        # Load training meta file
        meta_loaded = False
        if os.path.exists(self._training_meta_file_path):
            with open(self._training_meta_file_path, "r") as json_file:
                self.training_state = json.load(json_file)

            meta_loaded = True

        models_loaded_checks = [dyn_loaded, meta_loaded]

        all_models_loaded = all(models_loaded_checks)

        # Load admission features
        if os.path.exists(self._features_file_path):
            self._features_loaded = io.read_pickle(self._features_file_path)

            logging.info(f"Loaded features from {self._features_file_path}")

        return all_models_loaded

    def save_model_checkpoints(self, epoch: int):
        # Update current epoch
        self.training_state['epoch'] = epoch + 1

        # Training timing statistics and loss
        training_stats = {
            'start_time_utc': self.training_start_time_utc.isoformat(),
            'losses_train': self.losses_train,
            'losses_val': self.losses_val,
        }
        if self.training_state['done']:
            training_stats.update({
                'time_taken_human': hf.format_timespan(self.training_end_time_utc - self.training_start_time_utc),
                'end_time_utc': self.training_end_time_utc.isoformat()
            })
        self.training_state['stats'] = training_stats

        # Save model checkpoints to disk
        self._save_model(model=self._dyn_model, path=self._dyn_model_path.as_posix())
        self._save_model(model=self._dyn_model_enc, path=self._dyn_model_enc_path.as_posix())

        # Save training meta file
        with open(self._training_meta_file_path, "w") as json_file:
            json.dump(self.training_state, json_file, sort_keys=True, indent=4, separators=(",", ": "))

    @staticmethod
    def _save_model(model, path):
        model.save(path)

    @staticmethod
    def _load_model(path):
        return load_model(path)

