# Math and data
import numpy as np

# NN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Masking, RepeatVector, Lambda, TimeDistributed, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, BatchNormalization, LayerNormalization, Bidirectional, GaussianNoise, \
    concatenate, Activation
from tensorflow.keras.regularizers import L2
from tensorflow.keras import backend as keras_backend

from common import io

# Defaults for select hyperparameters (those that are also used elsewhere)
bottleneck_size_default = 46


class DynModel:
    """
    Neural network model. Serves as an abstraction for the specific details of a model and hides them from the
    user of the model.
    """

    def __init__(self, prep, trainer, rnn_size=158, bottleneck_size=None, dropout_rate=0.125, num_dec_dense_layers=1,
                 activation='elu', rnn_layers=1, rnn_type='gru', temporal_pool_mode='average',
                 normalization_type='layer', bidirectional_merge_mode='sum', input_noise_sigma=0.0625,
                 post_bottleneck_noise_sigma=0.0, reconstruct_times=True, times_to_encoder=True,
                 l2_regularization=0.0):
        # ---- Set up data-dependent model settings ----
        self.dyn_width = len(prep.dyn_data_columns)  # Number of attributes of dynamic data
        if prep.use_positional_encoding:
            self.dyn_width += prep.positional_encoding_dims

        # Determine the number of temporal dimensions
        self.time_dims = prep.positional_encoding_dims
        if prep.positional_encoding_dims > 1:
            # (if using positional encodings, time column (e.g. 'charttime') is also present)
            self.time_dims += 1
        # The last self.time_dims features of the input are temporal features

        # ---- Set up data-independent model settings ----
        if bottleneck_size is None:
            bottleneck_size = bottleneck_size_default
        self.bottleneck_size = int(bottleneck_size)
        self.bottleneck_size += self.bottleneck_size % 2  # Make sure the bottleneck size is divisible by two
        self.dropout_rate = float(dropout_rate)
        self.activation = str(activation)
        self.input_noise_sigma = float(input_noise_sigma)
        self.post_bottleneck_noise_sigma = float(post_bottleneck_noise_sigma)
        self.num_dec_dense_layers = int(num_dec_dense_layers)
        self.l2_regularization = float(l2_regularization)

        # Temporal "pooling"
        #   slice: Take last temporal slice
        #   average: Take average over time dimension
        #   max: Take maximum over time dimension
        assert temporal_pool_mode in ['slice', 'average', 'max',
                                      'all'], f"Temporal pool mode '{temporal_pool_mode}' unknown!"
        self.temporal_pool_mode = temporal_pool_mode

        # RNN
        self.rnn_size = int(rnn_size)
        self.rnn_layers = int(rnn_layers)
        rnn_type = str(rnn_type).lower()
        assert rnn_type in ['gru', 'lstm'], f"RNN type '{rnn_type}' unknown!"
        self._rnn_layer = {
            'gru': GRU,
            'lstm': LSTM
        }[rnn_type]

        # Reconstruct times? If True, timing features will be reconstructed. If False, timing features will not be
        # reconstructed by the model. In both cases, timing features are utilized to facilitate a reconstruction over
        # time.
        self.reconstruct_times = io.str_to_bool(reconstruct_times)

        # If True, allow encoder to see the timing information
        self.times_to_encoder = io.str_to_bool(times_to_encoder)

        # Support different kinds of normalization within model
        normalization_type = str(normalization_type).lower()
        assert normalization_type in ['batch', 'layer', 'disable'], f"Normalization type {normalization_type} unknown!"
        self.use_normalization = normalization_type != 'disable'
        self._norm_layer = {
            'batch': BatchNormalization,
            'layer': LayerNormalization
        }[normalization_type]

        # Bidirectional: If not disabled, process input forwards and backwards
        bidirectional_merge_mode = str(bidirectional_merge_mode).lower()
        assert bidirectional_merge_mode in ['sum', 'mul', 'ave', 'concat', 'disable'], \
            f"Bidirectional merge mode {bidirectional_merge_mode} not legal!"
        self.bidirectional_merge_mode = bidirectional_merge_mode
        self.using_bidirectional_rnn = self.bidirectional_merge_mode != 'disable'

        # Masking is used to "pad" admissions that are too short for a batch
        self.masking_value = trainer.masking_value

        # Dynamic models
        self._dyn_model = None
        self._dyn_model_enc = None

    def get_dyn_model(self) -> Model:
        if self._dyn_model is None:
            self._build_model()
        return self._dyn_model

    def get_dyn_model_enc(self):
        if self._dyn_model_enc is None:
            self._build_model()
        return self._dyn_model_enc

    def _build_model(self):
        # Input for charts
        charts_input = Input(
            shape=(None, self.dyn_width),  # shape: (# of time steps, # of chart columns)
            name="dyn_charts_input"
        )

        # Mask input since we want to batch multiple admissions with different sequence lengths
        charts_input_masked = Masking(mask_value=self.masking_value)(charts_input)
        # shape of charts_input_masked = (None, None, self.dyn_width) = (batch, steps, features)

        # Split timing info off from input
        timing_info = charts_input_masked[:, :, -self.time_dims:]
        charts_input_masked = charts_input_masked[:, :, :-self.time_dims]

        # Determine target feature dimensions
        output_features = self.dyn_width
        if not self.reconstruct_times:
            output_features -= self.time_dims

        # Noise - it helps the model to learn to generalize
        noisy_time_series = GaussianNoise(stddev=self.input_noise_sigma)(charts_input_masked)

        # Add timing information back in (if encoder is allowed to see it)
        if self.times_to_encoder:
            noisy_time_series = concatenate([noisy_time_series, timing_info])

        # Function for generating a stacked RNN
        def stacked_rnn(sizes, initial_input, names_prefix):
            # Make sure the number of feature is divisible by two
            sizes = [s + (s % 2) for s in sizes]

            state = initial_input
            for rnn_layer_idx, num_rnn_features in enumerate(sizes):
                # If concatenating features in a bidirectional RNN, use half the features for each RNN
                if self.bidirectional_merge_mode == 'concat':
                    num_rnn_features //= 2

                # Create RNN layer
                layer_name = f"{names_prefix}_{rnn_layer_idx}"
                rnn_layer = self._rnn_layer(
                    units=num_rnn_features,
                    return_sequences=True,
                    dropout=self.dropout_rate,
                    name=layer_name,
                    kernel_regularizer=L2(l2=self.l2_regularization),
                    recurrent_regularizer=L2(l2=self.l2_regularization),
                    bias_regularizer=L2(l2=self.l2_regularization)
                )

                # Make it bidirectional
                if self.using_bidirectional_rnn:
                    rnn_layer = Bidirectional(
                        layer=rnn_layer,
                        merge_mode=self.bidirectional_merge_mode
                    )

                # Apply RNN layer
                state = rnn_layer(state)

                # Normalization to combat internal covariance shift
                if self.use_normalization:
                    state = self._norm_layer()(state)
            # shape of state (batch, steps, features)

            return state

        # The encoder consists of layers of RNNs that have fewer and fewer features (the last layer of the encoder
        # has the same number of features as the bottleneck)
        enc_state = stacked_rnn(
            sizes=np.ceil(np.linspace(self.rnn_size, self.bottleneck_size, self.rnn_layers)).astype(int),
            initial_input=noisy_time_series,
            names_prefix="enc"
        )

        # Remove the temporal dimension of enc_state in order to arrive at the *bottleneck* of the autoencoder
        if self.temporal_pool_mode == 'slice':
            # Take the last temporal slice
            bottleneck = enc_state[:, -1, :]
        elif self.temporal_pool_mode == 'max':
            bottleneck = GlobalMaxPooling1D()(enc_state)
        elif self.temporal_pool_mode == 'average':
            bottleneck = GlobalAveragePooling1D()(enc_state)
        elif self.temporal_pool_mode == 'all':
            b_start = enc_state[:, 0, :]
            b_mid = enc_state[:, keras_backend.shape(enc_state)[1] // 2, :]
            b_end = enc_state[:, -1, :]
            b_max = GlobalMaxPooling1D()(enc_state)
            b_avg = GlobalAveragePooling1D()(enc_state)
            bottleneck = concatenate([b_start, b_mid, b_end, b_max, b_avg])
        # bottleneck.shape = (None, self.bottleneck_size) = (batch, features)

        # Dense layer for bottleneck
        bottleneck = Dense(
            self.bottleneck_size,
            name='bottleneck',
            kernel_regularizer=L2(l2=self.l2_regularization),
            bias_regularizer=L2(l2=self.l2_regularization)
        )(bottleneck)
        bottleneck = self._norm_layer()(bottleneck)
        bottleneck = Activation(self.activation)(bottleneck)

        # Add some noise after the bottleneck (the idea here is that in a small neighborhood around each admission, the
        # reconstruction should be similar)
        bottleneck = GaussianNoise(stddev=self.post_bottleneck_noise_sigma)(bottleneck)

        # ENCODER OVER
        # DECODER STARTS

        # For decoding, we first repeat the bottleneck along the time dimension
        def extend_in_time(args):
            layer = args[0]
            time_layer = args[1]
            return RepeatVector(keras_backend.shape(time_layer)[1])(layer)

        bottleneck_repeated = Lambda(
            function=extend_in_time,
            output_shape=(None, int(bottleneck.shape[-1])),
            name="extend_in_time"
        )([bottleneck, charts_input_masked])
        # bottleneck_repeated now has shape (None, None, self.bottleneck_size) = (batch, steps, features)

        # Add time information back onto bottleneck - otherwise, following decoder layers have no way of knowing which
        # point in time is to be reconstructed (this is because the input data does not have a fixed time grid)
        bottleneck_w_time = concatenate([bottleneck_repeated, timing_info])

        # The decoder consists of layers of RNNs that have more and more features
        # (Note that the linspace is cut by both its first and last entry since the RNN layers should start out having
        # *more* features than the bottleneck and *fewer* than the final data dimensionality)
        bottleneck_w_time_size = self.bottleneck_size + self.time_dims
        dec_layer_sizes = np.ceil(
            np.linspace(bottleneck_w_time_size, output_features, self.rnn_layers + 2)[1:-1]
        ).astype(int)
        dec_state = stacked_rnn(
            sizes=dec_layer_sizes,
            initial_input=bottleneck_w_time,
            names_prefix="dec"
        )

        # Final layers of the decoder: Dense layers applied for each time step
        for dec_dense_idx in range(self.num_dec_dense_layers):
            dec_state = TimeDistributed(
                Dense(
                    units=output_features,
                    kernel_regularizer=L2(l2=self.l2_regularization),
                    bias_regularizer=L2(l2=self.l2_regularization)
                ),
                name=f"dec_output_dense_{dec_dense_idx}"
            )(dec_state)

            # Normalization
            dec_state = TimeDistributed(
                self._norm_layer()
            )(dec_state)

            is_last_layer = dec_dense_idx == self.num_dec_dense_layers - 1
            if not is_last_layer:
                dec_state = TimeDistributed(
                    Activation(self.activation)
                )(dec_state)

        final_dec_output = dec_state

        # Concatenate time info back onto feature dimension
        if not self.reconstruct_times:
            final_dec_output = concatenate([final_dec_output, timing_info])

        # Compose autoencoder model (used for training and for reconstruction)
        self._dyn_model = Model(
            inputs=charts_input,
            outputs=final_dec_output
        )

        # Encoder version of model (used for generating features). It shares weights with the autoencoder model
        self._dyn_model_enc = Model(
            inputs=charts_input,
            outputs=bottleneck
        )
