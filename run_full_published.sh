#!/bin/bash

# The following command runs the full MIMIC-III pipeline (provided a hosted database is correctly configured) with the same settings that were used in the publication.

python full_pipeline_mimic.py --evaluated_fraction 1.0 --id publication_full_pipeline --admissions 60000 --early_stopping_patience 8 --model_num_dec_dense_layers 1 --model_rnn_size 158 --model_rnn_layers 1 --model_rnn_type gru --model_bottleneck_size 46 --model_dropout_rate 0.2673737979211852 --model_activation elu --model_temporal_pool_mode average --model_bidirectional_merge_mode sum --model_normalization_type layer --model_input_noise_sigma 0.05728409435373203 --model_reconstruct_times True --model_times_to_encoder True --model_residual_dense False --training_batch_size 4 --training_max_batch_len 0 --training_loss huber_loss --training_optimizer Adam --training_learning_rate 0.00075 --training_clip_grad_norm True --training_shuffle_admissions False --prep_scaling_mode quantile --prep_positional_encoding_dims 64

