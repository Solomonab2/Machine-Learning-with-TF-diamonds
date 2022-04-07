import os
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow import feature_column
import diamond_ktuner
from common import plot_metrics_curve
#-----------------------------------------------------------------------------------------------------------------------
"""
diamond_categorical_binary_classification.py

    This module uses the tfds dataset 'diamonds' to build a binary classification TF model.
    - Features:
        - 'carat' as numeric
        - 'color' as categorical with vocab embedded
        - 'clarity' as categorical with vocab embedded
    - Label:
        - 'price' 

    Additionally, this module uses Keras Tuner to tune the model's hyperparameters.
    
"""
# ----------------------------------------------------------------------------------------------------------------------
""" Load the dataset from tfds """
ds = tfds.load('diamonds', split='train', shuffle_files=True, as_supervised=True)

#-----------------------------------------------------------------------------------------------------------------------
""" Define the train, val, and test split """
train_split = 0.7
train_count = int(train_split * len(ds))

train_ds = ds.take(train_count)
test_ds = ds.skip(train_count)

validation_split = 0.2
train_count = train_count - int(validation_split * train_count)

val_ds = train_ds.skip(train_count)
train_ds = train_ds.take(train_count)

#-----------------------------------------------------------------------------------------------------------------------
""" Remove any outliers from train & test datasets """

# Filter function trims dataset by removing potential outliers
def filter_func(ds):
    """ trim entries with price over 10k and with a carat of over 2 """
    ds = ds.filter(lambda x, y: y < 10000)
    ds = ds.filter(lambda x, y: x['carat'] < 2)
    return ds

train_ds = train_ds.apply(filter_func)
test_ds = test_ds.apply(filter_func)
val_ds = val_ds.apply(filter_func)

# This map function maps the price labels to a binary value of 0 or 1 depending on the threshold
price_threshold = 5400
def map_func(features, labels):
    """ map prices as a binary classification based on threshold """
    bool_labels = tf.cast((labels > price_threshold), dtype=tf.int32)
    return features, bool_labels

train_ds = train_ds.map(map_func)
test_ds = test_ds.map(map_func)
val_ds = val_ds.map(map_func)

batch_size = 50
train_ds = train_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

#-----------------------------------------------------------------------------------------------------------------------
""" Create feature columns from desired data """
feature_columns = []

carat_numeric_column = tf.feature_column.numeric_column('carat')
feature_columns.append(carat_numeric_column)

vocab_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
categorical_clarity = feature_column.categorical_column_with_vocabulary_list(
    key='clarity', vocabulary_list=vocab_list,
    default_value=0)
categorical_clarity_embedding = feature_column.embedding_column(categorical_column=categorical_clarity, dimension=8)
feature_columns.append(categorical_clarity_embedding)

vocab_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
categorical_color = feature_column.categorical_column_with_vocabulary_list(
    key='color', vocabulary_list=vocab_list,
    default_value=0)
categorical_color_embedding = feature_column.embedding_column(categorical_column=categorical_color, dimension=8)
feature_columns.append(categorical_color_embedding)

#-----------------------------------------------------------------------------------------------------------------------
""" Instantiate the keras tuner """
# Parameters for the tuner
min_middle_layer_units = 256
max_middle_layer_units = 512
middle_layer_units_step = 32
learning_rates = [1e-2, 1e-3]
loss_function = keras.losses.BinaryCrossentropy()
classification_threshold = 0.65
metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=classification_threshold),
    tf.keras.metrics.Precision(thresholds=classification_threshold, name='precision'),
    tf.keras.metrics.Recall(thresholds=classification_threshold, name='recall'),
    tf.keras.metrics.AUC(num_thresholds=100, name='auc'),
]
objective = 'val_accuracy'
max_tuning_epochs = 10
tuning_factor = 3
directory = 'ktuner'
project_name = 'diamond_categorical_binary_classification_k_tuner'

tuner = diamond_ktuner.setup_tuner_binary_classification(feature_columns, min_middle_layer_units, max_middle_layer_units,
                                                         middle_layer_units_step, learning_rates, loss_function, metrics,
                                                         objective, max_tuning_epochs, tuning_factor, directory, project_name)

# -----------------------------------------------------------------------------------------------------------------------
""" Run the hyperparameter search """
search_epochs = 5
stop_early_monitor = 'val_loss'
stop_early_patience = 5
best_hps = diamond_ktuner.tune_hyperparameters(tuner, train_ds, val_ds, search_epochs, stop_early_monitor, stop_early_patience)

print(f"""
Optimal number of units in the first densely-connected
layer is: {best_hps.get('units')}  
""")

print(f"""
Optimal learning rate for the optimizer
is: {best_hps.get('learning_rate')}.
""")

# -----------------------------------------------------------------------------------------------------------------------
""" Train the model with the optimal hyperparameters and get the best epoch """
epochs_to_test = 4
objective = 'val_accuracy'
maximize_objective = True
best_epoch = diamond_ktuner.get_best_epoch(tuner, train_ds, val_ds, best_hps, objective, maximize_objective, epochs_to_test)
print('Best epoch: %d' % (best_epoch,))

# -----------------------------------------------------------------------------------------------------------------------
""" Build the hypermodel """
# Re-instantiate the tuner with the optimal hyperparameters
hyperm1 = tuner.hypermodel.build(best_hps)
# Retrain the model with our optimal number of epochs
hyperhistory = hyperm1.fit(train_ds, validation_data=val_ds, epochs=best_epoch)

# Convert history to dataframe
epochs = hyperhistory.epoch
hist = pd.DataFrame(hyperhistory.history)

#-----------------------------------------------------------------------------------------------------------------------
""" Save the model """
saved_models_path = 'savedmodels'
model_name = 'categorical_binary_classification_k_tuner'
save_path = os.path.join(saved_models_path, model_name)
hyperm1.save(save_path)

#-----------------------------------------------------------------------------------------------------------------------
""" Plot the accuracy, precision, and recall """
list_of_metrics_to_plot = ['accuracy', 'precision', 'recall']
plot_metrics_curve.plot(epochs, hist, list_of_metrics_to_plot)

# ----------------------------------------------------------------------------------------------------------------------
