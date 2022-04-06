import os
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow import feature_column
from matplotlib import pyplot as plt

from common import get_dir
from common import plot_metrics_curve
#-----------------------------------------------------------------------------------------------------------------------
"""
diamond_categorical_binary_classification.py

    This script uses the tfds dataset 'diamonds' to build a binary classification TF model.
    - Features:
        - 'carat' as numeric
        - 'color' as categorical with vocab embedded
        - 'clarity' as categorical with vocab embedded
    - Label:
        - 'price' 

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

batch_size = 50
price_threshold = 5400

# Filter function trims dataset by removing potential outliers
def filter_func(ds):
    ds = ds.filter(lambda x, y: y < 10000)
    ds = ds.filter(lambda x, y: x['carat'] < 2)
    return ds

train_ds = train_ds.apply(filter_func)
test_ds = test_ds.apply(filter_func)
val_ds = val_ds.apply(filter_func)

# This map function maps the price labels to a binary value of 0 or 1 depending on the threshold
def map_func(features, labels):
    bool_labels = tf.cast((labels > price_threshold), dtype=tf.int32)
    return features, bool_labels

train_ds = train_ds.map(map_func)
test_ds = test_ds.map(map_func)
val_ds = val_ds.map(map_func)

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

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

#-----------------------------------------------------------------------------------------------------------------------
""" Define hyperparameters for the model """
epochs = 6
batch_size = 10
learning_rate = 0.01
classification_threshold = 0.65

#-----------------------------------------------------------------------------------------------------------------------
""" Define the model structure """
m1 = tf.keras.Sequential([
    feature_layer,
    #layers.Dense(128, activation='relu'),
    layers.Dense(units=1, input_shape=(1,), activation=tf.sigmoid),])

#-----------------------------------------------------------------------------------------------------------------------
""" Define the model's metrics """
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=classification_threshold),
    tf.keras.metrics.Precision(thresholds=classification_threshold, name='precision'),
    tf.keras.metrics.Recall(thresholds=classification_threshold, name='recall'),
    tf.keras.metrics.AUC(num_thresholds=100, name='auc'),]

#-----------------------------------------------------------------------------------------------------------------------
""" Compile the model with proper optimizer and metrics """
m1.compile(optimizer=tf.keras.optimizers.RMSprop(
    learning_rate=learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=METRICS)

#-----------------------------------------------------------------------------------------------------------------------
""" Train the model and get the history"""
history = m1.fit(train_ds, epochs=epochs, validation_data=val_ds)

epochs = history.epoch
hist = pd.DataFrame(history.history)

#-----------------------------------------------------------------------------------------------------------------------
""" Save the model """

# Get main directory and the saved_models directory
main_directory_path = get_dir.getCurrDir()
saved_models_extension = 'savedmodels'
saved_models_path = os.path.join(main_directory_path, saved_models_extension)
categorical_binary_classification_extension = 'categorical_binary_classification'
categorical_binary_classification_path = os.path.join(saved_models_path, categorical_binary_classification_extension)
m1.save(categorical_binary_classification_path)

#-----------------------------------------------------------------------------------------------------------------------
""" Plot the accuracy, precision, and recall """
list_of_metrics_to_plot = ['accuracy', 'precision', 'recall']
plot_metrics_curve.plot(epochs, hist, list_of_metrics_to_plot)

# ----------------------------------------------------------------------------------------------------------------------
