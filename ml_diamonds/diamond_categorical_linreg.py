import os
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow import feature_column
from matplotlib import pyplot as plt

from common import get_dir
#-----------------------------------------------------------------------------------------------------------------------
"""
diamond_categorical_linreg.py

    This script uses the tfds dataset 'diamonds' to build a linear regression TF model.
    - Features:
        - 'carat' as numeric
        - 'color' as categorical with vocab embedded
        - 'clarity' as categorical with vocab embedded
    - Label:
        - 'price' 

"""
#-----------------------------------------------------------------------------------------------------------------------
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

batch_size = 32

def scale_filter_func(ds):
    ds = ds.filter(lambda x, y: y < 10000)
    ds = ds.filter(lambda x, y: x['carat'] < 2)
    ds = ds.map(lambda x, y: [x, y / 100])
    ds = ds.batch(batch_size)
    return ds

train_ds = train_ds.apply(scale_filter_func)
test_ds = test_ds.apply(scale_filter_func)
val_ds = val_ds.apply(scale_filter_func)

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
epochs = 4
learning_rate = 0.01

#-----------------------------------------------------------------------------------------------------------------------
""" Define the model structure """
m1 = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(units=1, input_shape=[1])
])

#-----------------------------------------------------------------------------------------------------------------------
""" Compile the model with proper optimizer and metrics """
m1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss="mean_squared_error",
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

#-----------------------------------------------------------------------------------------------------------------------
""" Train the model and get the history """
history = m1.fit(train_ds, epochs=epochs, validation_data=val_ds)

epochs = history.epoch
hist = pd.DataFrame(history.history)

loss = hist['loss']
val_loss = hist['val_loss']

rmse = hist['root_mean_squared_error']
val_rmse = hist['val_root_mean_squared_error']
print(epochs)
print(loss)

#-----------------------------------------------------------------------------------------------------------------------
""" Save the model """

# Get main directory and the saved_models directory
main_directory_path = get_dir.getCurrDir()
saved_models_extension = 'savedmodels'
saved_models_path = os.path.join(main_directory_path, saved_models_extension)
categorical_linreg_extension = 'categorical_linreg'
categorical_linreg_extension_path = os.path.join(saved_models_path, categorical_linreg_extension)
m1.save(categorical_linreg_extension_path)

#-----------------------------------------------------------------------------------------------------------------------
""" Plot Epochs versus Loss & Val. Loss """
plt.title('m1 Loss and Validation Loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(epochs[1:], loss[1:], 'r-', label='Training Loss')
plt.plot(epochs[1:], val_loss[1:], 'b-', label='Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
""" Plot Epochs versus RMSE & Val. RMSE """
plt.title('m1 RMSE and Validation RMSE')
plt.xlabel('Epoch Number')
plt.ylabel('RMSE')
plt.plot(epochs[1:], rmse[1:], 'g-', label='Training Loss')
plt.plot(epochs[1:], val_rmse[1:], 'c-', label='Validation Loss')
plt.legend(['Training RMSE', 'Validation RMSE'])
plt.show()
#-----------------------------------------------------------------------------------------------------------------------

