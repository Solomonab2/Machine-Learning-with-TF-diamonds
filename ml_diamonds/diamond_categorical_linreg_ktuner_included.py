import os
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow import feature_column
from tensorflow import keras
import keras_tuner as kt
from matplotlib import pyplot as plt
#-----------------------------------------------------------------------------------------------------------------------
"""
diamond_categorical_linreg_k_tuner.py

    This module uses the tfds dataset 'diamonds' to build a linear regression TF model.
    - Features:
        - 'carat' as numeric
        - 'color' as categorical with vocab embedded
        - 'clarity' as categorical with vocab embedded
    - Label:
        - 'price' 
        
    Additionally, this module uses Keras Tuner to tune the model's hyperparameters.

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
""" Define the model builder """
def model_builder(hp):
    """ builds the model """
    # Define the model type and input layer
    m1 = tf.keras.Sequential()
    m1.add(feature_layer)

    # Number of units in the first dense relu layer
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    m1.add(keras.layers.Dense(units=hp_units, activation='relu'))
    # Define the output layer
    m1.add(layers.Dense(units=1, input_shape=[1]))

    # Define the learning rate values
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])

    # Compile the model and return it
    m1.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="mean_squared_error",
        metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return m1

#-----------------------------------------------------------------------------------------------------------------------
""" Instantiate the keras tuner """
# Parameters for the tuner
objective='val_loss'
max_epochs = 10
factor = 3
directory= 'ktuner'
project_name = 'diamond_categorical_linreg_k_tuner'
# Instantiate the tuner
tuner = kt.Hyperband(model_builder,
                     objective=objective,
                     max_epochs=max_epochs,
                     factor=factor,
                     directory=directory,
                     project_name=project_name)
# Create a callback for the early stopping
patience = 5
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

#-----------------------------------------------------------------------------------------------------------------------
""" Run the hyperparameter search """
search_epochs = 5
tuner.search(train_ds, epochs=search_epochs, validation_data=val_ds, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
Optimal number of units in the first densely-connected
layer is: {best_hps.get('units')}  
""")

print(f"""
Optimal learning rate for the optimizer
is: {best_hps.get('learning_rate')}.
""")

#-----------------------------------------------------------------------------------------------------------------------
""" Train the model with the optimal hyperparameters and get the history """
# Create the model with our optimal hyperparameters
model = tuner.hypermodel.build(best_hps)
# Train the model and get the history
epochs_to_test = 3
history = model.fit(train_ds, epochs=epochs_to_test, validation_data=val_ds)
# Validation loss per epoch
val_loss_per_epoch = history.history['val_loss']
best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# Re-instantiate the hypermodel with the optimal hyperparameters
hypermodel = tuner.hypermodel.build(best_hps)
# Retrain the model with our optimal number of epochs
hyperhistory = hypermodel.fit(train_ds, validation_data=val_ds, epochs=best_epoch)

# Convert history to dataframe
epochs = hyperhistory.epoch
hist = pd.DataFrame(hyperhistory.history)

# Get loss and validation loss
loss = hist['loss']
val_loss = hist['val_loss']

#-----------------------------------------------------------------------------------------------------------------------
""" Save the model """
saved_models_path = 'savedmodels'
model_name = 'categorical_linreg_k_tuner'
save_path = os.path.join(saved_models_path, model_name)
hypermodel.save(save_path)

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


