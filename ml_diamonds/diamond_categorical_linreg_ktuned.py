import os
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import feature_column
import diamond_ktuner
from matplotlib import pyplot as plt

# -----------------------------------------------------------------------------------------------------------------------
"""
diamond_categorical_linreg_ktuned.py

    This module uses the tfds dataset 'diamonds' to build a linear regression TF model.
    - Features:
        - 'carat' as numeric
        - 'color' as categorical with vocab embedded
        - 'clarity' as categorical with vocab embedded
    - Label:
        - 'price' 

    Additionally, this module uses Keras Tuner to tune the model's hyperparameters.

"""
# -----------------------------------------------------------------------------------------------------------------------
""" Load the dataset from tfds """
ds = tfds.load('diamonds', split='train', shuffle_files=True, as_supervised=True)

# -----------------------------------------------------------------------------------------------------------------------
""" Define the train, val, and test split """
train_split = 0.7
train_count = int(train_split * len(ds))

train_ds = ds.take(train_count)
test_ds = ds.skip(train_count)

validation_split = 0.2
train_count = train_count - int(validation_split * train_count)

val_ds = train_ds.skip(train_count)
train_ds = train_ds.take(train_count)

# -----------------------------------------------------------------------------------------------------------------------
""" Remove any outliers from train & test datasets """

batch_size = 32

def scale_filter_func(ds):
    """ trim entries with price over 10k and with a carat of over 2 and scale the prices down by a factor of 100 """
    ds = ds.filter(lambda x, y: y < 10000)
    ds = ds.filter(lambda x, y: x['carat'] < 2)
    ds = ds.map(lambda x, y: [x, y / 100])
    ds = ds.batch(batch_size)
    return ds

train_ds = train_ds.apply(scale_filter_func)
test_ds = test_ds.apply(scale_filter_func)
val_ds = val_ds.apply(scale_filter_func)

# -----------------------------------------------------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------------------------------------------------
""" Instantiate the keras tuner """
# Parameters for the tuner
min_middle_layer_units = 256
max_middle_layer_units = 512
middle_layer_units_step = 32
learning_rates = [1e-1, 1e-2, 1e-3]
loss_function = 'mean_squared_error'
metrics = [
    tf.keras.metrics.RootMeanSquaredError()
]
objective = 'val_loss'
max_tuning_epochs = 10
tuning_factor = 3
directory = 'ktuner'
project_name = 'diamond_categorical_linreg_k_tuner'

tuner = diamond_ktuner.setup_tuner_linreg(feature_columns, min_middle_layer_units, max_middle_layer_units,
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
objective = 'val_loss'
maximize_objective = False
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

# Get loss and validation loss
loss = hist['loss']
val_loss = hist['val_loss']

# -----------------------------------------------------------------------------------------------------------------------
""" Save the model """
saved_models_path = 'savedmodels'
model_name = 'categorical_linreg_k_tuner'
save_path = os.path.join(saved_models_path, model_name)
hyperm1.save(save_path)

# -----------------------------------------------------------------------------------------------------------------------
""" Plot Epochs versus Loss & Val. Loss """
plt.title('m1 Loss and Validation Loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(epochs[1:], loss[1:], 'r-', label='Training Loss')
plt.plot(epochs[1:], val_loss[1:], 'b-', label='Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

# -----------------------------------------------------------------------------------------------------------------------


