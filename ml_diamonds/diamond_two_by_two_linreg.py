from os.path import exists
import os
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow import feature_column
from matplotlib import pyplot as plt

#-----------------------------------------------------------------------------------------------------------------------
"""
diamond_two_by_two_linreg.py

    This script uses the tfds dataset 'diamonds' to build a linear regression TF model.
    - builds 4 versions of the model for different optimizers and with and without relu layer
    - saves the models to savedmodels/two_by_two_linreg/
    - plots all four models

"""
#-----------------------------------------------------------------------------------------------------------------------
""" Load the dataset from tfds """
ds = tfds.load('diamonds', split='train', shuffle_files=True, as_supervised=True)

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
""" Scale, filter, and prepare input pipeline """
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

carat_numeric = feature_column.numeric_column('carat')
feature_columns.append(carat_numeric)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

#-----------------------------------------------------------------------------------------------------------------------
""" Define the build_model function """
def build_model(use_relu, optimizer):
    layers = []
    layers.append(feature_layer)
    if use_relu:
        layers.append(tf.keras.layers.Dense(128, activation='relu'))
    layers.append(tf.keras.layers.Dense(units=1, input_shape=[1]))

    model = tf.keras.Sequential(layers)

    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

#-----------------------------------------------------------------------------------------------------------------------
""" Define the train_model function """
def train_model(model, dataset, validation_set, epochs):
    history = model.fit(dataset, validation_data=validation_set, epochs=epochs)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    loss = hist['loss']
    val_loss = hist['val_loss']
    rmse = hist['root_mean_squared_error']
    val_rmse = hist['val_root_mean_squared_error']
    return epochs, hist, loss, val_loss, rmse, val_rmse

#-----------------------------------------------------------------------------------------------------------------------
""" Build the models """
learning_rate = 0.5
# Build model_1
m1_name = 'model_1_adam'
m1_optimizer = tf.keras.optimizers.Adam(learning_rate)
m1_use_relu = False
m1 = build_model(use_relu=m1_use_relu, optimizer=m1_optimizer)
# Build model_2
m2_name = 'model_2_adam_relu'
m2_optimizer = tf.keras.optimizers.Adam(learning_rate)
m2_use_relu = True
m2 = build_model(use_relu=m2_use_relu, optimizer=m2_optimizer)
# Build model_3
m3_name = 'model_3_RMSprop'
m3_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
m3_use_relu = False
m3 = build_model(use_relu=m3_use_relu, optimizer=m3_optimizer)
# Build model_4
m4_name = 'model_4_RMSprop_relu'
m4_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
m4_use_relu = True
m4 = build_model(use_relu=m4_use_relu, optimizer=m4_optimizer)

#-----------------------------------------------------------------------------------------------------------------------
""" Train the models """
epochs = 4
# Train model_1
print("Training model_1")
m1_epochs, m1_hist, m1_loss, m1_val_loss, m1_rmse, m1_val_rmse = train_model(m1, train_ds, val_ds, epochs)
# Train model_2
m2_epochs, m2_hist, m2_loss, m2_val_loss, m2_rmse, m2_val_rmse = train_model(m2, train_ds, val_ds, epochs)
# Train model_3
m3_epochs, m3_hist, m3_loss, m3_val_loss, m3_rmse, m3_val_rmse = train_model(m3, train_ds, val_ds, epochs)
# Train model_4
m4_epochs, m4_hist, m4_loss, m4_val_loss, m4_rmse, m4_val_rmse = train_model(m4, train_ds, val_ds, epochs)

all_models = [m1, m2, m3, m4]
all_model_names = [m1_name, m2_name, m3_name, m4_name]
all_epochs = [m1_epochs, m2_epochs, m3_epochs, m4_epochs]
all_hists = [m1_hist, m2_hist, m3_hist, m4_hist]
all_losses = [m1_loss, m2_loss, m3_loss, m4_loss]
all_val_losses = [m1_val_loss, m2_val_loss, m3_val_loss, m4_val_rmse]
all_rmses = [m1_rmse, m2_rmse, m3_rmse, m4_rmse]
all_val_rmses = [m1_val_rmse, m2_val_rmse, m3_val_rmse, m4_val_rmse]
all_max_losses = []
all_min_losses = []
all_max_rmses = []
all_min_rmses = []
#-----------------------------------------------------------------------------------------------------------------------

""" Save the models """
saved_models_path = 'savedmodels'
model_name = 'two_by_two_linear_regression'
save_path = os.path.join(saved_models_path, model_name)
history_extension = 'history'
history_file_name = 'history.csv'
history_save_path = os.path.join(history_extension, history_file_name)

for i in range(4):
    print('Saving {model_name}...'.format(model_name=all_model_names[i]))
    model_path = os.path.join(save_path, all_model_names[i])
    all_models[i].save(model_path)
    history_directory_path = os.path.join(model_path, history_extension)

    if not exists(history_directory_path):
        os.mkdir(history_directory_path)
    history_path = os.path.join(model_path, history_save_path)
    all_hists[i].to_csv(history_path)

    all_max_losses.append(max(all_losses[i][1:]))
    all_max_losses.append(max(all_val_losses[i][1:]))
    all_min_losses.append(min(all_losses[i][1:]))
    all_min_losses.append(min(all_val_losses[i][1:]))
    all_max_rmses.append(max(all_rmses[i][1:]))
    all_max_rmses.append(max(all_val_rmses[i][1:]))
    all_min_rmses.append(min(all_rmses[i][1:]))
    all_min_rmses.append(min(all_val_rmses[i][1:]))

#-----------------------------------------------------------------------------------------------------------------------
""" Plot the loss models """
fig, axs1 = plt.subplots(2, 4)
fig.tight_layout()
v_padding = 0.03
max_loss = max(all_max_losses) + (v_padding * max(all_max_losses))
min_loss = min(all_min_losses) - (v_padding * min(all_min_losses))
max_rmse = max(all_max_rmses) + (v_padding * max(all_max_rmses))
min_rmse = min(all_min_rmses) - (v_padding * min(all_min_rmses))

# Plot model_1
axs1[0, 0].set_title(all_model_names[0])
axs1[0, 0].plot(all_epochs[0][1:], all_losses[0][1:], 'b-', label='Training Loss')
axs1[0, 0].plot(all_epochs[0][1:], all_val_losses[0][1:], 'r-', label='Validation Loss')
axs1[0, 0].legend(['Training Loss', 'Validation Loss'])
axs1[0, 0].set_ylim(min_loss, max_loss)

axs1[0, 1].set_title(all_model_names[0])
axs1[0, 1].plot(all_epochs[0][1:], all_rmses[0][1:], 'g-', label='Training RMSE')
axs1[0, 1].plot(all_epochs[0][1:], all_val_rmses[0][1:], 'c-', label='Validation RMSE')
axs1[0, 1].legend(['Training RMSE', 'Validation RMSE'])
axs1[0, 1].set_ylim(min_rmse, max_rmse)

# Plot model_2
axs1[0, 2].set_title(all_model_names[1])
axs1[0, 2].plot(all_epochs[1][1:], all_losses[1][1:], 'b-', label='Training Loss')
axs1[0, 2].plot(all_epochs[1][1:], all_val_losses[1][1:], 'r-', label='Validation Loss')
axs1[0, 2].legend(['Training Loss', 'Validation Loss'])
axs1[0, 2].set_ylim(min_loss, max_loss)

axs1[0, 3].set_title(all_model_names[1])
axs1[0, 3].plot(all_epochs[1][1:], all_rmses[1][1:], 'g-', label='Training RMSE')
axs1[0, 3].plot(all_epochs[1][1:], all_val_rmses[1][1:], 'c-', label='Validation RMSE')
axs1[0, 3].legend(['Training RMSE', 'Validation RMSE'])
axs1[0, 3].set_ylim(min_rmse, max_rmse)

# Plot model_3
axs1[1, 0].set_title(all_model_names[2])
axs1[1, 0].plot(all_epochs[2][1:], all_losses[2][1:], 'b-', label='Training Loss')
axs1[1, 0].plot(all_epochs[2][1:], all_val_losses[2][1:], 'r-', label='Validation Loss')
axs1[1, 0].legend(['Training Loss', 'Validation Loss'])
axs1[1, 0].set_ylim(min_loss, max_loss)

axs1[1, 1].set_title(all_model_names[2])
axs1[1, 1].plot(all_epochs[2][1:], all_rmses[2][1:], 'g-', label='Training RMSE')
axs1[1, 1].plot(all_epochs[2][1:], all_val_rmses[2][1:], 'c-', label='Validation RMSE')
axs1[1, 1].legend(['Training RMSE', 'Validation RMSE'])
axs1[1, 1].set_ylim(min_rmse, max_rmse)

# Plot model_4
axs1[1, 2].set_title(all_model_names[3])
axs1[1, 2].plot(all_epochs[3][1:], all_losses[3][1:], 'b-', label='Training Loss')
axs1[1, 2].plot(all_epochs[3][1:], all_val_losses[3][1:], 'r-', label='Validation Loss')
axs1[1, 2].legend(['Training Loss', 'Validation Loss'])
axs1[1, 2].set_ylim(min_loss, max_loss)

axs1[1, 3].set_title(all_model_names[3])
axs1[1, 3].plot(all_epochs[3][1:], all_rmses[3][1:], 'g-', label='Training RMSE')
axs1[1, 3].plot(all_epochs[3][1:], all_val_rmses[3][1:], 'c-', label='Validation RMSE')
axs1[1, 3].legend(['Training RMSE', 'Validation RMSE'])
axs1[1, 3].set_ylim(min_rmse, max_rmse)

# Display the results
plt.show()
#-----------------------------------------------------------------------------------------------------------------------