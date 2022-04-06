from os.path import exists
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#-----------------------------------------------------------------------------------------------------------------------
"""
diamond_two_by_two_plot.py

    This script loads the models created by diamonds_two_by_two_*.py and plots them 

"""
#-----------------------------------------------------------------------------------------------------------------------
""" Get main directory and the /savedmodels directory """
def getCurrDir():
    return os.getcwd()
def getParentDir(path):
    return os.path.abspath(os.path.join(path, os.pardir))

main_directory_path = getCurrDir()
saved_models_extension = 'savedmodels'
saved_models_path = os.path.join(main_directory_path, saved_models_extension)
two_by_two_linreg_extension = 'two_by_two_linear_regression'
two_by_two_linreg_path = os.path.join(saved_models_path, two_by_two_linreg_extension)
history_extension = 'history'
history_file_name = 'history.csv'
history_file_path = os.path.join(history_extension, history_file_name)

#-----------------------------------------------------------------------------------------------------------------------
if exists(two_by_two_linreg_path):
    """ Load the models and histories """
    m1_name = 'model_1_adam'
    m2_name = 'model_2_adam_relu'
    m3_name = 'model_3_RMSprop'
    m4_name = 'model_4_RMSprop_relu'
    all_model_names = [m1_name, m2_name, m3_name, m4_name]
    all_hists = []
    all_epochs = []
    all_losses = []
    all_val_losses = []
    all_rmses = []
    all_val_rmses = []
    all_max_losses = []
    all_min_losses = []
    all_max_rmses = []
    all_min_rmses = []

    for i in range(4):
        m_path = os.path.join(two_by_two_linreg_path, all_model_names[i])
        m_history_path = os.path.join(m_path, history_file_path)
        m_hist = pd.read_csv(m_history_path)
        all_hists.append(m_hist)
        all_epochs.append(range(1, len(m_hist) + 1))
        all_losses.append(m_hist['loss'])
        all_val_losses.append(m_hist['val_loss'])
        all_rmses.append(m_hist['root_mean_squared_error'])
        all_val_rmses.append(m_hist['val_root_mean_squared_error'])
        all_max_losses.append(max(m_hist['loss'][1:]))
        all_max_losses.append(max(m_hist['val_loss'][1:]))
        all_min_losses.append(min(m_hist['loss'][1:]))
        all_min_losses.append(min(m_hist['val_loss'][1:]))
        all_max_rmses.append(max(m_hist['root_mean_squared_error'][1:]))
        all_max_rmses.append(max(m_hist['val_root_mean_squared_error'][1:]))
        all_min_rmses.append(min(m_hist['root_mean_squared_error'][1:]))
        all_min_rmses.append(min(m_hist['val_root_mean_squared_error'][1:]))

    # -----------------------------------------------------------------------------------------------------------------------
    """ Plot the loss models """
    fig, axs1 = plt.subplots(2, 4)
    fig.tight_layout()
    v_padding = 0.03
    max_loss = max(all_max_losses) + (v_padding * max(all_max_losses))
    min_loss = min(all_min_losses) - (v_padding * min(all_min_losses))
    max_rmse = max(all_max_rmses) + (v_padding * max(all_max_rmses))
    min_rmse = min(all_min_rmses) - (v_padding * min(all_min_rmses))

    # Plot model_1 loss & val_loss
    axs1[0, 0].set_title(all_model_names[0])
    axs1[0, 0].plot(all_epochs[0][1:], all_losses[0][1:], 'b-', label='Training Loss')
    axs1[0, 0].plot(all_epochs[0][1:], all_val_losses[0][1:], 'r-', label='Validation Loss')
    axs1[0, 0].legend(['Training Loss', 'Validation Loss'])
    axs1[0, 0].set_ylim(min_loss, max_loss)
    # Plot model_1 rmse & val_rmse
    axs1[0, 1].set_title(all_model_names[0])
    axs1[0, 1].plot(all_epochs[0][1:], all_rmses[0][1:], 'g-', label='Training RMSE')
    axs1[0, 1].plot(all_epochs[0][1:], all_val_rmses[0][1:], 'c-', label='Validation RMSE')
    axs1[0, 1].legend(['Training RMSE', 'Validation RMSE'])
    axs1[0, 1].set_ylim(min_rmse, max_rmse)

    # Plot model_2 loss & val_loss
    axs1[0, 2].set_title(all_model_names[1])
    axs1[0, 2].plot(all_epochs[1][1:], all_losses[1][1:], 'b-', label='Training Loss')
    axs1[0, 2].plot(all_epochs[1][1:], all_val_losses[1][1:], 'r-', label='Validation Loss')
    axs1[0, 2].legend(['Training Loss', 'Validation Loss'])
    axs1[0, 2].set_ylim(min_loss, max_loss)
    # Plot model_2 rmse & val_rmse
    axs1[0, 3].set_title(all_model_names[1])
    axs1[0, 3].plot(all_epochs[1][1:], all_rmses[1][1:], 'g-', label='Training RMSE')
    axs1[0, 3].plot(all_epochs[1][1:], all_val_rmses[1][1:], 'c-', label='Validation RMSE')
    axs1[0, 3].legend(['Training RMSE', 'Validation RMSE'])
    axs1[0, 3].set_ylim(min_rmse, max_rmse)

    # Plot model_3 loss & val_loss
    axs1[1, 0].set_title(all_model_names[2])
    axs1[1, 0].plot(all_epochs[2][1:], all_losses[2][1:], 'b-', label='Training Loss')
    axs1[1, 0].plot(all_epochs[2][1:], all_val_losses[2][1:], 'r-', label='Validation Loss')
    axs1[1, 0].legend(['Training Loss', 'Validation Loss'])
    axs1[1, 0].set_ylim(min_loss, max_loss)
    # Plot model_3 rmse & val_rmse
    axs1[1, 1].set_title(all_model_names[2])
    axs1[1, 1].plot(all_epochs[2][1:], all_rmses[2][1:], 'g-', label='Training RMSE')
    axs1[1, 1].plot(all_epochs[2][1:], all_val_rmses[2][1:], 'c-', label='Validation RMSE')
    axs1[1, 1].legend(['Training RMSE', 'Validation RMSE'])
    axs1[1, 1].set_ylim(min_rmse, max_rmse)

    # Plot model_4 loss & val_loss
    axs1[1, 2].set_title(all_model_names[3])
    axs1[1, 2].plot(all_epochs[3][1:], all_losses[3][1:], 'b-', label='Training Loss')
    axs1[1, 2].plot(all_epochs[3][1:], all_val_losses[3][1:], 'r-', label='Validation Loss')
    axs1[1, 2].legend(['Training Loss', 'Validation Loss'])
    axs1[1, 2].set_ylim(min_loss, max_loss)
    # Plot model_4 rmse & val_rmse
    axs1[1, 3].set_title(all_model_names[3])
    axs1[1, 3].plot(all_epochs[3][1:], all_rmses[3][1:], 'g-', label='Training RMSE')
    axs1[1, 3].plot(all_epochs[3][1:], all_val_rmses[3][1:], 'c-', label='Validation RMSE')
    axs1[1, 3].legend(['Training RMSE', 'Validation RMSE'])
    axs1[1, 3].set_ylim(min_rmse, max_rmse)

    # Display the results
    plt.show()
else:
    print('Data not found!')
#-----------------------------------------------------------------------------------------------------------------------


