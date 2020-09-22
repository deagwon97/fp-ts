import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import gc
import matplotlib.pyplot as plt
import pickle
import torch
from utils.preprocess_utils import *

# load data
time_data = pd.read_csv('data/original/time_data.txt', sep  = ' ') 
time_data = df2npy(time_data)

# 아침 점심 저녁 분리
morning_data = time_data[:,2,:,:]
lunch_data = time_data[:,1,:,:]
evening_data = time_data[:,0,:,:]

# split time data
# train_validation_test
# scaling
# make intput, output window
morning_data, train_valid_test_loc_index, m_time_scaler = split_train_valid_test(morning_data)
lunch_data, _, l_time_scaler = split_train_valid_test(lunch_data)
evening_data, _, e_time_scaler = split_train_valid_test(evening_data)

# split notime data
# notime_train, notime_valid, notime_test = notime
nontime_data = pd.read_csv('data/original/nontime_data.txt', sep = ' ')
notime, no_time_scaler = split_notime_data(nontime_data, train_valid_test_loc_index)

print('\n morning')
m_train_time, m_train_notime, m_train_y = make_data(morning_data[:2], notime)
m_valid_time, m_valid_notime, m_valid_y = make_data(morning_data[2], notime)
m_test_time, m_test_notime, m_test_y = make_data(morning_data[3], notime)

print('\n lunch')
l_train_time, l_train_notime, l_train_y = make_data(lunch_data[:2], notime)
l_valid_time, l_valid_notime, l_valid_y = make_data(lunch_data[2], notime)
l_test_time, l_test_notime, l_test_y = make_data(lunch_data[3], notime)

print('\n evening')
e_train_time, e_train_notime, e_train_y = make_data(evening_data[:2], notime)
e_valid_time, e_valid_notime, e_valid_y = make_data(evening_data[2], notime)
e_test_time, e_test_notime, e_test_y = make_data(evening_data[3], notime)


m_data_list = [m_train_time[:,:,2:], m_train_notime, m_train_y, 
                m_valid_time[:,:,2:], m_valid_notime, m_valid_y,
                m_test_time[:,:,2:], m_test_notime, m_test_y]

l_data_list = [l_train_time[:,:,2:], l_train_notime, l_train_y, 
                l_valid_time[:,:,2:], l_valid_notime, l_valid_y,
                l_test_time[:,:,2:], l_test_notime, l_test_y]

e_data_list = [e_train_time[:,:,2:], e_train_notime, e_train_y, 
                e_valid_time[:,:,2:], e_valid_notime, e_valid_y,
                e_test_time[:,:,2:], e_test_notime, e_test_y]    

data_list =  [np.concatenate([m_train_time[:,:,2:], l_train_time[:,:,2:], e_train_time[:,:,2:]]),
                np.concatenate([m_train_notime, l_train_notime, e_train_notime]),
                np.concatenate([m_train_y, l_train_y, e_train_y]),
                np.concatenate([m_valid_time[:,:,2:], l_valid_time[:,:,2:], e_valid_time[:,:,2:]]),
                np.concatenate([m_valid_notime, l_valid_notime, e_valid_notime]),
                np.concatenate([m_valid_y, l_valid_y, e_valid_y]),
                np.concatenate([m_test_time[:,:,2:], l_test_time[:,:,2:], e_test_time[:,:,2:]]),
                np.concatenate([m_test_notime, l_test_notime, e_test_notime]),
                np.concatenate([m_test_y, l_test_y, e_test_y])]


path = 'data/preprocess/'
with open(path + 'm_data_list.pkl', 'wb') as f:
    pickle.dump(m_data_list, f)
with open(path + 'l_data_list.pkl', 'wb') as f:
    pickle.dump(l_data_list, f)
with open(path + 'e_data_list.pkl', 'wb') as f:
    pickle.dump(e_data_list, f)

with open(path +'full_data_list.pkl', 'wb') as f:
    pickle.dump(data_list, f)


scalers = [m_time_scaler, l_time_scaler, e_time_scaler, no_time_scaler]
with open(path + 'scalers.pkl', 'wb') as f:
    pickle.dump(scalers,f)
