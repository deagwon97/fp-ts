
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
torch.manual_seed(1015)
import random

from utils.preprocess_utils import *

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
# define 'device' to upload tensor in gpu
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def plot_predict(train_x, train_y, train_pred,
                    valid_x, valid_y, valid_pred, plot_numbers = 10):
    for i in range(plot_numbers):

        i = random.randint(1,len(train_x)-1)
        plt.figure(figsize = (15, 3))
        
        plt.subplot(1,2,1)
        
        plt.plot(np.arange(21), train_x[i,:],   # m_train_time.cpu().detach().numpy()[i,:,-2],
                marker = 'o', color = 'black', label = 'True_input')# 검정
        plt.plot(np.arange(21,28), train_y[i],    #m_train_y[:,:,0].cpu().detach().numpy()[i],
                marker = 'o', color = '#BD434D', label = 'True_output', alpha = 0.5) #빨강
        plt.plot(np.arange(21,28),train_pred[i],    #.cpu().detach().numpy()[i],
                color = '#5A87B9', label = 'Predict', marker = 'x', ls = '--', alpha = 0.5) # 파랑
        start_day = train_idx2day(i)
        print(start_day)
        plt.xticks(ticks = np.arange(28), labels = make_xticks(start_day))
        plt.title(f'{idx2dong(i, len(train_x))}(Train)')
        plt.legend()

        i = random.randint(1,len(valid_x)-1)
        plt.subplot(1,2,2)
        plt.plot(np.arange(21), valid_x[i,:],
                marker = 'o', color = 'black', label = 'True_input')
        plt.plot(np.arange(22,29), valid_y[i],
                marker = 'o', color = '#BD434D', label = 'True_output', alpha = 0.5)
        plt.plot(np.arange(22,29), valid_pred[i],
                color = '#5A87B9', label = 'Predict', marker = 'x', ls = '--', alpha = 0.5)
        plt.title('validation')
        plt.legend()
        plt.show()


def plot_test_predict(train_x, train_y, train_pred):
    plt.figure(figsize = (15, 10))
    for i in range(4):
        plt.subplot(2,2,i+1)
        i = random.randint(1,21) + i*11
        plt.plot(np.arange(21), train_x[i,:],   # m_train_time.cpu().detach().numpy()[i,:,-2],
                marker = 'o', color = 'black', label = 'True_input')
        plt.plot(np.arange(22,29), train_y[i],    #m_train_y[:,:,0].cpu().detach().numpy()[i],
                marker = 'o', color = '#BD434D', label = 'True_output', alpha = 0.5)
        plt.plot(np.arange(22,29),train_pred[i],    #.cpu().detach().numpy()[i],
                color = '#5A87B9', label = 'Predict', marker = 'x', ls = '--', alpha = 0.5)
        plt.title('test')
        plt.legend()
    plt.show()