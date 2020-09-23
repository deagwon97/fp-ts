
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
# define 'device' to upload tensor in gpu
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def plot_predict(train_x, train_y, train_pred,
                    valid_x, valid_y, valid_pred, plot_numbers = 10):
    for i in range(plot_numbers):

        i = random.randint(1,21) + i*11
        plt.figure(figsize = (15, 3))
        
        plt.subplot(1,2,1)
        import numpy as np
        plt.plot(np.arange(21), train_x[i,:],   # m_train_time.cpu().detach().numpy()[i,:,-2],
                marker = 'o', color = 'black', label = 'True_input')
        plt.plot(np.arange(22,29), train_y[i],    #m_train_y[:,:,0].cpu().detach().numpy()[i],
                marker = 'o', color = 'red', label = 'True_output', alpha = 0.5)
        plt.plot(np.arange(22,29),train_pred[i],    #.cpu().detach().numpy()[i],
                color = 'blue', label = 'Predict', marker = 'x', ls = '--', alpha = 0.5)
        plt.title('train')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(np.arange(21), valid_x[i,:],
                marker = 'o', color = 'black', label = 'True_input')
        plt.plot(np.arange(22,29), valid_y[i],
                marker = 'o', color = 'red', label = 'True_output', alpha = 0.5)
        plt.plot(np.arange(22,29), valid_pred[i],
                color = 'blue', label = 'Predict', marker = 'x', ls = '--', alpha = 0.5)
        plt.title('validation')
        plt.legend()
        plt.show()


def plot_test_predict(train_x, train_y, train_pred):
    plt.figure(figsize = (10, 10))
    for i in range(4):
        i = random.randint(1,21) + i*11
        plt.subplot(2,2,i+1)
        import numpy as np
        plt.plot(np.arange(21), train_x[i,:],   # m_train_time.cpu().detach().numpy()[i,:,-2],
                marker = 'o', color = 'black', label = 'True_input')
        plt.plot(np.arange(22,29), train_y[i],    #m_train_y[:,:,0].cpu().detach().numpy()[i],
                marker = 'o', color = 'red', label = 'True_output', alpha = 0.5)
        plt.plot(np.arange(22,29),train_pred[i],    #.cpu().detach().numpy()[i],
                color = 'blue', label = 'Predict', marker = 'x', ls = '--', alpha = 0.5)
        plt.title('train')
        plt.legend()
        plt.show()
