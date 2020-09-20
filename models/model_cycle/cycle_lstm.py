import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
torch.manual_seed(1015)
# define 'device' to upload tensor in gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LSTMModel_cycle(nn.Module):
    def __init__(self, input_size, hidden_size, no_time_size):

        # time model
        super(LSTMModel_cycle, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.lstm_decode = nn.LSTM(32, hidden_size, batch_first=True)

        self.dropout = nn.Dropout(p=0.3)
        
        # no time model
        self.no_time_fc = nn.Sequential(
            nn.Linear(no_time_size,6),
            nn.ReLU(inplace=True),
            nn.Linear(6,6),
            nn.ReLU(inplace=True),
            nn.Linear(6,3)
        )

        self.time_fc_d1 = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 1)
        )
        self.time_fc_d2 = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 1)
        )
        self.time_fc_d3 = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 1)
        )
        self.time_fc_d4 = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 1)
        )
        self.time_fc_d5 = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 1)
        )
        self.time_fc_d6 = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 1)
        )
        self.time_fc_d7 = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 1)
        )

        self.merge_fc = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 7),
            nn.ReLU(inplace=True),
            nn.Linear(7, 7),
            nn.ReLU(inplace=True),
            nn.Linear(7, 7)
        )

    def forward(self, x_time, x_no_time):
        # time part
        hidden = (
            torch.zeros(1, x_time.size(0), self.hidden_size).to(device),
            torch.zeros(1, x_time.size(0), self.hidden_size).to(device)
        )        
        out_time, _ = self.lstm(x_time, hidden)
        out_time = self.dropout(out_time)
        out_time, _ = self.lstm_2(out_time)
        out_time = self.dropout(out_time)
        out_time, _ = self.lstm_2(out_time)
        out_time = self.dropout(out_time)
        out_time, _ = self.lstm_2(out_time)


        tomorrow = out_time[:, -1, :]
        pred = tomorrow.view([-1, 1, self.hidden_size])
        for _ in range(6):
            tomorrow, _ = self.lstm_decode(pred)
            pred = torch.cat((pred, tomorrow[:,-1,:].view([-1, 1, self.hidden_size])), 1)

        out_1 = self.time_fc_d1(pred[:, 0, :])
        out_2 = self.time_fc_d2(pred[:, 1, :])
        out_3 = self.time_fc_d3(pred[:, 2, :])
        out_4 = self.time_fc_d4(pred[:, 3, :])
        out_5 = self.time_fc_d5(pred[:, 4, :])
        out_6 = self.time_fc_d6(pred[:, 5, :])
        out_7 = self.time_fc_d7(pred[:, 6, :])

        
        out_time = torch.cat((out_1, out_2, out_3, out_4, out_5, out_6, out_7),1)
        #print(out_time.shape)


        # no_time part
        out_no_time = self.no_time_fc(x_no_time)
        out = torch.cat((out_time, out_no_time), 1)

        out = self.merge_fc(out)
        
        return out