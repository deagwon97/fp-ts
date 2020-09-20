
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
        #self.lstm_up = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.time_fc = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
        self.dropout = nn.Dropout(p=0.2)
        # no time model
        self.no_time_fc = nn.Sequential(
            nn.Linear(no_time_size,16),
            nn.ReLU(inplace=True),
            nn.Linear(16,16),
            nn.ReLU(inplace=True),
            nn.Linear(16,8),
            nn.ReLU(inplace=True),
            nn.Linear(8,1)
        )

        # merge model
        self.merge_fc = nn.Sequential(
            nn.Linear(7, 14),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(14, 14),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(14, 14),
            nn.ReLU(inplace=True),
            nn.Linear(14,7)
        )


    def forward(self, x_time, x_no_time):
        # time part
        # x_time의 12개 채널 정보
        # 'card_use', 'holiday', 'day_corona',
        # 'ondo', 'subdo', 'rain_snow',
        # 'dayofyear_sin', 'dayofyear_cos', 'weekday_sin', 'weekday_cos',
        # 'flow_trend', flow_cycle'
        # time part
        hidden = (
            torch.zeros(1, x_time.size(0), self.hidden_size).to(device),
            torch.zeros(1, x_time.size(0), self.hidden_size).to(device)
        )
        #x_time_1 = x_time[:,:,:8]
        #x_time_2 = x_time[:,:,10:]
        #x_time = torch.cat((x_time_1,x_time_2),2)

        out_time, _ = self.lstm(x_time, hidden)
        #out_time = self.dropout(out_time)
        #out_time, _ = self.lstm_up(out_time, hidden)
        #out_time의 shape : [batch, input_window_size, hidden_size]
        out_time = self.time_fc(out_time[:, -7:, :])#

        # no_time part
        out_no_time = self.no_time_fc(x_no_time)

        # merge part
        out_no_time = out_no_time.view((-1,1,1))
        out = out_time * out_no_time
        out = out.view(-1,7)      
        out = self.merge_fc(out)
        
        #return out_time.view(-1,7) 
        return out.view(-1,7) 

