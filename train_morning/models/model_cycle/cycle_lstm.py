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
        
        self.lstm_week1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm_week2 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm_week3 = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.time_fc_gru = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )

        self.time_fc_week1 = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
        self.time_fc_week2 = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
        self.time_fc_week3 = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )

        self.week_merge = nn.Sequential(
            nn.Linear(3, 1)
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
        self.merge_fc_lstm = nn.Sequential(
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

        self.merge_fc_gru = nn.Sequential(
            nn.Linear(7, 14),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(14, 14),
            nn.ReLU(inplace=True),
            nn.Linear(14,7)
        )

    def forward(self, x_time, x_no_time):
        #print(x_time[:,:,-1].shape)
        # time part
        # x_time의 12개 채널 정보
        # 'card_use', 'holiday', 'day_corona',
        # 'ondo', 'subdo', 'rain_snow',
        # 'dayofyear_sin', 'dayofyear_cos', 'weekday_sin', 'weekday_cos',
        # 'flow_trend', flow_cycle'
        # time part
        #x_time_2 = x_time[:,:,10:]
        #x_time = torch.cat((x_time_1,x_time_2),2)

        #out_time = self.dropout(out_time)
        #out_time_lstm, _ = self.lstm(x_time)\
        out_time_lstm_week1, _ = self.lstm(x_time[:,   :7 , :])
        out_time_lstm_week2, _ = self.lstm(x_time[:, 7 :14, :])
        out_time_lstm_week3, _ = self.lstm(x_time[:, 14:  , :])

        #out_time의 shape : [batch, input_window_size, hidden_size]
        out_time_lstm_week1 = self.time_fc_week1(out_time_lstm_week1)
        out_time_lstm_week2 = self.time_fc_week2(out_time_lstm_week2)
        out_time_lstm_week3 = self.time_fc_week3(out_time_lstm_week3)

        out_time = self.week_merge(torch.cat((out_time_lstm_week1.view(-1,7,1),
                                                 out_time_lstm_week2.view(-1,7,1), 
                                                 out_time_lstm_week3.view(-1,7,1)), 2))


        # no_time part
        out_no_time = self.no_time_fc(x_no_time)



        # merge part
        out_no_time = out_no_time.view(-1,1)
        out_time = out_time.view(-1,7)
        out = out_time * out_no_time
   
        out = out.view(-1,7)     

        #out = self.merge_fc_lstm(out)
         
        #return out_time.view(-1,7) 
        return out