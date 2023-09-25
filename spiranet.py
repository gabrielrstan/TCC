from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpiraNet(nn.Module):
  def __init__(self):
    super(SpiraNet, self).__init__()
    self.num_feature = Config.Audio.num_mels
    convs = [
      # cnn1
      nn.Conv2d(1, 32, kernel_size=(1,7), dilation=(1, 2)),
      nn.GELU(), nn.MaxPool2d(kernel_size=(1,2)), nn.Dropout(p=0.7),
      # cnn2
      nn.Conv2d(32, 16, kernel_size=(1, 5), dilation=(1, 2)),
      nn.GELU(), nn.MaxPool2d(kernel_size=(1,2)), nn.Dropout(p=0.7),
      # cnn3
      nn.Conv2d(16, 8, kernel_size=(1, 3), dilation=(1, 2)),
      nn.GELU(), nn.MaxPool2d(kernel_size=(1,2)), nn.Dropout(p=0.7),
      # cnn4
      nn.Conv2d(8, 4, kernel_size=(1, 2), dilation=(1, 1)),
      nn.GELU(), nn.Dropout(p=0.7)

    ]
    self.conv = nn.Sequential(*convs)

    # fc1
    toy_input = torch.zeros(1, 1, self.num_feature, Config.Train.max_seq_len)
    toy_activation_shape = self.conv(toy_input).shape
    fc1_input_dim = toy_activation_shape[1] * toy_activation_shape[2] * toy_activation_shape[3]
    self.fc1 = nn.Linear(fc1_input_dim, Config.Model.fc1_dim)

    # fc2 and activation functions
    self.fc2 = nn.Linear(Config.Model.fc1_dim, Config.Model.fc2_dim)
    self.gelu = nn.GELU()
    self.sigmoid = nn.Sigmoid()
    self.dropout = nn.Dropout(p=0.7)

  def forward(self, x):
   # print(x.shape)
    x = self.conv(x)
   # print(x.shape)
    x = x.view(x.size(0), -1)
   # print(x.shape)
    x = self.fc1(x)
   # print(x.shape)
    x = self.gelu(x)
    x = self.dropout(x)
    x = self.fc2(x)
   # print(x.shape)
    x = self.sigmoid(x)
    return x
