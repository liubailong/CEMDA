import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from torch import nn
from torch.autograd import Variable
from SelfAttention import  SelfAttention
# 定义模型
'''
input_size – 输入的特征维度
hidden_size – 隐状态的特征维度
num_layers – 层数（和时序展开要区分开）
'''


class GRU_reg(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(GRU_reg, self).__init__()

        self.gru = nn.GRU(input_size,
                          hidden_size,  # rnn hidden unit
                          num_layers,  # 有几层 RNN layers
                          batch_first=True, )

        # self.out = nn.Linear(hidden_size, output_size)
        self.attention = SelfAttention(hidden_size)

    def forward(self, x):
        gru_out, gru_hide = self.gru(x, None)
        # 经过attention后的输出，综合为所有节点的输出。
        out_pad, out_len = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)  # GRU的输出

        out,att = self.attention(out_pad)
        out=Variable(out,requires_grad=True)
        return out

