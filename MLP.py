# 建立一个四层感知机网络
import torch
import torch.nn.functional as F
class MLP(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self):
        super(MLP, self).__init__()  #
        # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = torch.nn.Linear(400, 300)  # 第一个隐含层，输入400
        self.drop1 = torch.nn.Dropout(0.6)#参数可调
        self.fc2 = torch.nn.Linear(300, 200)  # 第二个隐含层
        self.drop2 = torch.nn.Dropout(0.6)
        self.fc3 = torch.nn.Linear(200, 100)  # 输出层,没有dropout
        self.fc4 = torch.nn.Linear(100, 50)#从这层开始就映射到是否成对了，与md矩阵对应
        self.fc5 = torch.nn.Linear(50, 1)

    def forward(self, din):
        # 前向传播， 输入值：din, 返回值 dout
        dout = F.relu(self.drop1(self.fc1(din)))  #
        dout = F.relu(self.drop2(self.fc2(dout)))
        # dout = F.softmax(self.fc3(dout), dim=1)  #
        dout1 = F.relu(self.fc3(dout))
        dout2 = F.relu(self.fc4(dout1))
        dout2 = torch.sigmoid(self.fc5(dout2))
        return dout1,dout2#一个是节点对的编码，一个是节点对是否有连接，这里是必须有连接，因为元路径中获取的节点对

