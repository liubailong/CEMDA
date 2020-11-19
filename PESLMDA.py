import sys,os
import importlib
import torch
import MLP
import itertools
import numpy as np
importlib.reload(sys)
from torch.autograd import Variable
#from SelfAttention import SelfAttention
from GRU import GRU_reg
import warnings
warnings.filterwarnings('ignore')
# sys.setdefaultencoding('gbk')
# sys.setdefaultencoding('utf-8')
#################################################

# # # # # PATH # # # # #
full_path=os.path.realpath(__file__)
eop=full_path.rfind(__file__)
eop=full_path.rfind(os.path.basename(__file__))
main_path=full_path[0:eop]
folder_path=full_path[0:eop]+u'DATA'
mid_result_path=folder_path+u'/5.mid result/'
tmp_path=folder_path+u'/7.result/'
#1.自定义编码，或编码器编码
# embed = [[1,1,2],[1,2,2],[1,3,2],[1,4,2],[1,5,2],[1,6,2],[1,7,2],[1,8,2],[1,9,2],[1,10,2]]
# m_code = np.loadtxt(mid_result_path + "m_coder.txt", delimiter=',')
# d_code = np.loadtxt(mid_result_path + "d_coder.txt", delimiter=',')
# #两类节点编码放在一起，通过索引可以区别。0-494是mirna，495-877是疾病的编码。
# embed =np.append(m_code,d_code,0)
# embed=torch.FloatTensor(embed)
# embed = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embed))
M_num=495
D_num=383
MD_num=5430
#1.只用编号编码，mirna和disease共878个节点，映射到64维度，注意，0-494是mirna，495-877是疾病的编码。
# embed =torch.nn.Embedding(878,100)#得到每个节点的编码
# embed = np.loadtxt(mid_result_path + "embeding.txt", delimiter=',')
# embed=torch.FloatTensor(embed)
# embed = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embed))
# print(embed)
# #测试路径，mdmd,mmddd,md。存储所有
# meta_paths=[[3,6+M_num,5,6+M_num],[6,4,7+M_num,9+M_num,5+M_num],[4,5+M_num]]
#
# # batch = [[3,6,5,6,7,1],[6,4,7,9,5,1],[4,5,8,7,1]]
# batch = list(itertools.zip_longest(*meta_paths,fillvalue=100))
# lens=[4,5,2]
# lens=torch.LongTensor(lens)
# # batch = [[3,6,4],[6,4,5],[5,7,8],[6,9,7],[7,5,1],[1,1,2]]
# batch=Variable(torch.LongTensor(batch))
# print(batch)
# # embed(batch)
# embed_batch=embed(batch)
# print(embed_batch)
#
# batch_packed = torch.nn.utils.rnn.pack_padded_sequence(embed_batch, lens,enforce_sorted=False)
# print(batch_packed)
#
# # net=lstm_reg(6, 16, output_size=5, num_layers=1)
# gru = torch.nn.GRU(100,10,2)
# # tt=torch.nn.Linear(batch_packed)
# kk=[[3,6,5,6,7,1],[6,4,7,9,5,1],[4,5,8,7,1,1]]
# # kk=Variable(torch.from_numpy(np.array(kk)))
# kk=torch.from_numpy(np.array(kk))
# print(kk)
# output = gru(batch_packed)
#
# print(output)
EPOCH=100
BATCH_SIZE=64
#数据集，是序列的集合，每个数据代表节点的索引。
train_x=[[3,6,5,6,7,1],[6,4,7,9,5,1],[4,5,8,7,1,1],[3,6,5,6,7,1],[6,4,7,9,5],[4,5,8,7,1,1],[3,6,5,6,7,1],[6,4,7,9,5,1],[4,5,8,7,1,1],[3,6,5,6,7,1],[6,4,7,9,5,1],[4,5,8,7,1,1]]
#读取6.meta path文件夹的所有文件名
import os
meta_paths_filePath = 'DATA/6.meta path/'
fileList = os.listdir(meta_paths_filePath)
print(fileList)
# for file in fileList:
import pandas as pd
from sklearn.externals import joblib
train_x=[]
for file in fileList:
    meta_paths_pd = pd.read_csv(meta_paths_filePath + file, nrows=20000, header=None)#每个文件暂时取100个
    #这里的元路径索引号m和d都是从0开始的

    #还差mdd的编号要从495开始，因此要每个d的编号增加495
    index_update=[]
    i=0
    for ch in file[0:-4].split('_'):#第一次出现文件‘_’的位置，为了找到d
        if ch == 'm':
            meta_paths_pd[i]=meta_paths_pd[i]+0#如果是m，则索引号不变，增加0
        elif (ch == 'd'):
            meta_paths_pd[i]=meta_paths_pd[i]+495#如果是d，则索引号增加495
        i=i+1
    # for i in range(len(meta_paths_list)):
    #     meta_paths_list[i] = meta_paths_list[i] + index_update
    print("meta_paths_pd")
    print(meta_paths_pd)
    meta_paths_list = meta_paths_pd.values.tolist()  # 转成列表，
    train_x = train_x + meta_paths_list
# print(len(train_x))
print("train_x")
print(train_x)
#因此，每个节点还有编码，要根据节点索引喂入的是编码
#embed就存入了每个节点的编码，第1行对应0号节点的编码，一次类推。
#这里原始的embed是np或list，因此需要转成torch能够解析的编码格式类型。可以根据batch自动并行取的多个样本的对应节点
# embed=[[1,2,3,4],[1,1,1,1],[1,2,3,4],[1,1,1,1],[1,2,3,4],[1,1,1,1],[1,2,3,4],[1,1,1,1],[1,2,3,4],[1,1,1,1],[1,2,3,4],[1,1,1,1]]
'''读取每个节点编码  开始'''
embed = np.loadtxt(mid_result_path + "embeding.txt", delimiter=',')
print("embed")
print(embed)
embed=torch.FloatTensor(embed)
embed = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embed))
'''读取每个节点编码  结束'''

'''读取md节点对已知关系 开始'''
m_d = np.loadtxt(mid_result_path + "m_d.txt", delimiter=',')#行号是m，列号是d,注意，疾病的编号从0开始了
print("m_d")
print(m_d)
'''读取md节点对已知关系 结束'''
# lens=[]#记录每个样本的长度，应对变长GRU
# for item in train_x:
#     lens.append(len(item))
'''确立模型'''
model_gru=GRU_reg(100,100,2)#GRU
optimizer_gru = torch.optim.Adam(model_gru.parameters(), lr=1e-2)#gru优化器
model_mlp = MLP.MLP()#mlp
optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=1e-2)#mlp优化器
loss_func_mlp = torch.nn.BCELoss()

'''确立模型结束'''
'''数据训练开始'''

for epoch in range(EPOCH):
    for i in range(0,len(train_x),BATCH_SIZE):
        var_x = train_x[i:i+BATCH_SIZE]#多个元路径编号序列
        lens = [len(x) for x in var_x]#每个序列的长度
        lens = torch.LongTensor(lens)
        # var_x = torch.FloatTensor(var_x)
        batch = list(itertools.zip_longest(*var_x, fillvalue=0))#路线按照batchsize同时喂入gru。需要对应节点分离
        batch = Variable(torch.LongTensor(batch))#改变格式

        embed_batch = embed(batch)#取出节点对应的编码
        batch_packed = torch.nn.utils.rnn.pack_padded_sequence(embed_batch, lens, enforce_sorted=False)
        # model_gru = torch.nn.GRU(100, 100, 2)#GRU的输入维度[也就是每个节点的编码]，隐含层的维度，层数

        # h0 = torch.rand(100, 2, 10)
        # c0 = torch.rand(100, 2, 10)
        '''计算GRU的输出'''
        # output,hidden= model_gru(batch_packed,None)#注意输出接收的变量不可以省略。否则解析不对。
        output_gru_att = model_gru(batch_packed)
        #解析输出的batch中每个序列中节点对应的输出值。如一个batch有节点名称为1234和4568两个四节点的序列【即batch_size为2】，
        # 1234四个节点序列，经过gru分别有四个输出，同时5678也有4个输出，output是同时并行喂入两个序列得到的输出，
        # 由于序列是按照顺序喂入的，同时1和5并行喂入，以此类推
        # 因此output分别是15263748节点经过gru对应的输出，我们要得出的是1234经过gru对应的输出序列，和5678的输出序列，才有意义，
        # pad_packed_sequence就可以解决
        # out_pad, out_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)#GRU的输出
        # att_layer = SelfAttntion(100)
        # out_att_gru = att_layer(out_pad)#经过attention后的输出，综合为所有节点的输出。

        '''计算MLP的输出'''
        #Batch中的所有路径，首尾的节点组合编码，输入到MLP
        #获取训练样本中所有路径的首尾节点
        com_embeding = []#保存一个batch所有路径首尾节点对的组合编码
        pairs_in_batch = []#记录所有不同的首尾节点对
        index_of_pair = []#记录每个节点对的标识，如[[1,3],[1,3],[2,3]]的节点对序列，构成[0,0,1],代表前连个路径首尾节点一致，第三个不同。
        model_mlp = MLP.MLP()
        expected_out_mlp = []#存储本batch内所有路径首尾节点的md期望关系输出
        i=0
        for item in var_x:
            '''计算meta中一对pair的编码    开始'''
            first_node = item[0]#第一个节点编号
            end_node = item[-1]#最后一个节点编号,肯定是d，起始编号为495
            expected_out_mlp.append(m_d[first_node][end_node-M_num])#从m_d中获取
            # if [first_node,end_node] in pairs_in_batch:
            # else:
            #     pairs_in_batch.append([first_node,end_node])
            dot = embed.weight[first_node] * embed.weight[end_node]
            diff = embed.weight[first_node] - embed.weight[end_node]
            com_embeding.append(np.hstack((embed.weight[first_node], embed.weight[end_node], diff, dot)).tolist())#组合编码
            '''计算meta中一对pair的编码    结束'''
            # '''一对pair的编码输入到MLP    开始'''
            # com_embeding_mlp = Variable(torch.FloatTensor(com_embeding))
            # out_mlp = model_mlp(com_embeding_mlp)
            # '''一对pair的编码输入到MLP    结束'''
        #for end 得到了batch内的所有首尾节点组合编码
        '''所有的首尾节点对输入到MLP并输出---------》开始'''

        batch_mlp = Variable(torch.FloatTensor(com_embeding),requires_grad=True)  # 改变成torch的格式
        out_mlp_embed_pair ,out_mlp = model_mlp(batch_mlp)#MLP中一个batch的节点对的编码输出,以及是否有关联的输出
        #期望输出
        '''所有的首尾节点对输入到MLP并输出---------》结束'''

        '''GRU训练开始'''
        loss_path_and_pair= 0
        print("++++loss_path_and_pair")
        loss_temp = out_mlp_embed_pair.detach().numpy()*output_gru_att.detach().numpy()#batch的矩阵，对应元素相乘，每行是一对嵌入元路径的loss
        loss_path_and_pair = np.sum(loss_temp)
        loss_path_and_pair= torch.sigmoid(torch.tensor(loss_path_and_pair,requires_grad=True))#每个batch所有loss求和Variable(loss_path_and_pair,requires_grad=True)
        print(loss_path_and_pair)
        loss_path_and_pair.backward()
        optimizer_gru.step()
        optimizer_gru.zero_grad()
        # for index in range(BATCH_SIZE):
        #     loss_path_and_pair +=  -torch.nn.Sigmoid(np.dot(out_mlp_embed_pair[index].detach().numpy(),output_gru_att.detach().numpy())) #将路径的信息加入到节点对上,用sigmoid近似


        '''GRU训练结束'''


        '''MLP训练开始'''
        optimizer_mlp.zero_grad()
        expected_out_mlp = Variable(torch.FloatTensor(expected_out_mlp))#torch.tensor(np.array(expected_out_mlp), requires_grad=True)
        loss_mlp = loss_func_mlp(out_mlp, expected_out_mlp)
        loss_mlp.backward()
        optimizer_mlp.step()
        print("-----loss_mlp")
        print(loss_mlp)
'''数据训练结束'''
'''保存两个模型开始'''

# 创建文件目录
# dirs = 'testModel1'
# if not os.path.exists(dirs):
#     os.makedirs(dirs)
#
# # 保存模型
# joblib.dump(loss_mlp, dirs + '/mlp.pkl')
'''保存两个模型结束'''




'''模型预测最后结果开始'''
import xlrd
# md_table = xlrd.open_workbook(folder_path+u'/1.miRNA-disease associations/miRNA-disease_index.xlsx')
# md_sheet = md_table.sheet_by_index(0)
# MD_num = md_sheet.nrows
# A=[['0'],['0']]
m_d_expect=m_d
# for i in range(MD_num):
#     first_node = int(md_sheet.cell_value(rowx=i, colx=0)) - 1  # 名称编号从1开始，写入到矩阵就要从0下标开始。
#     end_node = int(md_sheet.cell_value(rowx=i, colx=1)) - 1+495
#     A[0].append(int(first_node))
#     A[1].append(int(end_node-495))
for i in range(M_num):
    for j in range(D_num):
        first_node=i
        end_node=j+495
        expect = []
        expect.append(m_d[first_node][end_node - M_num])  # 从m_d中获取
        dot = embed.weight[first_node] * embed.weight[end_node]
        diff = embed.weight[first_node] - embed.weight[end_node]
        comEmbed = []
        comEmbed.append(np.hstack((embed.weight[first_node], embed.weight[end_node], diff, dot)).tolist())  # 组合编码
        batch_mlp = Variable(torch.FloatTensor(comEmbed), requires_grad=True)  # 改变成torch的格式

        out_mlp_embed_pair, out_mlp = model_mlp(batch_mlp)  # MLP中一个batch的节点对的编码输出,以及是否有关联的输出
        expect = Variable(torch.FloatTensor(expect))  # torch.tensor(np.array(expected_out_mlp), requires_grad=True)
        print("expected_out_mlp",i)
        print(expect)
        loss_mlp = loss_func_mlp(out_mlp, expect)
        print("预测结果loss_mlp",i)
        print(loss_mlp)
        m_d_expect[i, j] = loss_mlp
print("m_d_expect")
np.savetxt(tmp_path + "m_d_expect.txt", m_d_expect, delimiter=',', fmt='%.5f')
print(m_d_expect)
'''模型预测最后结果结束'''