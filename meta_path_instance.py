'''生成meta path instance'''
import numpy as np
'''
M-D【M-(2^0)-D】
M-M-D  M-D-D【M-(2^1)-D】
M-M-M-D M-M-D-D M-D-M-D M-D-D-D【M-(2^2)-D】
M-(2^K)-D,这里的K是指中间节点数量，决定了此类长度元路径的数量。

'''
import sys,os
import importlib
importlib.reload(sys)
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
###########################################################
'''获取单点连接线路,中间结果路径'''

def index_of_one():
    m_m_1 = np.loadtxt(mid_result_path + "m_m_1.txt", delimiter=',')
    m_m_list = np.argwhere(m_m_1 == 1)#取m-m的连接对应的下标索引对

    d_d_1 = np.loadtxt(mid_result_path + "d_d_1.txt", delimiter=',')
    d_d_list = np.argwhere(d_d_1 == 1)  # 取d-d的连接对应的下标索引对

    m_d = np.loadtxt(mid_result_path + "m_d.txt", delimiter=',')
    m_d_list = np.argwhere(m_d == 1)  # 取m-d的连接对应的下标索引对
    d_m = np.transpose(m_d)
    d_m_list = np.argwhere(d_m == 1)  # 取m-d的连接对应的下标索引对
    # print(m_d_list)
    # print(d_m_list)
    return m_m_list,d_d_list,m_d_list,d_m_list,m_m_1,d_d_1,m_d,d_m
###########################################################
'''2L,MMD,MDD,DMD，DDD'''
def M_M_D(m_m,m_d):#
    mmd_list=[]
    # for mi,mj in m_m_list:#两个疾病之间的路线
    #     # print(mi,mj)
    #     for m,d in m_d_list:
    #         if mj==m:
    #             m_m_d_list.append([mi, mj, d])
    #             # print(m_m_d_list)
    # print(m_m_d_list)
    mmd=np.dot(m_m , m_d)#m到达某个d的路线条数矩阵
    mmd_num_index = np.argwhere(mmd >= 1)#找到所有能通过mmd到达的元路径实例的索引对
    # print(len(mmd_num_index))
    for mi,dj in mmd_num_index:
        line_mi=m_m[mi]
        mi_index=np.argwhere(line_mi >= 1)#与mi相连的所有mj索引
        # print(mi_index)
        for [mm] in mi_index:
            if m_d[mm,dj] ==1 :
                mmd_list.append([mi,mm,dj])
                # print(m_m_d_list)
    print(mmd_list)
    return mmd,mmd_list

def M_D_D(m_d,d_d):#
    mdd_list=[]
    mdd = np.dot(m_d, d_d)  # m到达某个d的路线条数矩阵
    mmd_num_index = np.argwhere(mdd >= 1)  # 找到所有能通过mdd到达的元路径实例的索引对
    # print(len(mmd_num_index))
    for mi, dj in mmd_num_index:
        line_mi = m_d[mi]
        dj_index = np.argwhere(line_mi >= 1)  # 与mi相连的所有dj索引
        # print(mi_index)
        for [dd] in dj_index:
            if d_d[dd, dj] == 1:
                mdd_list.append([mi, dd, dj])
                # print(m_m_d_list)
    print(mdd_list)
    return mdd, mdd_list

'''上述两个函数总结。写成通用函数
node_left,node_mid,node_right,元路径的组成，需要中间结点的过度，因此要有
两个矩阵，left_to_middle，左节点与中间结点邻接矩阵，中间节点与右节点连接矩阵
如：MDD是MD与DD邻接矩阵构成参数，MDDD是MD与DDD构成的邻接矩阵为参数。
注意：函数名中N代表起点可能是M也可能是D
但是：：：这个方法有bug，例如依赖DDD，无法找到所有的DDD实例，仍然只能发现di到dj的通路，
      ********此函数只对两跳元路径求解最快*******
'''
def N_X_D(left_to_middle,middle_to_right):#同一个函数，代入不同的矩阵，得到不同的元路径
    nxd_list=[]
    nxd = np.dot(left_to_middle, middle_to_right)  # N节点到达某个d的路线条数矩阵，元素可能大于等于1
    nxd_num_index = np.argwhere(nxd >= 1)  # 找到所有能通过nxd到达的元路径实例的索引对
    # print(len(mmd_num_index))
    for mi, dj in nxd_num_index:
        line_mi = left_to_middle[mi]#找到左矩阵的第mi行，
        mid_index = np.argwhere(line_mi >= 1)  # 与mi相连的所有中间节点的索引
        # print(mi_index)
        for [mid] in mid_index:
            if middle_to_right[mid, dj] == 1:
                nxd_list.append([mi, mid, dj])
                # print(m_m_d_list)
    print(nxd_list)
    nxd1=np.int64(nxd > 0)#nxd统计从n到d有多少个元路径，如果元路径增加长度，只需要有元路径即可，因此置1。
    # 例如mmd记录从某个m到某个d的元路径条数，当需要统计mmmd只要统计mm是否有链接，加上已知mmd就可以知道mmmmd元路径所有序列
    return nxd1, nxd_list

'''
mmdd元路径实例是由mm和所有的mdd元路径实例组合而成
这个方法比较慢，暂时不用

'''
def N_X_D1(left_to_middle,middle_to_right,middle_to_right_list):#同一个函数，代入不同的矩阵，得到不同的元路径
    nxd_list=[]
    nxd = np.dot(left_to_middle, middle_to_right)  # N节点到达某个d的路线条数矩阵，元素可能大于等于1
    nxd_num_index = np.argwhere(nxd >= 1)  # 找到所有能通过nxd到达的元路径实例的索引对
    # print(len(mmd_num_index))
    for mi, dj in nxd_num_index:
        line_mi = left_to_middle[mi]#找到左矩阵的第mi行，
        mid_index = np.argwhere(line_mi >= 1)  # 与mi相连的所有中间节点的索引
        # print(mi_index)
        for [mid] in mid_index:
            right_meta_path_start = [x[0] for x in middle_to_right_list]  # 取路径右边部分的所有起点
            right_meta_path_start_index=np.argwhere(right_meta_path_start == mid)#取起点与中间结点相同的已知列表索引
            for [yy] in right_meta_path_start_index:
                nxd_list.append(np.insert(middle_to_right_list[yy],0,mi).tolist())#拼接后加入列表
                # print(m_m_d_list)
        print(nxd_list)

    nxd1=np.int64(nxd > 0)#nxd统计从n到d有多少个元路径，如果元路径增加长度，只需要有元路径即可，因此置1。
    # 例如mmd记录从某个m到某个d的元路径条数，当需要统计mmmd只要统计mm是否有链接，加上已知mmd就可以知道mmmmd元路径所有序列
    return nxd1, nxd_list


'''验证后，这个方法最快'''
'''列出所有的MXD的元路径实例列表
first_to_second_list:元路径中，第一个节点到第二个节点的路径列表
second_to_right_list:元路径中，第二个节点到最后一个节点的路径列表
如，mmmdd，则first_to_second_list记录了所有mm的路径，second_to_right_list记录了所有mmdd的路径
然后拼接组合，成为mmmdd。
'''
def M_X_D_list(first_to_second_list,second_to_right_list):#
    mxd_list=[]
    for mi,xj in first_to_second_list:#m到x节点的路径
        # print(mi,mj)
        right_start = [x[0] for x in second_to_right_list]#找到second_to_right_list所有元路径的第一个节点
        right_start_index = np.argwhere(right_start == xj)#去第一个节点等于带求路径的第二个节点的元素
        for [yy] in right_start_index:
            mxd_list.append(np.insert(second_to_right_list[yy],0,mi).tolist())#组合一条路径
            # print(np.insert(m_d_list[yy],0,mi))
        # mxd_list_numpy = np.array(mxd_list)
        # print(mxd_list_numpy)
    mxd_list_numpy = np.array(mxd_list)
    # print(mxd_list_numpy)
    return mxd_list_numpy
###########################################################
''' MAIN '''
if __name__ == '__main__':
    mm_list, dd_list, md_list, dm_list,m_m,d_d,m_d,d_m=index_of_one()
    print(len(mm_list), len(dd_list), len(md_list), len(dm_list))
    np.savetxt(mid_result_path + "md_list.txt", md_list, delimiter=',', fmt='%d')

    #2L
    mmd, mmd_list = N_X_D(m_m, m_d)
    mdd, mdd_list = N_X_D(m_d, d_d)
    dmd, dmd_list = N_X_D(d_m, m_d)
    ddd, ddd_list = N_X_D(d_d, d_d)
    np.savetxt(mid_result_path + "mmd_list.txt", mmd_list, delimiter=',', fmt='%d')
    np.savetxt(mid_result_path + "mdd_list.txt", mdd_list, delimiter=',', fmt='%d')
    #3L
    mmmd_list = M_X_D_list(mm_list, mmd_list)
    mmdd_list = M_X_D_list(mm_list, mdd_list)
    mdmd_list = M_X_D_list(md_list, dmd_list)
    mddd_list = M_X_D_list(md_list, ddd_list)
    np.savetxt(mid_result_path + "mmmd_list.txt", mmmd_list, delimiter=',', fmt='%d')
    np.savetxt(mid_result_path + "mmdd_list.txt", mmdd_list, delimiter=',', fmt='%d')
    np.savetxt(mid_result_path + "mdmd_list.txt", mdmd_list, delimiter=',', fmt='%d')
    np.savetxt(mid_result_path + "mddd_list.txt", mddd_list, delimiter=',', fmt='%d')

    dmmd_list = M_X_D_list(dm_list, mmd_list)
    dmdd_list = M_X_D_list(dm_list, mdd_list)
    ddmd_list = M_X_D_list(dd_list, dmd_list)
    dddd_list = M_X_D_list(dd_list, ddd_list)
    #4L
    mmmmd_list = M_X_D_list(mm_list, mmmd_list)
    mmmdd_list = M_X_D_list(mm_list, mmdd_list)
    mmdmd_list = M_X_D_list(mm_list, mdmd_list)
    mmddd_list = M_X_D_list(mm_list, mddd_list)
    mdmmd_list = M_X_D_list(md_list, dmmd_list)
    mdmdd_list = M_X_D_list(md_list, dmdd_list)
    mddmd_list = M_X_D_list(md_list, ddmd_list)
    mdddd_list = M_X_D_list(md_list, dddd_list)
    np.savetxt(mid_result_path + "mmmmd_list.txt", mmmmd_list, delimiter=',', fmt='%d')
    np.savetxt(mid_result_path + "mmmdd_list.txt", mmmdd_list, delimiter=',', fmt='%d')
    np.savetxt(mid_result_path + "mmdmd_list.txt", mmdmd_list, delimiter=',', fmt='%d')
    np.savetxt(mid_result_path + "mmddd_list.txt", mmddd_list, delimiter=',', fmt='%d')
    np.savetxt(mid_result_path + "mdmmd_list.txt", mdmmd_list, delimiter=',', fmt='%d')
    np.savetxt(mid_result_path + "mdmdd_list.txt", mdmdd_list, delimiter=',', fmt='%d')
    np.savetxt(mid_result_path + "mddmd_list.txt", mddmd_list, delimiter=',', fmt='%d')
    np.savetxt(mid_result_path + "mdddd_list.txt", mdddd_list, delimiter=',', fmt='%d')

