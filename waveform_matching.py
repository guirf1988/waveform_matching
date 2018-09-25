import numpy as np
from feature import *
'''
matching_from_energy(data, data_dst, nstep=1,para=[30])
data 待搜索数据段   size  1*m         m 数据段长度
data_dst 模板数据段  size 1*n         n 模板数据长度
nstep   步长系数   窗口宽度=模板长度 n     步长=步长系数*窗口长度
pf  pf越小 精度越高  鲁棒性下降    推荐为0.1
para   推荐值为30    容错率  越大可能搜多到的信号越多
返回值：角标序列  如：[[0,300,...600],[100,400,...700]]
'''
def matching_from_energy(data, data_dst, nstep=1, pf = 0.1,para=[30]):  #短时能量判断
    s_index_list = []
    e_index_list = []
    pe = int(para[0])
    y = data_dst[0]
    win_len = int(len(y) * pf)
    win_step = int(win_len)
    y_input = np.reshape(y, (1, len(y)))
    dst_energy = ST_energy(y_input, win_len, win_step).T    #模板信号的短时能量
    s_max = np.max(dst_energy)
    i_max = np.argmax(dst_energy)
    for i in range(0, len(data)):
        x = data[i]
        wlen = len(y)
        step = int(nstep * wlen)
        nFrames = math.ceil((len(x) - wlen) / step) + 1
        pdist = np.zeros((nFrames, 1))
        index = np.zeros((nFrames, 1))
        for j in range(0, nFrames):
            curFrame = x[j * step: min(j * step + wlen, len(x))]
            if (len(curFrame) != len(y)):
                break
            cur = np.reshape(curFrame, (1, len(curFrame)))
            cur_energy = ST_energy(cur, win_len, win_step).T
            val = np.max(cur_energy)
            index[j] = j * step + (np.argmax(cur_energy) - i_max) * win_step    #相位补偿
            if index[j] <= 0:
                index[j] = 0
            pdist[j] = val
        thrd = (pe/100) * s_max
        thrd_l = s_max - thrd
        thrd_h = s_max + thrd
        s_index = np.array(index[(pdist > thrd_l) &(pdist < thrd_h)]).astype(np.int32)  #信号过滤
        e_index = s_index + wlen
        s_index_list.append(s_index)
        e_index_list.append(e_index)
        return [s_index_list[0], e_index_list[0]]
'''
match_from_sim(data, data_dst, nstep=1,para=[30])
data 待搜索数据段   size  1*m         m 数据段长度
data_dst 模板数据段  size 1*n         n 模板数据长度
nstep   步长系数   窗口宽度=模板长度 n     步长=步长系数*窗口长度
para   推荐值为50    百分位数  越大条件越苛刻
返回值：角标序列  如：[[0],[100]]
'''
def match_from_sim(data, data_dst, nstep=0.05,para=[50]):  #短时能量结合频谱
    pe = int(para[0])
    y = data_dst[0]
    win_len = int(len(y) * 0.01)
    win_step = int(win_len)
    y_input = np.reshape(y, (1, len(y)))
    dst_energy = ST_energy(y_input, win_len, win_step).T
    dst_spec = abs(np.fft.fft(y - np.mean(y)))
    s_max = np.max(dst_energy)
    i_max = np.argmax(dst_energy)
    for i in range(0, len(data)):
        x = data[i]
        wlen = len(y)
        step = int(nstep * wlen)
        nFrames = math.ceil((len(x) - wlen) / step) + 1
        pdist = np.zeros((nFrames, 1))
        pdist_mean = np.zeros((nFrames, 1))
        index = np.zeros((nFrames, 1))
        for j in range(0, nFrames):
            curFrame = x[j * step: min(j * step + wlen, len(x))]
            if (len(curFrame) != len(y)):
                break
            cur = np.reshape(curFrame, (1, len(curFrame)))
            cur_energy = ST_energy(cur, win_len, win_step).T
            val = np.max(cur_energy)
            index[j] = j * step + (np.argmax(cur_energy) - i_max) * win_step
            if index[j] <= 0:
                index[j] = 0
            data_src = curFrame - np.mean(curFrame)
            pdist[j] = val
            src_spec = abs(np.fft.fft(data_src))
            pdist_mean[j] = my_pdist(dst_spec.T, src_spec.T, 3)
        thrd = (pe/100) * s_max
        thrd_mean = np.max(pdist_mean)
        s_index = np.array(index[(pdist > thrd)  & (pdist_mean == thrd_mean)]).astype(np.int32)
        e_index = s_index + wlen
        return [s_index, e_index]