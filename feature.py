import pandas
import math
import pywt
import numpy as np
from scipy.spatial.distance import pdist
from numba import jit

@jit
def my_pdist(data1, data2, type=0):  ##不支持批量计算
    # 距离度量           data1  维度为(1,n)的向量    data2 维度为(1,n)的向量
    res = []
    x1 = data1
    x2 = data2
    if type == 0:
        y = np.sqrt(np.dot((x1 - x2), ((x1 - x2).T)))  # Euclidean Distance    欧式距离
        res.append(y)
    elif type == 1:
        y = np.sum(np.abs(x1 - x2))  # Manhattan Distance         曼哈顿距离
        res.append(y)
    elif type == 2:  # Chebyshev Distance         切比雪夫距离
        y = np.abs(x1 - x2).max()
        res.append(y)
    elif type == 3:
        y = np.dot(x1, x2.T) / (np.linalg.norm(x1) * np.linalg.norm(x2))  # Cosine         cos夹角
        res.append(y)
    elif type == 4:  # Hamming           汉明距离
        y = np.mean(x1 != x2)
        res.append(y)
    elif type == 5:
        up = np.double(np.bitwise_and((x1 != x2), np.bitwise_or(x1 != 0, x2 != 0)).sum())  # jaccard          杰卡德距离
        down = np.double(np.bitwise_or(x1 != 0, x2 != 0).sum())
        y = (up / down)
        res.append(y)
    elif type == 6:  # Pearson correlation       皮尔逊相关性
        x1_ = x1 - np.mean(x1)
        x2_ = x2 - np.mean(x2)
        y = np.dot(x1_, x2_.T) / (np.linalg.norm(x1_) * np.linalg.norm(x2_))
        res.append(y)
    else:
        res.append(-1)
        pass
    return float(res[0])
@jit
def dist_martix(sdata, type=0):

    z = sdata
    b = z
    m15 = b.T
    k = 0
    [x, y] = np.shape(m15)
    n = int(y * (y - 1) / 2)
    print(x, y)
    dist15 = np.zeros((n, 3))
    for i in range(0, y):
        for j in range(i + 1, y):
            a = m15[:, i]
            b = m15[:, j]
            dist15[k, 0] = i
            dist15[k, 1] = j
            k = k + 1
    dist15[:, 2] = pdist(z, 'cosine')
    print(dist15[:, 0])
    print(dist15[:, 1])
    xx = dist15
    ND = int(max(xx[:, 1])) + 1
    NL = int(max(xx[:, 0])) + 1
    if NL > ND:
        ND = NL
    N = np.shape(xx)[0]
    dist = np.zeros((ND, ND))
    print(np.shape(dist))
    for i in range(0, N):
        ii = int(xx[i, 0])
        jj = int(xx[i, 1])
        dist[ii, jj] = xx[i, 2]
        dist[jj, ii] = xx[i, 2]
    return dist
@jit
def nextpow2(i):
    """
    @brief Find 2^n that is equal to or greater than.
    @param i [int] 数据长度
    @return 返回i长度的下个2**n次的n
    """
    n = 1
    while n < i: n *= 2
    return n
@jit
def my_fft(data, fs=1024):                                 #fft计算
    fft_list = []
    for item in data:
        X = item
        n = len(X)
        N = nextpow2(n)
        X = X - np.mean(X)
        Y = np.fft.fft(X,N,axis=0)/n
        length = int(N/2) + 1
        f = fs/2*np.linspace(0,1,length)
        A = 2 * np.abs(Y[0:length])
        fft_list.append(A)
    fft_martix = np.array(fft_list)
    return fft_martix
@jit
def mywavelet_packet(input, n, slice_no, wavelet = 'db2', mode = 'sym'):
    #小波包分解   input 输入数据 (m,n)  m为样本数 n为样本长度  n为深度  slice_no
    wave_list = []
    for item in input:
        X = item
        packet = pywt.WaveletPacket(X, wavelet, mode, n)  # 小波包分解
        if slice_no >= pow(2, n):
            wave_list.append(X)
        if slice_no < 0:
            wave_list.append(pywt.WaveletPacket.reconstruct(packet))  # 重构
        q = packet.get_level(n)

        cl = q[slice_no].data
        for i in range(0, pow(2, n)):  # 拆解每一层
            l = len(q[i].data)
            q[i].data = np.zeros(l)
        q[slice_no].data = cl
        val = pywt.WaveletPacket.reconstruct(packet)  # 重构
        wave_list.append(val)
    wave_matrix = np.array(wave_list)
    return wave_matrix
def TD_feature(data, keys_list):                   #时域特征
    """
    时域信号的相关特征，包括均值，标准差，偏度指标，峭度指标，峰值等

    * input：数据
    * returns：包含众多时域指标的二维矩阵（例如有m个时域指标，则返回n*m的矩阵）
    """

    td_feature = {}
    df = pandas.DataFrame(data)
    T1 = df.mean(axis=1)
    td_feature['T1'] = np.array(T1)
    T2 = df.std(axis=1)
    td_feature['T2'] = np.array(T2)
    T3 = df.skew(axis=1)
    td_feature['T3'] = np.array(T3)
    T4 = df.kurt(axis=1)
    td_feature['T4'] = np.array(T4)
    T5 = np.max(np.abs(data), axis=1)
    td_feature['T5'] = T5
    T6 = T5 / np.mean(np.abs(data), axis=1)
    td_feature['T6'] = T6
    T7 = np.mean(np.sqrt(np.abs(data)), axis=1) ** 2
    td_feature['T7'] = T7
    T8 = T5 / T7
    td_feature['T8'] = T8
    T9 = np.sqrt(np.mean(np.square(data), axis=1))
    td_feature['T9'] = T9
    T10 = T9 / np.mean(np.abs(data))
    td_feature['T10'] = T10
    T11 = T5 / T9
    td_feature['T11'] = T11
    feature_list = {}
    for item in keys_list:
        feature_list[item] = td_feature[item]
    return feature_list
def FD_feature(data, fs, keys_list):           #频域特征
        """
        频域信号的相关特征，包括均值，方差，偏度指标，峭度指标，峰值等

        * input：数据
        * returns：包含众多频域指标的二维矩阵（例如有m个时域指标，则返回n*m的矩阵）
        """
        fd_feature = {}
        fftdata = my_fft(data,fs)
        fft_f = fs / 2 * np.linspace(0, 1, np.shape(fftdata)[1])
        df = pandas.DataFrame(fftdata)
        F1 = df.mean(axis=1)
        #fd_feature['F1'] = float(F1)
        fd_feature['F1'] = np.array(F1)
        F2 = df.std(axis=1)
        fd_feature['F2'] = np.array(F2)
        F3 = df.skew(axis=1)
        fd_feature['F3'] = np.array(F3)
        F4 = df.kurt(axis=1)
        fd_feature['F4'] = np.array(F4)
        F5 = np.dot(fft_f,fftdata.T).T/np.sum(fftdata,axis=1)
        fd_feature['F5'] = F5
        n_samples = fftdata.shape[0]
        n_length = fftdata.shape[1]
        b = np.vstack((fft_f for i in range(n_samples))) - np.vstack((F5 for i in range(n_length))).T
        F6 = np.sqrt(np.mean(np.dot((b**2),fftdata.T), axis=1))
        fd_feature['F6'] = F6
        F7 = F6/F5
        fd_feature['F7'] = F7
        F8 = np.mean(np.dot((b**3),fftdata.T), axis=1) / (F6 ** 3)
        fd_feature['F8'] = F8
        F9 = np.mean(np.dot((b**4),fftdata.T), axis=1) / (F6 ** 4)
        fd_feature['F9'] = F9
        F10 = np.mean(np.dot((np.abs(b)**0.5),fftdata.T), axis=1) / (F6 ** 0.5)
        fd_feature['F10'] = F10
        F11 = np.sqrt(np.dot(fft_f ** 2, fftdata.T).T / np.sum(fftdata, axis=1))
        fd_feature['F11'] = F11
        F12 = np.sqrt(np.dot(fft_f**4,fftdata.T).T/(np.dot(fft_f**2,fftdata.T).T))
        fd_feature['F12'] = F12
        F13 = np.sqrt(np.mean(np.square(fftdata),axis=1))
        fd_feature['F13'] = F13
        F14 = np.max(np.abs(fftdata),axis=1)
        fd_feature['F14'] = F14
        feature_list = {}
        for item in keys_list:
            feature_list[item] = fd_feature[item]
        return feature_list
@jit
def ST_energy(data, frameSize=256, nstep=128):  # 短时能量
    """
    获取短时能量
    * framesiaze：帧长
    * length：每帧的间隔
    * returns：短时能量
    """
    energy_list = []
    for item in data:
        data_ = item
        wlen = len(data_)
        step = nstep
        frameNum = math.ceil(wlen / step)
        energy = np.zeros((frameNum))
        for i in range(frameNum):
            curFrame = data_[i * step: min(i * step + frameSize, wlen)]
            curFrame = np.array(curFrame).astype(np.float32)

            #curFrame = data_[np.arange(i * step, min(i * step + frameSize, wlen))]
            energy[i] = np.sum(curFrame ** 2)/len(curFrame)

        energy_list.append(energy)
    energy_matrix = np.array(energy_list)
    return energy_matrix
def ST_sum(data, frameSize=256, nstep=128):  # 短时能量
    """
    获取短时能量
    * framesiaze：帧长
    * length：每帧的间隔
    * returns：短时能量
    """
    energy_list = []
    for item in data:
        data_ = item
        wlen = len(data_)
        step = nstep
        frameNum = math.ceil(wlen / step)
        energy = np.zeros((frameNum))
        for i in range(frameNum):
            curFrame = data_[np.arange(i * step, min(i * step + frameSize, wlen))]
            energy[i] = np.sum(curFrame)
        energy_list.append(energy)
    energy_matrix = np.array(energy_list)
    return energy_matrix
@jit
def ST_ZCR(data,frameSize=256, nstep=128):                     #短时过零率
    """
    获取短时过零率
    * framesiaze：帧长
    * returns：短时过零率（横坐标是帧序号）
    """
    zcr_list = []
    for item in data:
        data_ = item
        wlen = len(data_)
        step = nstep
        frameNum = math.ceil(wlen/step)
        zcr = np.zeros((frameNum))
        for i in range(frameNum):
            curFrame = data_[np.arange(i*step,min(i*step+frameSize,wlen))]
            zcr[i] = sum(np.abs(np.sign(curFrame[0:-1])-np.sign(curFrame[1::])))/2/frameSize
        zcr_list.append(zcr)
    zcr_matrix = np.array(zcr_list)
    return zcr_matrix
def Normalize(data):                  #标准化
    normalize_list = []
    for item in data:
        datas = item
        std = np.std(datas)
        if std != 0:
            normal = (datas - np.mean(datas))/np.std(datas)
        else:
            normal = 0
        normalize_list.append(normal)
    normalize_martix = np.vstack(normalize_list)
    return normalize_martix
def mapminmax(data):              #归一化
    map_list = []
    for item in data:
        maps = item
        peak2peak = max(maps) - min(maps)
        if peak2peak != 0:
            map = (maps - min(maps)) /peak2peak
        else:
            map = 0
        map_list.append(map)
    map_martix = np.vstack(map_list)
    return map_martix
class TimeFeature():
    def __init__(self, keys_list = ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11']):
        self.__keys_list = keys_list
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, y=None, **fit_params):
        t_dic = TD_feature(X, keys_list=self.__keys_list)
        td_list = []
        for keys in t_dic:
            td_list.append(t_dic[keys])
        td_martix = np.vstack(td_list)
        return td_martix.T
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)
class FreqFeature():
    def __init__(self, freq,keys_list = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14']):
        self.__keys_list = keys_list
        self.__freq = freq
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, y=None, **fit_params):
        f_dic = FD_feature(X, self.__freq,keys_list=self.__keys_list)
        fd_list = []
        for keys in f_dic:
            fd_list.append(f_dic[keys])
        fd_martix = np.vstack(fd_list)
        return fd_martix.T
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)
class ST_Feature():
    def __init__(self,metric=0,pf=0.1,ps=0.1):
        self.__pf = pf
        self.__ps = ps
        self.__metric = metric
        pass
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, y=None, **fit_params):
        if self.__metric == 0:
            d_shape = np.shape(X)
            frameSize = int(self.__pf*d_shape[1])
            nstep = int(self.__ps*d_shape[1])
            return ST_energy(X, frameSize, nstep)
        elif self.__metric == 1:
            d_shape = np.shape(X)
            frameSize = int(self.__pf * d_shape[1])
            nstep = int(self.__ps * d_shape[1])
            return ST_sum(X, frameSize, nstep)
        elif self.__metric == 2:
            d_shape = np.shape(X)
            frameSize = int(self.__pf * d_shape[1])
            nstep = int(self.__ps * d_shape[1])
            return ST_ZCR(X, frameSize, nstep)
        else:
            return X
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)
class MYFFT():
    def __init__(self,fs):
        self.__fs = fs
        pass
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, y=None, **fit_params):
        return my_fft(X,fs=self.__fs)
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)
class MYWAVE():
    def __init__(self,n,slice_no):
        self.__n = n
        self.__slice_no = slice_no
        pass
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, y=None, **fit_params):
        return mywavelet_packet(X,n = self.__n,slice_no = self.__slice_no)
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)
class ORIGIN():
    def __init__(self):
        pass
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, y=None, **fit_params):
        return X
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)
def test():
    data = np.random.random((1,10000))
    print(np.shape(data[0]))
    val = TD_feature(data,['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11'])
    print(val)
    val = FD_feature(data,5000,['F1','F2','F3','F4','F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14'])
    print(val)
    data1 = np.random.random((10, 10000))
    data2 = np.random.random((10, 10000))
    data1 = Normalize(data1)
    print(np.shape(data1))
    data1 = mapminmax(data1)
    print(np.shape(data1))
    val = ST_energy(data1,256,128)
    print(np.shape(val))
    val = ST_ZCR(data1,256,128)
    print(np.shape(val))
    val = my_pdist(data,data1[0:1],0)
    print(val)
    val = dist_martix(data1,data2,0)
    print(val)
    print(np.shape(val))
    val = mywavelet_packet(data1,2,0)
    print(np.shape(val))