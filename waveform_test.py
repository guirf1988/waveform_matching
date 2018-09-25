import data_load
import feature
import os
import matplotlib.pyplot as plt
from waveform_matching import *
def test_fsk(plot = False):
    FindPath = "./data_fsk/"
    FileNames = os.listdir(FindPath)
    names = []
    for file_name in FileNames:
        fullfilename=os.path.join(FindPath,file_name)
        names.append(fullfilename)
    data_all = data_load.data_read_single(names[0])
    datas = []
    datas.append(data_all)  #data_src
    datad = []
    datad.append(data_all[45000:75000]) #data_dst   模板数据段
    data_ex = data_all[45000:75000]
    plt.subplot(3,1,1)
    plt.plot(data_all)
    plt.subplot(3,1,2)
    plt.plot(data_ex)
    plt.subplot(3, 1, 3)
    plt.plot(data_ex[2500:4500])
    plt.show()
    val = matching_from_energy(datas,datad) #目标搜索
    data_dst = []
    data_dst.append(data_ex[2500:4500])
    for i in range(0,len(val[0])):
        data_ex = data_all[val[0][i]:val[1][i]]
        data_s = []
        data_s.append(data_ex)
        index = match_from_sim(data_s,data_dst,para=[30])
        if len(index[0]) > 0:
            plt.subplot(2, 1, 1)
            plt.plot(data_ex)
            plt.subplot(2, 1, 2)
            plt.plot(data_ex[index[0][0]:index[1][0]])
            filename = './png/fsk_' + str(i) + '.png'
            plt.savefig(filename)
            print(filename)
            if plot == False:
                plt.close()
            else:
                plt.show()
if __name__=='__main__':
    test_fsk(plot=True)
