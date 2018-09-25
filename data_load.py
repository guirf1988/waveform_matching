import os
import struct
def data_read(file_list,s_index=0,e_index=0):       #批量读取二进制文件 ，类型 float
    datas = []
    for i in range(0,len(file_list)):
        file = file_list[i]
        if os.path.isfile(file) == False:
            print('file not found')
            return datas
        fid = open(file, 'rb')
        data = fid.read()
        data = struct.unpack(str(int(len(data) / 4)) + 'f', data)
        datas.extend(data)
        fid.close()
    if e_index - s_index <= 0:
        return datas
    else:
        return datas[s_index:e_index]
def data_read_single(file_list,s_index=0,e_index=0):       #读取二进制文件 ，类型 float
    datas = []
    file = file_list
    if os.path.isfile(file) == False:
        print('file not found')
        return datas
    length = (e_index - s_index)*4      #计算长度
    fid = open(file,'rb')
    fid.seek(s_index*4,0)
    if length<=0:
        data = fid.read()
    else:
        data = fid.read(length)
    input = struct.unpack(str(int(len(data)/4)) + 'f', data)
    datas.extend(input)
    fid.close()
    return datas
