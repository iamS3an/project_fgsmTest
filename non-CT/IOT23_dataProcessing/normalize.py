import numpy as np
import pandas as pd
import matplotlib
from glob import glob


def read_file_colName(FS_Fname): # 讀TXT的funtion
    with open(FS_Fname, "r") as f:
        names = f.readlines()
    names = [x.strip().replace("\n", "") for x in names]# 把末尾的'\n'刪掉用""代替
    return names

IOTLabel = read_file_colName("IOTLabel.txt")# 用function讀TXT
# print(IOTLabel)# TXT內容
# print(len(IOTLabel)) # TXT有幾行

files = glob('output_combine.csv')#合併檔案用['onehot_1.csv', 'onehot_3.csv']
# print(files)

mindata_tst = pd.read_table("Last_min.csv", names=IOTLabel, sep=',', comment='#', index_col=False,  engine='python')

mindata = mindata_tst.squeeze()

maxdata_tst = pd.read_table("Last_max.csv", names=IOTLabel, sep=',', comment='#', index_col=False,  engine='python')

max_data = maxdata_tst.squeeze()


#多檔 標準化到0.1間
for file in files:
    data_tst = pd.read_table(file, names=IOTLabel, header=None, sep=',', comment='#', index_col=False, engine='python',chunksize=5000)

    for chunk in data_tst:
        # print(chunk['local_orig'])
        denominator = max_data - mindata
        new_data = (chunk-mindata)/denominator

        new_data.fillna(0, inplace=True)
        new_data.replace("", "0", inplace=True)  # 補空值
        print(new_data)
        new_data.to_csv("normalize_combine.csv",mode='a', index=False, header=None)

