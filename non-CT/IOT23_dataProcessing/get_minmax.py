import numpy as np
import pandas as pd
from glob import glob


def read_file_colName(FS_Fname): # 讀TXT的funtion
    with open(FS_Fname, "r") as f:
        names = f.readlines()
    names = [x.strip().replace("\n", "") for x in names]# 把末尾的'\n'刪掉用""代替
    return names

IOTLabel = read_file_colName("IOTLabel.txt")# 用function讀TXT


data_tst = pd.read_table("output_combine.csv", names=IOTLabel, header=None, sep=',', comment='#', index_col=False, engine='python',chunksize=50000)

for chunk in data_tst:
    print(chunk)
    chunk_max_list = []
    chunk_min_list = []

    chunk_max_data = chunk.max()
    chunk_min_data = chunk.min()


    chunk_max_list.append(chunk_max_data.tolist())
    chunk_min_list.append(chunk_min_data.tolist())

    """print(max_list)
    print(min_list)

    print("-" * 30)"""
    chunk_max_df = pd.DataFrame(np.array(chunk_max_list), dtype='float64')
    chunk_min_df = pd.DataFrame(np.array(chunk_min_list), dtype='float64')

    chunk_max_df.to_csv("chunk_max.csv",mode= 'a', index=None, header=None)
    chunk_min_df.to_csv("chunk_min.csv",mode= 'a', index=None, header=None)

    print(chunk_max_df)
    print(chunk_min_df)



# find the max value in all max dataframe


maxdata_tst = pd.read_table("chunk_max.csv", names=IOTLabel, header=None, sep=',', comment='#', index_col=False, engine='python',chunksize=50000)
for chunk in maxdata_tst:

    max_list = []

    max_data = chunk.max()

    max_list.append(max_data.tolist())

    Last_max_list = pd.DataFrame(np.array(max_list), dtype='float64')
    Last_max_list.to_csv("Last_max.csv",mode= 'a', index=False, header=None)

    print(Last_max_list)

mindata_tst = pd.read_table("chunk_min.csv", names=IOTLabel, header=None, sep=',', comment='#', index_col=False, engine='python',chunksize=50000)
for chunk in mindata_tst:

    min_list = []

    min_data = chunk.min()

    min_list.append(min_data.tolist())

    Last_min_list = pd.DataFrame(np.array(min_list), dtype='float64')
    Last_min_list.to_csv("Last_min.csv",mode= 'a', index=False, header=None)

    print(Last_min_list)
