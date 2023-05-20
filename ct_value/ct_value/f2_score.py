"""
會將程式的輸出加上時間戳記
並同步儲存於Log資料夾中
"""
import os, sys, time

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

class Logger(object):
    def __init__(self, filename='Log/default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    @classmethod
    def timestamped_print(self, *args, **kwargs):
        _print(time.strftime("[%Y/%m/%d %X]"), *args, **kwargs)

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def log_history(name_s_log):
    # log
    createFolder('Log/')
    sys.stdout = Logger('Log/' + name_s_log + '.log', sys.stdout)
    sys.stderr = Logger('Log/' + name_s_log + '.err', sys.stderr)

if __name__ == '__main__':
    _print = print
    print = Logger.timestamped_print
    log_history(os.path.basename(__file__))

from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def loadColumns(path):
    file = open(path, "r")
    index = [line.strip() for line in file]
    file.close()
    return index

def Statistic(DataSavePath, StatisticsCounter, filename):
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
    
    sortedCounter = OrderedDict(sorted(StatisticsCounter.items()))
    sumValues = list(sortedCounter.keys())
    count = list(sortedCounter.values())
    sumValues = [str(i) for i  in sumValues]

    plt.figure(figsize=(30,16))
    plt.plot(sumValues, count, 'r-o')
    plt.xticks(rotation=90)
    plt.xlabel('該特徵值出現的次數')
    plt.ylabel('有多少特徵值(個)')
    plt.savefig(os.path.join(DataSavePath,'2_distribute',filename+'.jpg'), dpi=100)
    plt.close()

def score(read_path, save_path, print=print):
    os.makedirs(os.path.join(save_path, '1_statistic', '1_raw_Counter'), exist_ok=True)
    os.makedirs(os.path.join(save_path, '1_statistic', '2_P-value'), exist_ok=True)
    os.makedirs(os.path.join(save_path, '3_weight'), exist_ok=True)

    # 統計所有特徵出現的次數，畫出統計圖表
    totalStatisticsCounter = Counter()

    # 讀檔
    for i, filename in enumerate(os.listdir(read_path)):
        if not filename.endswith('.csv'): continue
        print(i, filename)
        data = pd.read_csv(os.path.join(read_path, filename), nrows=1)
        column_list = data.columns.to_list()

        filename = filename.replace('.csv', '/')
        os.makedirs(os.path.join(save_path, '1_statistic', '1_raw_Counter', filename), exist_ok=True)
        os.makedirs(os.path.join(save_path, '1_statistic', '2_P-value', filename), exist_ok=True)
        os.makedirs(os.path.join(save_path, '2_distribute', filename), exist_ok=True)
        os.makedirs(os.path.join(save_path, '3_weight', filename), exist_ok=True)

        for c, col in enumerate(column_list):
            if col in ['attack_cat', 'label']: continue

            print(str(c) + ' ' + col)

            b = Counter()  # b = benign
            m = Counter()  # m = malicious
            # calculate the probability

            filename = filename.replace('/', '.csv')
            data = pd.read_csv(os.path.join(read_path, filename), usecols=[col, 'label'], low_memory=False)
            filename = filename.replace('.csv', '/')

            # 將不同的label拆分為不同的資料集
            benign = data[data['label'] == 0]
            malicious = data[data['label'] != 0]

            # 某特徵c在label為benign/malicious出現的次數
            b += Counter(benign[column_list[c]])
            m += Counter(malicious[column_list[c]])

            bc_df = pd.DataFrame.from_dict(b, orient='index').reset_index()
            mc_df = pd.DataFrame.from_dict(m, orient='index').reset_index()
            
            #計算權重
            bm_raw_counter = b + m
            weight = np.log10(np.array([value for value in bm_raw_counter.values()]))
            bm_df = pd.DataFrame.from_dict(bm_raw_counter, orient='index').reset_index()
            bm_df.insert(loc = 2, column='weight', value=weight)

            # raw counter save
            bc_df.to_csv(os.path.join(save_path, '1_statistic', '1_raw_Counter', filename, 
            'b_'+str(c)+'_'+column_list[c]+'.csv'), index=None, header=['value', 'count'])
            mc_df.to_csv(os.path.join(save_path, '1_statistic', '1_raw_Counter', filename, 
            'm_'+str(c)+'_'+column_list[c]+'.csv'), index=None, header=['value', 'count'])
            
            # weight save
            bm_df.to_csv(os.path.join(save_path, '3_weight', filename, 
            str(c)+'_'+column_list[c]+'.csv'), index=None, header=['value', 'count', 'weight'])

            # 統計圖表資料
            StatisticsCounter = Counter([val for val in bm_raw_counter.values()])
            totalStatisticsCounter += StatisticsCounter


            for val, freq in b.items():
                b[val] = b[val] / bm_raw_counter[val]
            for val, freq in m.items():
                m[val] = m[val] / bm_raw_counter[val]

            b_pvalue_df = pd.DataFrame.from_dict(b, orient='index').reset_index()
            m_pvalue_df = pd.DataFrame.from_dict(m, orient='index').reset_index()

            b_pvalue_df.to_csv(os.path.join(save_path, '1_statistic', '2_P-value', filename,
            'b_'+str(c)+'_'+column_list[c]+'.csv'), index=None, header=['value', 'pvalue'])
            m_pvalue_df.to_csv(os.path.join(save_path, '1_statistic', '2_P-value', filename,
            'm_'+str(c)+'_'+column_list[c]+'.csv'), index=None, header=['value', 'pvalue'])

            Statistic(save_path, StatisticsCounter, filename + str(c)+'_'+column_list[c])
        Statistic(save_path, totalStatisticsCounter, filename + 'total')
            
if __name__ == '__main__':
    read_path = os.path.join('data', '1_preprocess')

    save_path = os.path.join('data', '2_score')
    os.makedirs(save_path, exist_ok=True)

    score(read_path, save_path)