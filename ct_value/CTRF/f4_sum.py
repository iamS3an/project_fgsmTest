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

import os, sys
import pandas as pd
import numpy as np

def sum(read_path, save_path, print=print):
    # 讀檔
    for i, filename in enumerate(os.listdir(read_path)):
        if not filename.endswith('.csv'): continue
        print(i, filename)
        data = pd.read_csv(os.path.join(read_path, filename), low_memory=False)
        data = data.drop(columns=['label'])

        nparr = data.to_numpy()
        sum =  np.sum(nparr, axis = 1)

        data = pd.read_csv(os.path.join(read_path, filename), low_memory=False)
        data['sum'] = sum
        data.to_csv(os.path.join(save_path, filename), index=None)

if __name__ == '__main__':
    read_path = os.path.join('data', '3_DataWithPvalue', 'ct-value')

    save_path = os.path.join('data', '4_sum')
    os.makedirs(save_path, exist_ok=True)

    sum(read_path, save_path)