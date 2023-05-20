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

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sys, os

def RandomForest(read_path, save_path, print=print):
    for i, dirname in enumerate(os.listdir(read_path)):
        print(i, dirname, "\n")
        file_path = os.path.join(read_path, dirname)
        for f2, filename in enumerate(os.listdir(file_path)):
            if not filename.endswith('.csv'): continue
            print(f2, filename)
            
            if filename.startswith('CT'):
                # continue
                # 讀檔
                data = pd.read_csv(os.path.join(file_path,filename), low_memory=False)

                # 刪除detailed-label、sum做2分類的隨機森林
                data = data.drop(columns=['sum'])
        
            elif filename.startswith('ori'):
                # 讀檔
                print(filename)
                data = pd.read_csv(os.path.join(file_path,filename), low_memory=False)
        
            # 顯示資料集資訊
            label_counts = data["label"].value_counts().to_dict()
            print(label_counts)
            
            # 拆分訓練資料集測試資料
            train_data = data.sample(frac=0.8, random_state=42)
            test_data = data.drop(train_data.index)

            # 預測目標、特徵
            target = 'label'
            features = [i for i in data.columns.to_list() if i not in ['label']]
            
            # 建立隨機森林分類器
            rf = RandomForestClassifier(n_estimators = 1, random_state=42, max_depth=1, bootstrap=False)

            # 訓練模型
            rf.fit(train_data[features], train_data[target])

            # 進行預測
            predictions = rf.predict(test_data[features])

            accuracy = (predictions == test_data[target]).mean()
            print('Accuracy: {:.3%}\n'.format(accuracy))


if __name__ == '__main__':
    read_path = os.path.join('data', '5_sample')

    save_path = os.path.join('data', '6_randomforest')
    os.makedirs(save_path, exist_ok=True)

    RandomForest(read_path, save_path)