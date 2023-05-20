import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from sklearn.preprocessing import MinMaxScaler

def grapth(data, filename):
    # 直方圖，一個特徵一個圖，分20個區間
    data.hist(bins=20, figsize=(20,15))
    plt.savefig(filename+'.png')
    plt.close()

def show_distribute(data, path, distribute_name, interval_list, mean = None):
    # 畫圖
    grapth(data, os.path.join(path,'grapth',distribute_name))
    # 將分布數據存入sample.txt
    with open(os.path.join(path,'sample.txt'), mode='a') as f:
        f.write(distribute_name+':\n')

        # 計算平均
        if mean != None:
            f.write(f'平均 : {mean}\n')

        f.write('分隔的區間\n')
        save_array = np.array(interval_list).reshape(1, -1)
        np.savetxt(f, save_array, fmt='%-5.2f')
        for col in data.columns:
            if col in ['sum', 'label', 'label detailed-label']: continue
            col_data = data[col].values
            interval_counts, intervals = np.histogram(col_data, bins=interval_list)
            save_array = interval_counts.reshape(1, -1)
            np.savetxt(f, save_array, fmt='%5d')
        
def sample(read_path_ct, read_path_ori, save_path, benign, print=print):
    # 讀CT值的csv檔案
    for i, filename in enumerate(os.listdir(read_path_ct)):
        if not filename.endswith('.csv'): continue
        print(i, filename)
        data = pd.read_csv(os.path.join(read_path_ct, filename), low_memory=False)
        data['label'].replace(benign, 'benign', inplace=True)
        data['label'] = data['label'].replace(to_replace = data['label'].unique()[1:], value = 'malicious')
        
        # 計算各label的數量
        label_counts = data['label'].value_counts().to_dict()
        print(label_counts)
        if len(label_counts) < 2: continue
        min_label = min(label_counts['benign'], label_counts['malicious'])

        # 保留較少數量的label，較多資料的label做之後的採樣
        if label_counts['benign'] > label_counts['malicious']:
            data_reserve = data[data['label'] != 'benign']
            data = data[data['label'] == 'benign']
        else:
            data_reserve = data[data['label'] == 'benign']
            data = data[data['label'] != 'benign']

        # 讀原始檔案
        ori_data = pd.read_csv(os.path.join(read_path_ori, filename), low_memory=False)
        ori_data['label'].replace(benign, 'benign', inplace=True)
        ori_data['label'] = ori_data['label'].replace(to_replace = ori_data['label'].unique()[1:], value = 'malicious')
        
        # 將檔案中的inf轉為float64的inf
        max_val = np.finfo(np.float64).max
        ori_data.replace([np.inf, -np.inf], max_val, inplace=True)

        # 把原始檔案min_max標準化
        scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        scaled_data = scaler.fit_transform(ori_data.iloc[:, :-1])
        ori_data.iloc[:, :-1] = pd.DataFrame(scaled_data, columns=ori_data.columns[:-1])

        # 把label中大於1的都改為1，0改成BENIGN，1改成milicious
        # ori_data.loc[ori_data['label'] > 1, 'label'] = 1
        # ori_data['label'] = ori_data['label'].replace({0: benign, 1: malicious})

        # 保留較少數量的label，較多資料的label做之後的採樣
        ori_data_reserve = ori_data.loc[data_reserve.index]
        ori_data = ori_data.loc[data.index]

        # 創建儲存資料夾
        filename = filename.replace('.csv', '/')
        os.makedirs(os.path.join(save_path, filename), exist_ok=True)
        
        # 初始化sample.txt
        with open(os.path.join(save_path, filename, 'sample.txt'), mode='w') as f:
            pass

        # 輸出原始資料分布
        os.makedirs(os.path.join(save_path, filename, 'grapth'), exist_ok=True)
        grapth(ori_data, os.path.join(save_path, filename, 'grapth', '0.原始資料'))

        # 把label相關column丟掉
        # data = data.drop(columns=['label', 'label detailed-label'])

        # CT值分成20個區間
        interval_list = np.linspace(-0.5, 0.5, 20+1).tolist()
        
        # 刪除誤差，把數字四捨五入到四位小數
        interval_list = list([round(x, 4) for x in interval_list])

        # 輸出Min-Max標準化後分布
        show_distribute(
            ori_data, 
            os.path.join(save_path, filename), 
            '1.原始資料Min-Max標準化', 
            interval_list
        )

        # 輸出CT值分布
        show_distribute(
            data, 
            os.path.join(save_path, filename), 
            '2.CT值資料', 
            interval_list,
            data['sum'].mean()
        )
        
        # # 隨機採樣20%
        # random_data = data.sample(frac=0.2, random_state=42)
        # random_ori_data = ori_data.loc[random_data.index]

        # 隨機採樣最少數量label的個數
        random_data = data.sample(n=min_label, random_state=42)
        random_ori_data = ori_data.loc[random_data.index]

        # 輸出隨機採樣原始分布
        show_distribute(
            random_ori_data, 
            os.path.join(save_path, filename), 
            '3.原始資料隨機採樣', 
            interval_list
        )
        sample_data = pd.concat([random_ori_data, ori_data_reserve])
        sample_data.to_csv(os.path.join(save_path, filename, 'ori_random_data.csv'), index=None)

        # 輸出隨機採樣CT分布
        show_distribute(
            random_data, 
            os.path.join(save_path, filename), 
            '4.CT資料隨機採樣', 
            interval_list,
            random_data['sum'].mean()
        )
        sample_data = pd.concat([random_data, data_reserve])
        sample_data.to_csv(os.path.join(save_path, filename, 'CT_random_data.csv'), index=None)

        # 依照sum的大小來排序
        sorted_idx = data.sort_values(by='sum', ascending=False).index
        data = data.reindex(sorted_idx)
        ori_data = ori_data.reindex(sorted_idx)

        # 取前20%
        data_top_20_percent = data.head(min_label)
        ori_data_top_20_percent = ori_data.head(min_label)

        # 輸出20%的CT分布
        show_distribute(
            data_top_20_percent, 
            os.path.join(save_path, filename), 
            '5.前20%CT資料', 
            interval_list,
            data_top_20_percent['sum'].mean()
        )
        sample_data = pd.concat([data_top_20_percent, data_reserve])
        sample_data.to_csv(os.path.join(save_path, filename, 'CT_top20_data.csv'), index=None)

        # 輸出20%的原始分布
        show_distribute(
            ori_data_top_20_percent, 
            os.path.join(save_path, filename), 
            '6.前20%原始資料', 
            interval_list
        )
        sample_data = pd.concat([ori_data_top_20_percent, ori_data_reserve])
        sample_data.to_csv(os.path.join(save_path, filename, 'ori_top20_data.csv'), index=None)

        # 取前18%
        data_top_18_percent = data.head(int(min_label))

        # 原本資料筆數
        len_18_percent = len(data_top_18_percent)

        # data改為剩下的82%
        data = data.loc[~data.index.isin(data_top_18_percent.index)]
        
        # 每個區間至少要有的資料筆數
        thread1 = 1

        # 分20個區，對每個 feature 紀錄每個區間內出現次數
        for col in data_top_18_percent.columns:
            if col in ['sum', 'label', 'label detailed-label']: continue
            additional_data = []
            col_data = data_top_18_percent[col].values
            interval_counts, intervals = np.histogram(col_data, bins=interval_list)

            # 在剩下的82%中，把每個區間都補至少有thread筆
            for i, counts in enumerate(interval_counts):
                # 已經滿足的就跳過
                nees_counts = thread1 - counts
                if nees_counts <= 0: continue
                # 找出區間範圍
                interval_start, interval_end = intervals[i], intervals[i+1]
                # 這個feature所有的資料
                col_values = data[col].values
                # 找出在區間內的所有資料
                in_interval = (col_values >= interval_start) & (col_values <= interval_end)
                additional_data = data.loc[in_interval & ~data.index.isin(data_top_18_percent.index)]
                # 選資料的CT值最高的
                additional_data = additional_data.iloc[:nees_counts]
                # 找不到就跳過
                if len(additional_data) <= 0: continue
                
                data_top_18_percent = data_top_18_percent._append(additional_data, ignore_index=False)
                data = data.loc[~data.index.isin(additional_data.index)]

        # 輸出HQSC後CT分布
        show_distribute(
            data_top_18_percent, 
            os.path.join(save_path, filename), 
            '7.HQSC後CT值資料', 
            interval_list,
            data_top_18_percent['sum'].mean()
        )
        sample_data = pd.concat([data_top_18_percent, data_reserve])
        sample_data.to_csv(os.path.join(save_path, filename, 'CT_HQSC_data.csv'), index=None)

        # 輸出HQSC後原始分布
        ori_data_top_18_percent = ori_data.loc[data_top_18_percent.index]
        show_distribute(
            ori_data_top_18_percent, 
            os.path.join(save_path, filename), 
            '8.HQSC後原始資料', 
            interval_list,
        )
        sample_data = pd.concat([ori_data_top_18_percent, ori_data_reserve])
        sample_data.to_csv(os.path.join(save_path, filename, 'ori_HQSC_data.csv'), index=None)

        print("增加資料筆數：", len(data_top_18_percent)-len_18_percent)

if __name__ == '__main__':
    read_path_ct = os.path.join('data', '4_sum')
    read_path_ori = os.path.join('data', '1_preprocess')

    save_path = os.path.join('data', '5_sample')
    os.makedirs(save_path, exist_ok=True)

    sample(read_path_ct, read_path_ori, save_path)