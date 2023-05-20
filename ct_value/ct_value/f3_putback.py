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

def putback(read_path_data, read_path_pvalue, save_path, print=print):
    os.makedirs(os.path.join(save_path, 'p-value'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'ct-value'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'wct-value'), exist_ok=True)
    # 讀檔
    for i, filename in enumerate(os.listdir(read_path_data)):
        if not filename.endswith('.csv'): continue
        print(i, filename)
        # pvalue data
        Pdata = pd.read_csv(os.path.join(read_path_data, filename), low_memory=False)
        # CTvalue data
        CTdata = pd.read_csv(os.path.join(read_path_data, filename), low_memory=False)
        # WCTvalue data
        WCTdata = pd.read_csv(os.path.join(read_path_data, filename), low_memory=False)

        pvaluePath = os.path.join(read_path_pvalue, '1_statistic', '2_P-value')
        weigthPath = os.path.join(read_path_pvalue, '3_weight')

        for c, col in enumerate(Pdata.columns.to_list()):
            if col in ['attack_cat', 'label']: continue
            print("Process", c, col)

            #將csv變成dict {'value': p-value}
            filename = filename.replace('.csv', '/')
            pvalueDataFilenameByColunm = os.path.join(pvaluePath,filename+'b_'+str(c)+'_'+col+'.csv')
            pvalueDF = pd.read_csv(pvalueDataFilenameByColunm, header=0, low_memory=False)
            print('benign P-value Data Shape', pvalueDF.shape)
            tmpDitc = pvalueDF.to_dict(orient='list')
            bDict = dict()

            for i in range(len(tmpDitc['value'])):
                bDict[tmpDitc['value'][i]] = tmpDitc['pvalue'][i]

            #將csv變成dict {'value': p-value}
            pvalueDataFilenameByColunm = os.path.join(pvaluePath,filename+'m_'+str(c)+'_'+col+'.csv')
            pvalueDF = pd.read_csv(pvalueDataFilenameByColunm, header=0, low_memory=False)
            print('malicious P-value Data Shape', pvalueDF.shape)
            tmpDitc = pvalueDF.to_dict(orient='list')
            mDict = dict()

            for i in range(len(tmpDitc['value'])):
                mDict[tmpDitc['value'][i]] = tmpDitc['pvalue'][i]

            #將csv變成dict {'value': weigth}
            weigthDataFilenameByColunm = os.path.join(weigthPath,filename+str(c)+'_'+col+'.csv')
            weigthDF = pd.read_csv(weigthDataFilenameByColunm, header=0, low_memory=False, usecols=['value', 'weight'])
            print('weight P-value Data Shape', weigthDF.shape)
            tmpDitc = weigthDF.to_dict(orient='list')
            weigth = dict()

            for i in range(len(tmpDitc['value'])):
                weigth[tmpDitc['value'][i]] = tmpDitc['weight'][i]

            #將原本csv內的值改為p_value
            for i in range(len(Pdata.index)):
                feature = Pdata.at[i, col]
                if Pdata.at[i, 'label'] == 'malicious':
                    Pdata.at[i, col] = mDict.get(feature, 0) * weigth.get(feature, 0)
                    CTdata.at[i, col] = (mDict.get(feature, 0)-0.5)
                    WCTdata.at[i, col] = (mDict.get(feature, 0)-0.5) * weigth.get(feature, 0)
                else:
                    Pdata.at[i, col] = bDict.get(feature, 0) * weigth.get(feature, 0)
                    CTdata.at[i, col] = (bDict.get(feature, 0)-0.5)
                    WCTdata.at[i, col] = (bDict.get(feature, 0)-0.5) * weigth.get(feature, 0)

        #存檔
        filename = filename.replace('/', '')
        Pdata.to_csv(os.path.join(save_path, 'p-value',filename+'.csv'), index=None)
        CTdata.to_csv(os.path.join(save_path, 'ct-value',filename+'.csv'), index=None)
        WCTdata.to_csv(os.path.join(save_path, 'wct-value',filename+'.csv'), index=None)

if __name__ == '__main__':
    read_path_data = os.path.join('data', '1_preprocess')

    read_path_pvalue = os.path.join('data', '2_score')

    save_path = os.path.join('data', '3_DataWithPvalue')
    os.makedirs(save_path, exist_ok=True)

    putback(read_path_data, read_path_pvalue, save_path)