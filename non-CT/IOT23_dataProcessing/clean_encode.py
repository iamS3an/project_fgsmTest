import pandas as pd  # 引用套件(package)並縮寫為 pd 可以從setting→package導入
from collections import Counter
import netaddr
from IPy import IP


def read_file_colName(FS_Fname):  # 讀TXT的funtion
    with open(FS_Fname, "r") as f:
        names = f.readlines()
    names = [x.strip().replace("\n", "") for x in names]  # 把末尾的'\n'刪掉用""代替
    return names

def ip2decimalism(ip):#IP轉10進制
    #print(ip)
    if IP(ip).version()== 4:
        #print('IPv4')
        ip = int(netaddr.IPAddress(ip))
        #print(ip)
    elif IP(ip).version() == 6:
        #print('IPv6')
        ip = int(netaddr.IPNetwork(ip).value)
        #print(ip)
    else:
        ip = ip
        print(ip)
    #print("-" * 30)
    return float(ip)

IOTLabel = read_file_colName("IOTLabel_raw.txt")  # 用function讀TXT

# 多檔
data_tst = pd.read_table('conn.log.labeled', names=IOTLabel, header=None, sep='\\s+', comment='#', index_col=False, engine='python', chunksize=100000)
data_size = 0

for chunk in data_tst:
    print(chunk)
    chunk_benign = chunk.loc[chunk['label'] == 'Benign'].sample(n=750, random_state=999)
    chunk_malicious = chunk.loc[chunk['label'] == 'Malicious'].sample(n=750, random_state=999)
    chunk = pd.concat([chunk_benign, chunk_malicious])
    print(chunk)
    chunk = chunk.drop(['ts', 'uid', 'duration', 'orig_bytes', 'resp_bytes', 'local_orig', 'local_resp', 'tunnel_parents', 'detailed-label'], axis=1)
    chunk.replace("-", "0", inplace=True)  # 補空值
    chunk.replace("(empty)", "0", inplace=True)
    chunk['label'].replace("benign", "Benign", inplace=True)  # 修改Benign的大小寫問題
    # 做Label Encoding
    chunk['proto'].replace("icmp", 0, inplace=True)
    chunk['proto'].replace("udp", 1, inplace=True)
    chunk['proto'].replace("tcp", 2, inplace=True)
    chunk['service'].replace("dhcp", 0, inplace=True)
    chunk['service'].replace("dns", 1, inplace=True)
    chunk['service'].replace("http", 2, inplace=True)
    chunk['service'].replace("irc", 3, inplace=True)
    chunk['service'].replace("ssh", 4, inplace=True)
    chunk['service'].replace("ssl", 5, inplace=True)
    chunk['conn_state'].replace("S0", 0, inplace=True)
    chunk['conn_state'].replace("S1", 1, inplace=True)
    chunk['conn_state'].replace("SF", 2, inplace=True)
    chunk['conn_state'].replace("REJ", 3, inplace=True)
    chunk['conn_state'].replace("S2", 4, inplace=True)
    chunk['conn_state'].replace("S3", 5, inplace=True)
    chunk['conn_state'].replace("RSTO", 6, inplace=True)
    chunk['conn_state'].replace("RSTR", 7, inplace=True)
    chunk['conn_state'].replace("RSTOS0", 8, inplace=True)
    chunk['conn_state'].replace("RSTRH", 9, inplace=True)
    chunk['conn_state'].replace("SH", 10, inplace=True)
    chunk['conn_state'].replace("SHR", 11, inplace=True)
    chunk['conn_state'].replace("OTH", 12, inplace=True)
    chunk['history'].replace("^D", 0, regex=True, inplace=True)
    chunk['history'].replace('^S', 1, regex=True, inplace=True)
    chunk['label'].replace("Benign", 0, inplace=True)
    chunk['label'].replace("Malicious", 1, inplace=True)
    chunk['label'].to_csv('2label_combine.csv', mode='a', index=False, header=0)
    chunk = chunk.drop('label', axis=1)
    for i in chunk['id.orig_h']:
        new_ip = ip2decimalism(i)
        chunk['id.orig_h'] = chunk['id.orig_h'].replace(i, new_ip)
    for i in chunk['id.resp_h']:
        new_ip = ip2decimalism(i)
        chunk['id.resp_h'] = chunk['id.resp_h'].replace(i, new_ip)
    print(Counter(chunk['id.orig_h']))
    print(chunk)
    data_size += chunk.shape[0]
    print(f"Now data size: {data_size} rows")
    chunk.to_csv('output_combine.csv', mode='a', index=False, header=0)
    print("-" * 30)

# 不要讀取header 每列名稱為IOTLabel的內容 用\\s+即TAB和多個空格都視為分格號
# 沒有low memory因為engin與skipfooter衝突
# print('y_train : %s' % sorted(Counter(data_tst['class']).items()))
# %s用str將字串輸出
# Sorted→list里的值由小排到大
# 用glob套件的concat合併欄位
# ignore_index 避免索引值重複導致看不出來檔案已合併
# comment='#'忽略開頭是#的那整行資料
