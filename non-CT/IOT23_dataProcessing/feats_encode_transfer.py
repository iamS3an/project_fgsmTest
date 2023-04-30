import pandas as pd
from collections import Counter


data_tst = pd.read_table('combine.csv', header=0, sep=',', comment='#', index_col=False, engine='python')
print(data_tst)

data_tst = data_tst.drop(['tunnel_parents', 'label detailed-label'], axis=1)
print(data_tst)

proto_list = list(Counter(data_tst['proto']).keys())
print(proto_list)
for val in proto_list:
    data_tst['proto'].replace(val, proto_list.index(val), inplace=True)
print(Counter(data_tst['proto']))

service_list = list(Counter(data_tst['service']).keys())
print(service_list)
for val in service_list:
    data_tst['service'].replace(val, service_list.index(val), inplace=True)
print(Counter(data_tst['service']))

conn_state_list = list(Counter(data_tst['conn_state']).keys())
print(conn_state_list)
for val in conn_state_list:
    data_tst['conn_state'].replace(val, conn_state_list.index(val), inplace=True)
print(Counter(data_tst['conn_state']))

history_list = list(Counter(data_tst['history']).keys())
print(history_list)
for val in history_list:
    data_tst['history'].replace(val, history_list.index(val), inplace=True)
print(Counter(data_tst['history']))

label_list = list(Counter(data_tst['label']).keys())
print(label_list)
for val in label_list:
    data_tst['label'].replace(val, label_list.index(val), inplace=True)
print(Counter(data_tst['label']))

data_tst = data_tst.sample(frac=1)
print(data_tst)

data_tst['label'].to_csv('2label_combine.csv', mode='w', index=False, header=False)
data_tst = data_tst.drop('label', axis=1)
print(data_tst)

data_tst.to_csv('trans_combine.csv', mode='w', index=False, header=True)
