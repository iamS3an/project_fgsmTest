import pandas as pd

data_tst = pd.read_table('combine_CT.csv', header=0, sep=',', comment='#', index_col=False, engine='python')
print(data_tst)

data_tst = data_tst.drop(['label detailed-label', 'label', 'sum'], axis=1)
print(data_tst)

data_tst.to_csv('normalize_combine.csv', mode='w', index=False, header=False)