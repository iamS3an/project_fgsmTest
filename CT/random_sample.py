import pandas as pd
from collections import Counter

data_tst = pd.read_table("CTU-IoT-Malware-Capture-43-1.csv", header=0, sep=',', comment='#', index_col=False, engine='python', chunksize=100000)

data_size = 0
for chunk in data_tst:
    print(chunk)
    chunk_benign = chunk.loc[chunk['label'] == 'benign'].sample(n=750, random_state=999)
    chunk_malicious = chunk.loc[chunk['label'] == 'malicious'].sample(n=750, random_state=999)
    chunk = pd.concat([chunk_benign, chunk_malicious])
    print("-" * 30)
    chunk['label'].replace("benign", "0", inplace=True)
    chunk['label'].replace("malicious", "1", inplace=True)
    print(chunk)
    chunk['label'].to_csv('2label_combine.csv', mode='a', index=False, header=0)
    print("-" * 30)
    chunk.drop('label', axis=1, inplace=True)
    chunk.drop('tunnel_parents', axis=1, inplace=True)
    chunk.to_csv('normalize_combine.csv', mode='a', index=False, header=False)
    print(chunk)
    data_size += chunk.shape[0]
    print(f"Now data size: {data_size} rows")
