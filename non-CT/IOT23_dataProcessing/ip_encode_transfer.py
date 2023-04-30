import pandas as pd
from collections import Counter
import netaddr
from IPy import IP


def ip2decimalism(ip):
    if IP(ip).version()== 4:
        ip = int(netaddr.IPAddress(ip))
    elif IP(ip).version() == 6:
        ip = int(netaddr.IPNetwork(ip).value)
    else:
        ip = ip
        print(ip)
    return float(ip)

data_tst = pd.read_table('trans_combine.csv', header=0, sep=',', comment='#', index_col=False, engine='python', chunksize=10000)

for chunk in data_tst:
    print(chunk)
    for i in chunk['id.orig_h']:
        new_ip = ip2decimalism(i)
        chunk['id.orig_h'] = chunk['id.orig_h'].replace(i, new_ip)
    print(Counter(chunk['id.orig_h']))
    for i in chunk['id.resp_h']:
        new_ip = ip2decimalism(i)
        chunk['id.resp_h'] = chunk['id.resp_h'].replace(i, new_ip)
    print(Counter(chunk['id.resp_h']))
    print(chunk)
    chunk.to_csv('output_combine.csv', mode='a', index=False, header=False)