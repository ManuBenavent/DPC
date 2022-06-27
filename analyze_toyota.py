import pandas as pd
seq_len=5
num_seq = 8
downsample=3
split = 'process_data/data/toyota_smarthome/train.csv'
video_info = pd.read_csv(split, header=None)
counter = 0
suma =0
total = 0
for idx, row in video_info.iterrows():
    vpath, vlen = row
    suma += vlen
    total += 1
    if vlen-num_seq*seq_len*downsample <= 0:
        counter += 1


print(counter)
print(num_seq*seq_len*downsample)
print(suma/total)