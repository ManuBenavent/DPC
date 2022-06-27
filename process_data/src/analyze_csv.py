import pandas as pd
import os

num_seq = 3
seq_len = 5
downsample = 3

eliminados = 0
incluidos = 0
total_frames = 0
total_videos = 0

videoinfo_file = os.path.join('../data/toyota_smarthome', 'train.csv')
videinfo_df = pd.read_csv(videoinfo_file, header=None)
for index, row in videinfo_df.iterrows():
    vpath, vlen = row
    total_frames += vlen
    total_videos += 1
    print(vlen/30)
    if vlen-num_seq*seq_len*downsample <= 0:
        eliminados += 1
    else:
        incluidos += 1

print(f"Eliminados: {eliminados} - Incluidos: {incluidos}")

eliminados = 0
incluidos = 0

videoinfo_file = os.path.join('../data/toyota_smarthome', 'test.csv')
videinfo_df = pd.read_csv(videoinfo_file, header=None)
for index, row in videinfo_df.iterrows():
    vpath, vlen = row
    total_frames += vlen
    total_videos += 1
    if vlen-num_seq*seq_len*downsample <= 0:
        eliminados += 1
    else:
        incluidos += 1

print(f"Media: {(total_frames/total_videos)/30}")
print(f"TEST Eliminados: {eliminados} - Incluidos: {incluidos}")
