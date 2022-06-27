import pandas as pd
import datetime
import time
from statistics import mean
import os
import shutil

def str_to_secs(value):
    time_value = time.strptime(value.split('.')[0],'%H:%M:%S')
    return datetime.timedelta(hours=time_value.tm_hour,minutes=time_value.tm_min,seconds=time_value.tm_sec).total_seconds()

df = pd.read_csv('EPIC_train_action_labels.csv', sep=',')
df = df[df['participant_id']=='P01']
start = list(map(str_to_secs, df['start_timestamp'].tolist()))
end = list(map(str_to_secs, df['stop_timestamp'].tolist()))
length = []
for start_i, end_i in zip(start,end):
    length.append(end_i - start_i)
# print(mean(length)*60) # 218.8349
# 220 frames -> 3.7s of video (2.5-1.2)
# 200 frames -> 3.3s of video (2.1-0.9)
selected_by_length = list(map(lambda x: x*60 >= 200, length))
df = df[selected_by_length]

df['action'] = df.apply (lambda x: x['verb'] + '_' + x['noun'], axis=1)
# print(df['action'].value_counts().nlargest(20))
# top10: value_counts > 10
df = df.groupby('action').filter(lambda x: len(x) >= 10)
# print(set(df['action'].tolist()))
with open('actionlist.txt','w') as f:
    f.write('\n'.join(set(df['action'].tolist())) + '\n')

df.to_csv('EK55_reduced.csv',index=False)


# /workspace/toyota_smarthome/rgb_frames/Walk/Walk_p17_r01_v04_c05/,90

# No training, just evaluation with t-SNE or PCA
with open('test.csv','w') as f:
    for idx, row in df.iterrows():
        frames = int(row['stop_frame']) - int(row['start_frame'])
        f.write('/workspace/epic-kitchens55/rgb_frames/' + row['action'] + '/' + str(row['uid']) + '/,' + str(frames) + '\n')


with open('video_ids.txt','w') as f:
    f.write('\n'.join(set(df['video_id'].tolist())) + '\n')

base_source_path = '/datasets/EPIC-KITCHENS/P01/rgb_frames/'
base_dest_path = '/workspace/epic-kitchens55/rgb_frames/'
for idx, row in df.iterrows():
    source_path = os.path.join(base_source_path, row['video_id'])
    dest_path = os.path.join(base_dest_path, row['action'], str(row['uid']))
    if not os.path.exists(source_path):
        print(f'ERROR: could not find source path {source_path}')
        continue
    # Remove folder and contents if exists
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    os.makedirs(dest_path)
    
    for img_file in range(row['start_frame'],row['stop_frame']):
        img_source_path = os.path.join(source_path, 'frame_' + str(img_file).zfill(10) + '.jpg')
        img_dest_path = os.path.join(dest_path, 'image_' + str(img_file-row['start_frame'] + 1).zfill(5) + '.jpg')
        shutil.copyfile(img_source_path, img_dest_path)

