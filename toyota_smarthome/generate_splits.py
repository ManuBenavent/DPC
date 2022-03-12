import os 
import glob
from random import random, shuffle 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
import pandas
from collections import Counter

def to_file(filename, data):
    with open(filename, 'w') as f:
        f.write('\n'.join(data) + '\n')

def plot_splits(fig_name, file_name, train, test, y, keys, bottom_adjustment):
    # Plot splits
    train_counts = Counter(train)
    plt.bar(train_counts.keys(), train_counts.values())
    test_counts = Counter(test)
    plt.bar(test_counts.keys(), test_counts.values())
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=bottom_adjustment)
    plt.legend(['Train','Test'])
    plt.savefig('splits/' + fig_name + '.png')
    plt.clf()

    counts = Counter(y)
    # Save percentages per split
    with open('splits/' + file_name + '.csv', 'w') as f:
        f.write('class dataset train test diff_splits\n')
        for key in keys:
            # st_dataset = round(counts[key]/len(y), 4)
            st_dataset = counts[key]
            st_train = round(train_counts[key]/len(train), 4)
            st_test = round(test_counts[key]/len(test), 4)
            st_diff_splits = round(abs(st_train - st_test), 4)
            f.write(key + ' ' + str(st_dataset) + ' ' + str(st_train) + ' ' + str(st_test) + ' ' + str(st_diff_splits) + '\n')

# Get data
if not os.path.exists('splits'): os.makedirs('splits')
videos_path = '/datasets/toyota_smarthome/rgb/mp4/*.mp4'
videos = glob.glob(videos_path)

# Group annotations
keys = set()
reduced_keys = set()
x = []
y = []
reduced_y = []
for video in videos:
    name = os.path.basename(video)
    label = name.split('_')[0]
    if '.' in label:
        reduced_label = label.split('.')[0]
    else:
        reduced_label = label
    reduced_keys.add(reduced_label)
    keys.add(label)
    x.append(name)
    y.append(label)
    reduced_y.append(reduced_label)

# Save classes
to_file('splits/classes.txt', keys)
to_file('splits/reduced_classes.txt', reduced_keys)

# Separate splits
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.2, shuffle = False)
to_file('splits/train.txt', x_train)
to_file('splits/test.txt', x_test)
reduced_x_train, reduced_x_test, reduced_y_train, reduced_y_test = model_selection.train_test_split(x,reduced_y, test_size=0.2, shuffle = False)
to_file('splits/reduced_train.txt', reduced_x_train)
to_file('splits/reduced_test.txt', reduced_x_test)


plot_splits('splits', 'stats', y_train, y_test, y, keys, 0.38)
plot_splits('reduced_splits', 'reduced_stats', reduced_y_train, reduced_y_test, reduced_y, reduced_keys, 0.25)


