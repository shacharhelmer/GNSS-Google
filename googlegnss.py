
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gmplot
from IPython.display import IFrame
import tensorflow as tf

df = pd.read_csv('google-smartphone-decimeter-challenge/train/2020-05-14-US-MTV-1/Pixel4/Pixel4_derived.csv', nrows=1)
derived = df.drop(0)
df = pd.read_csv('google-smartphone-decimeter-challenge/train/2020-05-14-US-MTV-1/Pixel4/ground_truth.csv', nrows=1)
ground = df.drop(0)

for dirname, _, filenames in os.walk('google-smartphone-decimeter-challenge/train'):
    for filename in filenames:
        if 'derived.csv' in filename:
            r = os.path.join(dirname, filename)
            derived = derived.append(pd.read_csv(os.path.join(dirname, filename)))
        elif 'ground_truth.csv' in filename:
            ground = ground.append(pd.read_csv(os.path.join(dirname, filename)))

bsln_trn = pd.read_csv('google-smartphone-decimeter-challenge/baseline_locations_train.csv')
bsln_tst = pd.read_csv('google-smartphone-decimeter-challenge/baseline_locations_test.csv')
target = ground.merge(bsln_trn, how='inner', on=['collectionName', 'phoneName', 'millisSinceGpsEpoch'], suffixes=('_grnd', '_bsln'))
target['latDiff'] = target['latDeg_grnd']-target['latDeg_bsln']
target['lngDiff'] = target['lngDeg_grnd']-target['lngDeg_bsln']
fig = plt.figure(figsize=(10,5))
plt.subplot(121); plt.title('Baseline vs. ground truth latDeg'); plt.plot(target['latDeg_grnd'], target['latDeg_bsln'], color='r')
plt.subplot(122); plt.title('Baseline vs. ground truth lngDeg'); plt.plot(target['lngDeg_grnd'], target['lngDeg_bsln'], color='b')
target[['latDiff', 'lngDiff']].describe()