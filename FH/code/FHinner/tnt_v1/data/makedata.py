# import tensorflow as tf

import numpy as np
import os
import datetime

trainDataDir='.//data_new.npz'
val_sz=0
test_sz=20
c=2

# data = np.load(trainDataDir)
data = np.load(trainDataDir)
ct_label = np.expand_dims(data['ct_data'].astype('float32'), 3).transpose(2,1,0,3)  # CT域label
sin357 = data['sin357_data'].astype('float32').transpose(2,1,0)  # 正弦域label
sin605 = data['sin605_data'].astype('float32').transpose(2,1,0)  # 正弦域label
sin_input = np.expand_dims(sin357[:, 0::c, :], 3)

sin_label = np.expand_dims(sin605, 3)

print('shapes of ct_label:', ct_label.shape)
print('shape of sin_label:', sin_label.shape)
print('shape of sin_input:', sin_input.shape)

# 处理数据集，随机选取前dataset_sz-val_sz个作为数据并shuffle，剩下的val_sz个作为验证集
dataset_sz = sin_input.shape[0]
print(dataset_sz)
train_sz = 100
ids = np.random.permutation(dataset_sz)
train_ids = ids[0:train_sz]
val_ids = ids[train_sz:dataset_sz - test_sz]
val_data = [sin_input[val_ids], sin_label[val_ids], ct_label[val_ids]]
test_ids = ids[train_sz + val_sz:dataset_sz]
print(test_ids)

test_data = [sin357[test_ids], sin605[test_ids],ct_label[test_ids]]
np.savez('./test_new.npz', sin357=sin357[test_ids],sin605=sin605[test_ids], ct_label=ct_label[test_ids])
np.savez('./train_new.npz', sin357=sin357[train_ids], sin605=sin605[train_ids],ct_label=ct_label[train_ids])

# train_data = tf.data.Dataset.from_tensor_slices((sin_input[train_ids], sin_label[train_ids], ct_label[train_ids])). \
#     shuffle(dataset_sz - val_sz)

print('shape of val_data:', val_data[0].shape, val_data[1].shape, val_data[2].shape)