import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# import tensorflow as tf
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# class CTReconModel(tf.keras.Model):
#     def __init__(self):
#         super(CTReconModel, self).__init__()
#         # initialize  the sinLayers
#         self.sinModule = []
#         self.sinModule.append(
#             tf.keras.layers.Conv2D(32, 5, padding='valid', name='sine_conv1', activation='tanh'))
#         self.sinModule.append(
#             tf.keras.layers.Conv2D(32, 5, padding='valid', name='sine_conv2', activation='tanh'))
#         self.sinModule.append(
#             tf.keras.layers.Conv2D(32, 5, padding='valid', name='sine_conv3', activation='tanh'))
#         self.sinModule.append(
#             tf.keras.layers.Conv2D(1, kernel_size=1, padding='same', name='sine_conv4'))
#         self.cut_sz = 6  # (5-1)/2 + (5-1)/2 + (5-1)/2 + (1-1)/2
#
#     def inputModule(self, sin_in):
#         # extend input for sinLayer
#         axis1_ext_end = tf.gather(sin_in, range(0, self.cut_sz), axis=1)
#         axis1_ext_start = tf.gather(sin_in, range(sin_in.shape[1] - self.cut_sz, sin_in.shape[1]), axis=1)
#         sin_in_ex = tf.concat([axis1_ext_start, sin_in, axis1_ext_end], 1)
#         # linear interpolation
#         sin_interp = (sin_in + tf.gather(sin_in_ex, range(self.cut_sz + 1, self.cut_sz
#                                                           + sin_in.shape[1] + 1), axis=1)) / 2
#
#         axis2_ext = tf.zeros([sin_in_ex.shape[0], sin_in_ex.shape[1], self.cut_sz, 1], tf.float32)
#         return tf.concat([axis2_ext, sin_in_ex, axis2_ext], 2), sin_interp
#
#     def call(self, train_batch):  # 定义正向传播过程
#         # extend input for sinLayer
#         sin_in = train_batch
#         sin_in_ex, sin_interp = self.inputModule(sin_in)
#         # print('extended input: ', tf.shape(sin_in_ex))
#         # 正弦域网络
#         sin_out = self.sinModule[0](sin_in_ex)
#         sin_out = self.sinModule[1](sin_out)
#         sin_out = self.sinModule[2](sin_out)
#         sin_out = self.sinModule[3](sin_out)
#         sin_out = sin_interp + sin_out
#
#         # combine the input and output of the sinLayer
#         sin_map = tf.reshape(tf.concat([sin_in, sin_out], 2), [sin_in.shape[0], -1, sin_in.shape[2], 1])
#
#         return sin_map


class FbpLayer(nn.Module):
    def __init__(self, ):
        super(FbpLayer, self).__init__()
        # load AT, fbp_filter
        _rawAT = np.load('./model/My_AT.npz')
        indice = _rawAT['arr_0'].astype('int32')
        data = _rawAT['arr_1'].astype('float32')
        shape = _rawAT['arr_2']
        shape = (shape[0], shape[1])
        indice = list(indice.transpose())
        data = list(data)
        A = torch.sparse_coo_tensor(torch.tensor(indice), torch.tensor(data), shape)
        self.A_Matrix = A
        _out_sz = round(np.sqrt(float(self.A_Matrix.shape[0])))


        self.out_shape = (_out_sz, _out_sz)

        # FBP时使用的滤波器
        fbp_filter_weight = torch.tensor(_rawAT['arr_3'].astype('float32'))
        self.fbp_filter_weight = nn.Parameter(fbp_filter_weight.reshape(1, 1, 1, -1))
        self.fbp_filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(713, 1), stride=(1, 1), padding='same')
        self.fbp_filter.weight.data = self.fbp_filter_weight
        self.fbp_filter.bias.data = torch.tensor([0.])
        # self.fbp_filter_weight2 = tf.Variable(_rawAT['arr_3'].astype('float32').reshape(-1, 1, 1))

        self.scale = nn.Parameter(torch.tensor(10.0))  # scale for CT image
        self.bias = nn.Parameter(torch.tensor(0.0))


    def forward(self, sin_fan):
        sin_sz = sin_fan.shape[1] * sin_fan.shape[2] * sin_fan.shape[3]
        r = sin_fan.permute(0, 3, 1, 2)  # n,1,357,360
        sin_fan_flt = self.fbp_filter(r).permute(0, 2, 3, 1)  # n,360,357,1
        sin_fan_flt = torch.reshape(sin_fan_flt, [-1, sin_sz]).transpose(1, 0)
        fbpOut = torch.sparse.mm(self.A_Matrix, sin_fan_flt).transpose(1, 0)

        fbpOut = torch.reshape(fbpOut, [-1, self.out_shape[0], self.out_shape[1], 1])

        output = fbpOut * self.scale + self.bias
        return output


# def load_traindata(trainDataDir=r"./Data/mymodel/My_data_256_180.npz", val_sz=2, c=1):
#     # data = np.load(trainDataDir)
#     train_data = np.load(trainDataDir)
#     f_img = train_data['f_img'].astype('float32')  # 正弦域input
#     ct_label = train_data['ct_label'].astype('float32')  # 正弦域label

#     sin_input = np.expand_dims(f_img[:, 0::c, :], 3)
#     sin_label = np.zeros([f_img.shape[0], int(f_img.shape[1] / c), f_img.shape[2], c - 1])
#     for i in range(c - 1):
#         sin_label[:, :, :, i] = f_img[:, i + 1::c, :]

#     sin_label = sin_label.astype('float32')

#     print('shapes of ct_label, sin_label, :', ct_label.shape)
#     print('shape of sin_label:', sin_label.shape)
#     print('shape of sin_input:', sin_input.shape)

#     # 处理数据集，随机选取前dataset_sz-val_sz个作为数据并shuffle，剩下的val_sz个作为验证集
#     dataset_sz = sin_input.shape[0]
#     train_sz = dataset_sz - val_sz
#     ids = np.random.permutation(dataset_sz)

#     val_ids = ids[train_sz:dataset_sz]
#     val_data = [sin_input[val_ids], sin_label[val_ids], ct_label[val_ids]]

#     print('shape of val_data:', val_data[0].shape, val_data[1].shape, val_data[2].shape)
#     return val_data


# def load_data(trainDataDir="./Data/mymodel/My_data.npz", val_sz=2):
#     data = np.load(trainDataDir)
#     ct_label = np.expand_dims(data['arr_0'].astype('float32'), 3)  # CT域label
#     sin_label = np.expand_dims(data['arr_1'].astype('float32'), 3)  # 正弦域label
#     sin_input = np.expand_dims(data['arr_2'].astype('float32'), 3)  # 正弦域input
#     # sin_input = sin_input[:, 6:186, 6:363, :]  # !!!!!!!!!!!!!!!!!!!!!!!!!!YB!!!!!!!!!!!!!!!!!!!

#     print('shapes of ct_label, sin_label, :', ct_label.shape)
#     print('shape of sin_label:', sin_label.shape)
#     print('shape of sin_input:', sin_input.shape)

#     # 处理数据集，随机选取前dataset_sz-val_sz个作为数据并shuffle，剩下的val_sz个作为验证集
#     dataset_sz = sin_input.shape[0]
#     train_sz = dataset_sz - val_sz
#     ids = np.random.permutation(dataset_sz)
#     train_ids = ids[0:train_sz]
#     val_ids = ids[train_sz:dataset_sz]
#     val_data = [sin_input[val_ids], sin_label[val_ids], ct_label[val_ids]]

#     print('shape of val_data:', val_data[0].shape, val_data[1].shape, val_data[2].shape)
#     return val_data


# val_data = load_data(r"D:\毕设\CT重建\CT_tf\Data\mymodel\My_data.npz")
#
# sin_in, sin_label, ct_label = val_data
# x = sin_label
#
# model1 = CTReconModel()
# model1.build((1, 180, 357, 1))
# model1.load_weights("D:\毕设\CT重建\CT_tf\model.h5", by_name=True, skip_mismatch=True)
# y = model1(x).numpy()
#
# y = torch.tensor(y)
# fbpModule = FbpLayer()
# y = fbpModule(y)
# z = y[0].detach().numpy()
# plt.imshow(z)
# plt.show()

