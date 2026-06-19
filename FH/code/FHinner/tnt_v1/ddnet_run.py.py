#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import torchvision.transforms


class MyDataSet(Dataset):
    def __init__(self,sin_in, label):
        self.sin_in = sin_in
        self.label = label


    def __len__(self):
        return len(self.sin_in)

    def __getitem__(self, item):

        sin_in = self.sin_in[item]
        label = self.label[item]

        sin_in = sin_in
        label = label
        return sin_in,label

    @staticmethod
    def collate_fn(batch):
        sin_in,label = tuple(zip(*batch))
        sin_in = torch.stack(sin_in, dim=0)
        label = torch.stack(label, dim=0)
        return sin_in,label


# In[2]:


import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
# import sys
from tqdm import tqdm
import os

def reshape(x):
    B,channel, angle, sensor = x.shape
    x_end=x[:,0,:,:]
    for i in range(channel-1):
        x_end=torch.cat((x_end,x[:,i+1,:,:]),dim=2)
    x_end=torch.reshape(x_end,(B,angle*channel,sensor))
    return x_end


def inter(data):
    n,angel,sensor = data.shape
    org = data[:, 0::c, :]
    sample_angle=int(360/c)
    
    ex=torch.cat((org,org[:,0,:].unsqueeze(1)),dim=1)
    sin_in = torch.zeros((n,c,int(360/c),sensor))

    for i in range(c):
        if i ==0:
            sin_in[:,i,:,:]=data[:,i::c,:]
        else:
            sin_in[:,i,:,:]=((c-i)*org+(i)*ex[:,1:,:])/c

    return sin_in

def load_data(trainDataDir="./Data/mymodel/My_data.npz"):
    data = np.load(trainDataDir)
    
    sine357 = torch.tensor(data['sin357'])
    sin_in = reshape(inter(sine357))
    # sin_in = inter(sine357)
    ct = torch.tensor(data['ct_label']).permute(0,3,1,2)
    data_set = MyDataSet(sin_in, ct)
    return data_set

c=4
print("开始")
train_data = load_data("./data/train_meiaonew.npz")
val_data = load_data("./data/test_meiaonew.npz")
print("数据初始化完成")


# In[3]:


batch_size = 3
train_dataset = torch.utils.data.DataLoader(train_data,
                                            batch_size=batch_size,
                                            shuffle=False,  # 打乱顺序
                                            pin_memory=True,  # 写入内存
                                            # num_workers=nw,
                                            collate_fn=train_data.collate_fn)  # 解包？

val_dataset = torch.utils.data.DataLoader(val_data,
                                          batch_size=batch_size,
                                          shuffle=False,  # 打乱顺序
                                          pin_memory=True,  # 写入内存
                                          # num_workers=nw,
                                          collate_fn=val_data.collate_fn)  # 解包？


# In[4]:


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[5]:


class FbpLayer(nn.Module):
    def __init__(self, ):
        super(FbpLayer, self).__init__()
        # load AT, fbp_filter
        _rawAT = np.load('./model/A_new.npz')
        indice = _rawAT['indice'].astype('int32')
        data = _rawAT['data'].astype('float32')
        self.cos = torch.tensor(_rawAT['cos'].astype('float32').transpose()).to(device)
        shape = (65536, 128520)
        shape = (shape[0], shape[1])
        indice = torch.tensor(indice.transpose())
        data = torch.tensor(data).reshape(-1)

        data=np.array(data)
        indice=np.array(indice)
        
        A = torch.sparse_coo_tensor(indice, data, shape)
        self.A_Matrix = A.to(device)

        _out_sz = round(np.sqrt(float(self.A_Matrix.shape[0])))

        self.out_shape = (_out_sz, _out_sz)

        # FBP时使用的滤波器
        fbp_filter_weight = torch.tensor(_rawAT['filt'].astype('float32')).to(device)
        self.fbp_filter_weight = nn.Parameter(fbp_filter_weight.reshape(1, 1, 1, -1)).to(device)
        self.fbp_filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(713, 1), stride=(1, 1), padding='same')
        self.fbp_filter.weight.data = self.fbp_filter_weight
        self.fbp_filter.bias.data = torch.tensor([0.])

#         self.scale = nn.Parameter(torch.tensor(10.0))  # scale for CT image
#         self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, sin_fan):
        sin_fan = sin_fan.unsqueeze(1)
        sin_sz = sin_fan.shape[1] * sin_fan.shape[2] * sin_fan.shape[3]
        r = sin_fan
        r = r * self.cos
#         print(torch.sum(self.cos))
#         plt.imshow(r[0,0,:,:].cpu().detach().numpy())
#         plt.show()
        sin_fan_flt = self.fbp_filter(r.to(device)).permute(0, 2, 3, 1)  # n,360,357,1
#         plt.imshow(sin_fan_flt[0,:,:,0].cpu().detach().numpy())
#         plt.show()
#         sin_fan_flt = torch.reshape(sin_fan_flt.to(device), [sin_sz, -1])
#         fbpOut = torch.sparse.mm(self.A_Matrix, sin_fan_flt)

#         fbpOut = torch.reshape(fbpOut, [-1, self.out_shape[0], self.out_shape[1],1])
        
        sin_fan_flt = torch.reshape(sin_fan_flt, [-1, sin_sz]).transpose(1, 0)
        fbpOut = torch.sparse.mm(self.A_Matrix, sin_fan_flt).transpose(1, 0)
        fbpOut = torch.reshape(fbpOut, [-1, self.out_shape[0], self.out_shape[1], 1])
        
        # fbpOut = torch.reshape(fbpOut, [-1, 1, self.out_shape[0], self.out_shape[1]])

        # output = fbpOut * self.scale + self.bias
        fbpOut = fbpOut.clamp(0, 1)
        return fbpOut


# In[6]:


class fp(nn.Module):
    def __init__(self):
        super(fp, self).__init__()
        a = np.load('./weights/index_fpnew.npy').transpose((1, 0))
        b = list(a)
        data = np.load('./weights/data_fpnew.npy')
        
        data=np.array(data)
        b=np.array(b)

        shape = (128520, 65536)
        self.A = torch.sparse_coo_tensor(torch.tensor(b), torch.tensor(data), shape).to(device)
        
    def forward(self,I):
        b, c, _, _ = I.shape
#         I=torch.rot90(I,1,[2,3])
        img = I.reshape(-1, 65536).permute(1, 0)
        sinout = torch.sparse.mm(self.A.float(), img.float()).reshape(360, 357, -1).permute(2, 0, 1)
       
        return sinout




# In[7]:


from Unet import U_Net
angle = int(360/c)
class mymodel(nn.Module):
    def __init__(self, ):
        super(mymodel, self).__init__()
        
        self.sin = U_Net(in_ch=1, out_ch=1)
        self.ct = U_Net()
        self.fbp = FbpLayer()
        self.act = nn.ReLU()
        self.fp=fp()

    def forward(self, x):
        ct_old=self.fbp(x).permute(0,3,1,2)
        sin_new=self.sin(x.unsqueeze(1))
        ct_new=self.fbp(sin_new.squeeze(1))
        ct_new=self.ct(ct_new.permute(0,3,1,2))
        ct_pre=ct_new+ct_old


        return ct_pre


# In[8]:


model = mymodel().to(device)


# In[9]:


weights = r''
# weights = r"./weights/ct_dual_pos{}.pth".format(c)
# weights = r'./weights/cat_{}_pos.pth'.format(c)
if weights != "":
    weights_dict = torch.load(weights, map_location=device)
    print(model.load_state_dict(weights_dict, strict=False))
    
# weights = r'./weights/ct.pth'
# if weights != "":
#     weights_dict = torch.load(weights, map_location=device)
#     print(model.load_state_dict(weights_dict, strict=False))
    


# In[10]:


for p in model.fbp.parameters():
    p.requires_grad=False
# for p in model.sin.parameters():
#     p.requires_grad=False
# for p in model.ct.parameters():
#     p.requires_grad=False

params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params,
#                             lr=0.1,
#                             momentum=0.9)

optimizer = torch.optim.Adam(params,
                            lr=0.0001)


# learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=10,
                                               gamma=0.33)


# In[11]:


def computeloss(predict, target):
    n,c,x,y=predict.shape
    psrn_all = 0
    for i in range(n):
        img1 = predict[i].cpu()
        img2 = target[i].cpu()

        ma=torch.max(img2)-torch.min(img2)
        psnr_pix = sum(sum(sum((img1 - img2) ** 2)))/(x*y)
        psrn_all += 10 * np.log((ma**2) / psnr_pix.cpu().detach().numpy())/np.log(10)
    return psrn_all/n


# In[12]:


import ssim

for i in range(50):

    loss_all = 0.
    model.train()
    # train_dataset = tqdm(train_dataset, file=sys.stdout)
    for step, data in enumerate(train_dataset):
        sin_in,label= data 
        ct= model(sin_in.to(device).to(torch.float32))

        loss = nn.MSELoss()(ct, label.to(device).to(torch.float32))
        
        loss_all+=loss
        ct_psrn = computeloss(ct,label.to(device))
        
        loss_all = loss+loss_all
        # train_dataset.desc = "epoch:{},loss: {:.3f},ct_psrn: {:.3f} ".format(
        #     i, loss_all,ct_psrn)
        
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            # sys.exit(1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
#     print(ct.shape,label.shape)
#     print(torch.max(ct),torch.min(ct),torch.max(label))
    
    plt.subplot(1, 2, 1)
    plt.title("ct", loc='center')
    plt.imshow(ct.cpu().detach().numpy()[0][0])
    plt.subplot(1, 2, 2)
    plt.title("label", loc='center')
    plt.imshow(label.cpu()[0][0])
    plt.show()
    if loss<1:
        torch.save(model.state_dict(), "./weights/ddnet_c{}_best.pth".format(c))

    with torch.no_grad():
        model.eval()
        ct_psnr=0
        ct_ssim=0
       
        for step, data in enumerate(val_dataset):
            sin_in,label= data 
            ct= model(sin_in.to(device).to(torch.float32))
            
            ct = ct.clamp(0, 1)
            ct_psnr += computeloss(ct,label.to(device))
            ct_ssim += ssim.ssim(ct,label.to(device))
            # plt.subplot(1, 2, 1)
            # plt.title("ct", loc='center')
            # plt.imshow(ct.cpu().detach().numpy()[0][0])
            # plt.subplot(1, 2, 2)
            # plt.title("label", loc='center')
            # plt.imshow(label.cpu()[0][0])
            # plt.show()
   
        step =step+1
        print("ct_psnr： ",ct_psnr/step)
        print("ct_ssim： ",ct_ssim/step)
        
#         ct_psnr = ct_psnr/step 
#         if ct_psnr>34 and ct_psnr<35:
#             torch.save(model.state_dict(), "./weights/ct_predict_{}_3479.pth".format(c))
#         if ct_psnr>35 and ct_psnr<36:
#             torch.save(model.state_dict(), "./weights/ct_predict_{}_3578.pth".format(c))
#         if ct_psnr>36 and ct_psnr<37:
#             torch.save(model.state_dict(), "./weights/ct_predict_{}_3671.pth".format(c))
#         if ct_psnr>37 and ct_psnr<38:
#             torch.save(model.state_dict(), "./weights/ct_predict_{}_3797.pth".format(c))
#         if ct_psnr>38 :
#             torch.save(model.state_dict(), "./weights/ct_predict_{}_3962.pth".format(c))

#         plt.subplot(1, 2, 1)
#         plt.title("ct", loc='center')
#         plt.imshow(ct.cpu().detach().numpy()[0][0])
#         plt.subplot(1, 2, 2)
#         plt.title("label", loc='center')
#         plt.imshow(label.cpu()[0][0])
#         plt.show()
        


# In[ ]:




