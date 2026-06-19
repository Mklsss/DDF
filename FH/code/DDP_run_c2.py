#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import torchvision.transforms
import ssim

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
import sys
from tqdm import tqdm
import os
import swanlab

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

c=2
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
                                          batch_size=2,
                                          shuffle=False,  # 打乱顺序
                                          pin_memory=True,  # 写入内存
                                          # num_workers=nw,
                                          collate_fn=val_data.collate_fn)  # 解包？


# In[3]:


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[4]:


def reshape(x):
    B,channel, angle, sensor = x.shape
    x_end=x[:,0,:,:]
    for i in range(channel-1):
        x_end=torch.cat((x_end,x[:,i+1,:,:]),dim=2)
    x_end=torch.reshape(x_end,(B,angle*channel,sensor))
    return x_end


# In[5]:


import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Attention_pos(nn.Module):
    def __init__(self, dim, num_heads=7, qkv_bias=False, qk_scale=None, attn_drop_ratio=0.1, proj_drop_ratio=0.1):
        super(Attention_pos, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk = nn.Linear(dim * 4, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim )
        self.proj1 = nn.Linear(dim, dim * 3)
        self.proj2 = nn.Linear(dim * 3, dim)
        self.normal1 = nn.LayerNorm(357, eps=1e-6)
        self.normal2 = nn.LayerNorm(357,eps=1e-6)
        
        beta = torch.squeeze(torch.tensor(range(-178, 179, 1)) / 360 * torch.pi)
        beta = beta.repeat(360, 1)
        self.beta = torch.atan(beta).to(device)
        
        
        angle = torch.squeeze(torch.tensor(range(0, 360, 1)) / 360 * torch.pi)
        angle = angle.repeat(357, 1)
        angle = torch.sin(angle)
        self.angle = torch.rot90(angle).to(device)

        
        self.mask = torch.zeros(360,357).to(device)
        self.mask[0::c, :] = 1
        
    def forward(self, x1):
        B, N, C = x1.shape

        beta=self.beta.repeat(B, 1, 1).to(device)
        angle = self.angle.repeat(B, 1, 1).to(device)
        mask = self.mask.repeat(B,1,1).to(device)
        
        
        x2 = self.normal1(x1)
        qk = torch.cat((x2, angle,beta, mask), dim=2)
#         plt.imshow(qk[0].cpu().detach().numpy())
#         plt.show()
        qk = self.qk(qk).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 2,3,7,360,51
        q, k = qk[0], qk[1]
        v = self.v(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x2 = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x2 = self.proj(x2)      
        x2 = x2+x1

        return x2


# mode = Attention(357)
# data = torch.rand((3, 360, 357))
# y=mode(data,4)
# print(y)


# In[6]:


import torch
import os
import swanlab
from functools import partial
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Attention(nn.Module):
    def __init__(self, dim, num_heads=7, qkv_bias=False, qk_scale=None, attn_drop_ratio=0.1, proj_drop_ratio=0.1):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim )
        self.proj1 = nn.Linear(dim, dim * 3)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(dim * 3, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.normal1=nn.LayerNorm(357,eps=1e-6)
        self.normal2=nn.LayerNorm(357,eps=1e-6)

        
    def forward(self, x1):
        B, N, C = x1.shape
        x2=self.normal1(x1)
        qkv = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)          
        x2 = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x2 = self.proj(x2)      
        
        x2 = x2+x1
        x = self.normal2(x2)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = x+x2
        return x


class sin_angle(nn.Module):
    def __init__(self, num_sensor, angle, num_heads=7):  # num_sensor每个角度有多少传感器,angle为被采样之后的°
        super().__init__()
        
        self.attn_pos = Attention_pos(num_sensor, num_heads=num_heads)
        self.attn1 = Attention(num_sensor, num_heads=num_heads)
        self.attn2 = Attention(num_sensor, num_heads=num_heads)
        self.attn3 = Attention(num_sensor, num_heads=num_heads)
        self.attn4 = Attention(num_sensor, num_heads=num_heads)
        
        self.act=nn.ReLU()



    def forward(self, x_in):


        
        x_i = x_in
        x_i = self.attn_pos(x_i)
        x_i = self.attn1(x_i)
        x_i = self.attn2(x_i)
        x_i = self.attn3(x_i)
        x_i = self.attn4(x_i)+x_in
        x_i = self.act(x_i)
#         for i in range(self.sample-1):
#             x_end[:,i,:,:] = x_i[:,i+1::self.sample,:]
         
        return x_i  


# In[7]:


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


# In[8]:


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




# In[9]:


class gmlp(nn.Module):
    def __init__(self):
        super(gmlp, self).__init__()
        
        self.con1= Attention(357, num_heads=1)
        self.con2= Attention(357, num_heads=1)
        self.act=nn.GELU()
        self.con3 = Attention(357, num_heads=1)
        
    def forward(self,x1,x2):
        x1=x1.squeeze()
        x2=x2.squeeze()
        
        g=self.con1(x1)
        g=self.act(g)
        
        x2=self.con2(x2)
        x2=x2*g
        x=self.con3(x2)
        return x


# In[10]:


# from model .maxim import MAXIM_dns_3s
from nafnet.NAFNet_arch import NAFNet
from cgb import CrossGatingBlock
angle = int(360/c)
class mymodel(nn.Module):
    def __init__(self, ):
        super(mymodel, self).__init__()
        
        self.sin = sin_angle(num_sensor=357, angle=angle,num_heads=1)
        self.fbp = FbpLayer()
        self.act = nn.ReLU()
        self.fp=fp()
        self.gmlp=gmlp()

        #####nafnet######
        img_channel = 1
        width = 32
        enc_blks = [1, 1, 1, 28]
        middle_blk_num = 1
        dec_blks = [1, 1, 1, 1]
        self.ct = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

        self.fus_ct1 = CrossGatingBlock()
        

    def forward(self, x):
#         print(x)
#         print("*****************************************************************************************************************")
#         plt.imshow(x[0].cpu().detach().numpy())
#         plt.show()
        sin1 = self.sin(x)
#         print(torch.sum(sin1))
#         print("*****************************************************************************************************************")
#         plt.imshow(sin1[0].cpu().detach().numpy())
#         plt.show()
        fbp1 = self.fbp(sin1).permute(0, 3, 1, 2)  
        
#         print(fbp1)
#         print("*****************************************************************************************************************")
#         plt.imshow(fbp1[0][0].cpu().detach().numpy())
#         plt.show()        
        ct1 = self.ct(fbp1)
#         print(ct1)
#         print("*****************************************************************************************************************")
#         plt.imshow(ct1[0][0].cpu().detach().numpy())
#         plt.show()    
        sin_new = self.fp(ct1).unsqueeze(1).to(device)
#         print(sin_new)
#         print("*****************************************************************************************************************")
#         plt.imshow(sin_new[0][0].cpu().detach().numpy())
#         plt.show()            
#         sin_new = torch.cat((sin_new,sin1.unsqueeze(1)),dim=1)
#         sin2 = self.fus_sin1(sin_new)
        sin2=self.gmlp(sin_new,sin1.unsqueeze(1))
    
#         fbp2 = self.fbp(sin2.squeeze(1)).permute(0, 3, 1, 2)
        
        fbp2 = self.fbp(sin2.squeeze(1)).permute(0, 3, 1, 2)  
        
        ct2, _ = self.fus_ct1(ct1, fbp2)
        
#         ct = self.out(ct2)
#         plt.imshow(ct[0][0].cpu().detach().numpy())
#         plt.show()     
#         ct_out = self.ct2(ct)
#         plt.imshow(ct[0][0].cpu().detach().numpy())
#         plt.show()   

        return ct2


# In[11]:


model = mymodel().to(device)


# In[12]:


# from thop import profile

input = torch.randn(3, 360, 357).to(device)
# macs, params = profile(model, inputs=(input, ))
# print(' FLOPs: ', macs*2)   # 一般来讲，FLOPs是macs的两倍
# print('params: ', params)


# In[13]:


a=torch.rand(2,360,357).to(device)

import time 
s=time.time()
for i in range(0):
    x=model(a)
e=time.time()
# print(e-s)


# In[15]:


weights = r''
# weights = r'./weights/ct_predict_8_3671.pth'
# weights = r'./weights/FTRcat_{}.pth'.format(c)
resume_ckpt = r'./weights/DDF_c2_ckpt.pth'
if resume_ckpt != "":
    ckpt = torch.load(resume_ckpt, map_location=device)
    print(model.load_state_dict(ckpt["model"], strict=False))
    psnrmax = ckpt.get("best_psnr", 0)
    
# weights = r'./weights/ct.pth'
# if weights != "":
#     weights_dict = torch.load(weights, map_location=device)
#     print(model.load_state_dict(weights_dict, strict=False))
    


# In[16]:


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
swanlab.init(project="DDF-reproduction", experiment_name="DDF_c2_continue_500ep", config={"method": "DDF", "c": c, "epochs": 500, "batch_size": batch_size, "lr": 0.0001, "resume": "weights/DDF_c2_ckpt.pth"})
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=10,
                                               gamma=0.33)


# In[17]:


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


# In[ ]:


psnrmax = locals().get("psnrmax", 0)
for i in range(500):

    
    loss_all = 0.
    model.train()
    train_dataset = tqdm(train_dataset, file=sys.stdout)
    for step, data in enumerate(train_dataset):
        sin_in,label= data 
#         print(sin_in.shape)
        ct= model(sin_in.to(device).to(torch.float32))

        loss = nn.MSELoss()(ct, label.to(device).to(torch.float32))
        
        loss_all+=loss
        ct_psrn = computeloss(ct,label.to(device))
        train_dataset.desc = "epoch:{},loss: {:.3f},ct_psrn: {:.3f} ".format(
            i, loss_all,ct_psrn)
        
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        model.eval()
        ct_psnr=0
        ct_ssim=0
        print(ct_psnr)
        for step, data in enumerate(val_dataset):
            sin_in,label= data 
            ct= model(sin_in.to(device).to(torch.float32))
            ct = ct.clamp(0, 1)
            ct_psnr += computeloss(ct,label.to(device))
            ct_ssim += ssim.ssim(ct,label.to(device))

        step=step+1
        ct_psnr=ct_psnr/step
        ct_ssim=ct_ssim/step
        print("x1： ",ct_psnr)
        print("ct_ssim： ",ct_ssim)
        swanlab.log({"val_psnr": float(ct_psnr), "val_ssim": float(ct_ssim), "epoch": i}, step=i)
        if ct_psnr>psnrmax:
            psnrmax=ct_psnr
            print("变化")
            torch.save(model.state_dict(), "./weights/DDF_c{}_best.pth".format(c))
            torch.save({
                "epoch": i,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_psnr": psnrmax,
            }, "./weights/DDF_c{}_ckpt.pth".format(c))
        
        


# In[ ]:


# with torch.no_grad():
#     model.eval()
#     ct_psnr=0
#     ct_ssim=0
#     print(ct_psnr)
#     for step, data in enumerate(val_dataset):
#         sin_in,label= data 
#         ct,sin1= model(sin_in.to(device).to(torch.float32))
#         print(torch.sum(sin1))

#         ct = ct.clamp(0, 1)
#         ct_psnr += computeloss(ct,label.to(device))
#         ct_ssim += ssim.ssim(ct,label.to(device))

#         plt.imshow(ct.cpu().detach().numpy()[0][0],cmap = 'gray')
#         plt.axis('off')   # 去坐标轴
#         plt.xticks([])    # 去 x 轴刻度
#         plt.yticks([])    # 去 y 轴刻度
#         plt.savefig('./predict_img/2/37/{}.png'.format(step), dpi=600)

# #         plt.imshow(label.cpu()[0][0],cmap = 'gray')
# #         plt.axis('off')   # 去坐标轴
# #         plt.xticks([])    # 去 x 轴刻度
# #         plt.yticks([])    # 去 y 轴刻度
# #         plt.savefig('./predict_img/org/{}.png'.format(step), dpi=600)
# #         plt.show()
        
#         plt.imshow(label.cpu()[0][0]-ct.cpu().detach().numpy()[0][0],cmap = 'gray')
#         plt.axis('off')   # 去坐标轴
#         plt.xticks([])    # 去 x 轴刻度
#         plt.yticks([])    # 去 y 轴刻度
#         plt.savefig('./predict_img/2/37/{}_cha.png'.format(step), dpi=600)
#         plt.show()
#     print(ct_psnr,step)     
#     step =step+1
#     print("ct_psnr： ",ct_psnr/step)
#     print("ct_ssim： ",ct_ssim/step)
        


# In[ ]:




