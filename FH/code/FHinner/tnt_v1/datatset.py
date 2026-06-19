from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import torchvision.transforms


class MyDataSet(Dataset):
    def __init__(self,org,interp, label, ct):
        self.interp = interp
        self.label = label
        self.ct = ct
        self.org = org

    def __len__(self):
        return len(self.interp)

    def __getitem__(self, item):
        org = self.org[item]
        interp = self.interp[item]
        label = self.label[item]
        ct = self.ct[item]
        a=torch.max(org)#60
        b=torch.max(interp)
        c=torch.max(label)#52
        org = org /a
        interp = interp /b
        label = label /c
        return org, interp,label, ct

    @staticmethod
    def collate_fn(batch):
        org, interp,label, ct = tuple(zip(*batch))
        org = torch.stack(org, dim=0)
        interp = torch.stack(interp, dim=0)
        label = torch.stack(label, dim=0)
        ct = torch.stack(ct, dim=0)
        return org, interp,label, ct
