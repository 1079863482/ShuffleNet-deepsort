import torch
import numpy as np
import os
from torch.utils import data
from PIL import Image
import cv2
import random
import torchvision

class Datasets(data.Dataset):
    def __init__(self,path,train = False):
        self.img_path = []
        self.label_data = []
        self.train = train
        self.transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop((128,64),padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transforms_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128,64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        count = 0
        for dir in os.listdir(path):
            for img_path in os.listdir(os.path.join(path,dir)):
                self.img_path.append(os.path.join(path,dir,img_path))
                self.label_data.append(count)
            count +=1

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img_path = self.img_path[index]
        img_data = Image.open(img_path)
        label = self.label_data[index]
        if self.train:
            img_data = self.transforms_train(img_data)
        else:
            img_data = self.transforms_test(img_data)

        return img_data,label


if __name__=="__main__":
    train_path = r"F:\BaiduNetdiskDownload\Market-1501-v15.09.15\pytorch\val"

    train_Data = Datasets(train_path,True)

    train = data.DataLoader(train_Data,batch_size=64,shuffle=False)

    for i,(x,y) in enumerate(train):
        print(x.size())
        print(y)
        break