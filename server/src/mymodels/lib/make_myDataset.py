from re import M
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

from PIL import Image
import matplotlib.pyplot as plt
import json

import os
import sys
sys.path.append(os.pardir)
from sklearn import preprocessing


class MakeMNISTDataset(data.Dataset):
    '''
    this clas is for simulate data extraction for semi-learning on my app.
    just not for my app.
    '''
    
    def __init__(self, pathList, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform
        self.fileList = pathList
        self.encoded = self.label_encoding()

    def __len__(self):
        return len(self.fileList)
    
    def label_encoding(self):
        labels = []
        for path in self.fileList:
            labels.append(path.split("/")[-1].split("_")[0])
        set(labels)
        le = preprocessing.LabelEncoder()
        encoded = le.fit_transform(labels)
        encoded = torch.from_numpy(encoded)
        return encoded

    def __getitem__(self, index):
        path = self.fileList[index]
        mnistLabel = self.encoded[index]
        image = Image.open(path).convert('RGB')
        # '{0}/{1}/{2}/{3}'
        # imageLabel = imageLabel.split("_")[0]
        imageTransformed = self.transform(image)
        return imageTransformed, mnistLabel

class MakeImageNetDataset(data.Dataset):
    '''
    this clas is for simulate data extraction for semi-learning on my app.
    just not for my app.
    '''

    def __init__(self, pathList, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform
        self.fileList = pathList
        self.encoded = self.label_encoding()

    def __len__(self):
        return len(self.fileList)
    
    def label_encoding(self):
        labels = []
        for path in self.fileList:
            labels.append(path.split("/")[-1].split("_")[0])
        set(labels)
        le = preprocessing.LabelEncoder()
        encoded = le.fit_transform(labels)
        encoded = torch.from_numpy(encoded)
        self.w_id_table(encoded, le)
        return encoded
    
    def w_id_table(self, encoded, labelEncoder):
        print("Table of Label Encoder:")
        print(labelEncoder.classes_)
        
        f = open("./mycodes/data/id_table.txt", 'w')
        f.writelines(labelEncoder.classes_)
        f.close()
            
        

    def __getitem__(self, index):
        path = self.fileList[index]
        mnistLabel = self.encoded[index]
        image = Image.open(path).convert('RGB')
        # '{0}/{1}/{2}/{3}'
        # imageLabel = imageLabel.split("_")[0]
        imageTransformed = self.transform(image)
        return imageTransformed, mnistLabel


class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            #transforms.Resize(224),
            # transforms.RandomResizedCrop(224),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)

if __name__ == '__main__':
    path = "/mnt/media/irielab/win_drive/ImageNet/imagenet-object-localization-challenge-2012/ILSVRC/Data/CLS-LOC/10class_train_val/val"
    # path = "/mnt/media/irielab/win_drive/workspace/MNIST/raw/dataset/val"
    fullList = []
    dirs = os.listdir(path)
    for dir in dirs:
        for f in os.listdir(os.path.join(path,dir)):
            fullList.append(os.path.join(path,os.path.join(dir,f)))
    org_dataset = MakeMNISTDataset(fullList, transform=ImageTransform(0.485,0.228))
    
    train_dataloader = data.DataLoader(org_dataset, batch_size=32, shuffle=True)
    imgs, labels = iter(train_dataloader).next()
    print("image shape ==>", imgs[0].shape)

    pic = transforms.ToPILImage(mode='RGB')(imgs[0])
    plt.imshow(pic)
    print("Label is", labels[0])
    plt.show()
    