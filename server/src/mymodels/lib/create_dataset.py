import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

from PIL import Image
import matplotlib.pyplot as plt
import json

class MakeDatasetfromJson():

    def __init__(self,label):
        self.DatasetFromJson = self.LoadDataPaths(label)

    def OpenJson(self, jsonPath):
        jsons = open(jsonPath, 'r')
        jsons = json.load(jsons)
        return jsons

    def LoadDataPaths(self, label: str):
        jsonFile = self.OpenJson
        return jsonFile[label]


    def __len__(self):
        return len(self.jsons)

class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.toTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)

class MnistImageDataset(data.Dataset):
    def __init__(self, fileList, transform):
        self.fileList = fileList
        self.transform = transform
    
    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, index):
        imagePath = self.fileList(index)
        image = Image.open(imagePath).convert('RGB')
        # '{0}/{1}/{2}/{3}'
        imageLabel = imagePath.split("/")[-1]
        # '{0}_{1}_{2}' format: label_indexNumber.png
        imageLabel = imageLabel.split("_")[0]
        imageTransformed = self.transform(image)
        return imageTransformed, imageLabel, imagePath

    