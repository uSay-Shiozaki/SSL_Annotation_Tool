import os
import sys
sys.path.append('/mnt/home/irielab/workspace/projects/my_research')
import numpy as np
import csv
import cv2
import xml.etree.ElementTree as ET

class getBboxesXml():
    
    def __init__(self, filePath):
        tree = ET.parse(filePath)
        self.root = tree.getroot()
    
    def getCoordData(self):
        root = self.root
        coords = ["xmin", "ymin", "xmax", "ymax"]
        arr = []
        for x in coords:
            for v in root.iter(x):
                arr.append(v.text)
        print("[xmin, ymin, xmax, ymax]")
        return arr
    
    def getImageSize(self):
        root = self.root
        coords = ["width", "height", "depth"]
        arr = []
        for x in coords:
            for v in root.iter(x):
                arr.append(v.text)
        print("[width, height, depth]")
        return arr

if __name__ == "__main__":
    path = "/mnt/media/irielab/win_drive/ImageNet/imagenet-object-localization-challenge-2012/ILSVRC/Annotations/CLS-LOC/val"
    files = os.listdir(path)
    getone = os.path.join(path,files[0])
    print(getBboxesXml(getone).getCoordData())
    print(getBboxesXml(getone).getImageSize())