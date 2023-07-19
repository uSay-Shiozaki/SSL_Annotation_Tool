import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import logging

pathToCkp = "/mnt/media/irielab/ubuntu_data/workspace/swav_checkpoints"
pathToLogs = "IN2012_logs/tensorboard_logs/IN2012_10class_random_200ep_seed251"
pathToFolder = os.path.join(pathToCkp, pathToLogs)
logFolders = ["log_0.01", "log_0.05" , "log_0.1", "log_0.5", "log_1"]

def splitRatio(folderName):
    ratio = float(folderName.split("_")[1])
    return ratio

def getFullPathList(pathToFolder, logFolders):
    res = []
    for f in logFolders:
        ratio = splitRatio(f)
        path = os.path.join(pathToFolder, f)
        path = os.path.join(path, f"val_{ratio}_paths.json")
        res.append(path)
    return res

def loadJson(targetJsonPath):
    with open(targetJsonPath, 'r') as f:
        res = json.load(f)
    return res

def getTargetLabel(path):
    label = path.split("/")[-1]
    label = label.split("_")[0]
    return label

def countLabel(pathList):
    countMap = {}
    for path in pathList:
        label = getTargetLabel(path)
        if label in countMap.keys():
            countMap[label] += 1
        else:
            countMap[label] = 1
    return countMap

def mapKeySort(map):
    res = {}
    sortedList = sorted(map.items())
    for (key, value) in sortedList:
        res[key] = value
    return res

def encodeMapKeys(map):
    res = []
    for i in range(len(map.keys())):
        res.append(i)
    return res

def makeBar(map):
    x = encodeMapKeys(map)
    labels = map.keys()
    values = map.values()

    return x, labels, values

def main():
    fullPathList = getFullPathList(pathToFolder, logFolders)
    for jsonPath in fullPathList:
        jsonFile = loadJson(jsonPath)
        print(f"input data length is {len(jsonFile['paths'])} ")
        countMap = countLabel(jsonFile['paths'])
        countMap = mapKeySort(countMap)

        x, labels, values = makeBar(countMap)

        plt.bar(x, values)
        plt.xticks(x, labels, rotation=90)
        
        plt.show()
        
if __name__ == "__main__":
    main()
