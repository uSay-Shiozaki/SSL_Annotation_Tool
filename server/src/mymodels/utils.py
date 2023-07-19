import pandas as pd
import matplotlib.pyplot as plt
import os


def getFiles(path):
  files = []
  for v in os.listdir(path):
    fileList = os.listdir(os.path.join(path, v))
    for file in fileList:
        path = os.path.join(os.path.join(path, v), file)
        files.append(path)
  return files

