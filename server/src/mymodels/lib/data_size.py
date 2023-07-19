import os
import sys
import json
import logging
logging.basicConfig(level=logging.DEBUG)
# basePath = '/mnt/media/irielab/win_drive/workspace/MNIST/raw/dataset/'
basePath = "/mnt/media/irielab/win_drive/ImageNet/imagenet-object-localization-challenge-2012/ILSVRC/Data/CLS-LOC/10class_train_val"
folderNames = ['val0.002', 'val0.004', 'val0.008', 'val0.012','val0.016', 'val0.024','val0.032', 'val0.04', 'val0.05']
def main():
    for f in folderNames:
        path = os.path.join(basePath, f)
        list = os.listdir(path)
        cnt = 0
        for dir in list:
            newPath = os.path.join(path, dir)
            files = os.listdir(newPath)
            res = len(files)
            cnt += res
        logging.info(f"Total amount is {cnt} in {f}")
            


if __name__ == '__main__':
    main()