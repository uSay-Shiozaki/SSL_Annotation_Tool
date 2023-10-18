import glob
import random
import os
import shutil
import math

INPUT_DIR = "/mnt/media/irielab/ubuntu_data/datasets/my_10class_ImageNet2012/val"
OUTPUT_DIR = "/mnt/media/irielab/ubuntu_data/datasets/my_10class_ImageNet2012/val0.1"
# INPUT_DIR = "/mnt/media/irielab/win_drive/workspace/MNIST/raw/dataset/val"
# OUTPUT_DIR = "/mnt/media/irielab/win_drive/workspace/MNIST/raw/dataset/val0.005"

SAMPLING_RATIO = 0.1

def file_list(dir, target):
    return os.listdir(os.path.join(dir,target))

def random_sample_file():
    print("get dirs")
    # print(os.listdir(INPUT_DIR))
    dirs = [f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f))]
    # print(dirs)

    #クラスフォルダ複製

    for dir in dirs:
        if not (os.path.isdir(os.path.join(OUTPUT_DIR, dir))):
            os.makedirs(OUTPUT_DIR + '/' + dir)
            print("make dirs")
            
    
    
    for dir in dirs:
        print("dir", dir)
        random_sample_file = random.sample(file_list(INPUT_DIR, dir), \
            math.ceil(len(file_list(INPUT_DIR, dir))*SAMPLING_RATIO))
        # print("random_sample", random_sample_file)
        dir_path = os.path.join(INPUT_DIR,dir)
        for file in random_sample_file:
            shutil.copy2(os.path.join(dir_path, file), OUTPUT_DIR + '/' + dir + '/')
    print("DONE")
if __name__ == '__main__':
    #print(os.listdir('/mnt/media/irielab/HDD(3TB)/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train'))
    random_sample_file()