import os
import shutil
import random


def image_dir_train_test_sprit(original_dir, base_dir, train_size=0.7):
    '''
    画像データをトレインデータとテストデータにシャッフルして分割します。フォルダもなければ作成します。

    parameter
    ------------
    original_dir: str
      オリジナルデータフォルダのパス その下に各クラスのフォルダがある
    base_dir: str
      分けたデータを格納するフォルダのパス　そこにフォルダが作られます
    train_size: float
      トレインデータの割合
    '''
    try:
        os.mkdir(base_dir)
    except FileExistsError:
        print(base_dir + "は作成済み")

    #クラス分のフォルダ名の取得
    dir_lists = os.listdir(original_dir)
    dir_lists = [f for f in dir_lists if os.path.isdir(os.path.join(original_dir, f))]
    original_dir_path = [os.path.join(original_dir, p) for p in dir_lists]

    num_class = len(dir_lists)

    # フォルダの作成(トレインとバリデーション)
    try:
        train_dir = os.path.join(base_dir, 'val')
        os.mkdir(train_dir)
    except FileExistsError:
        print(train_dir + "は作成済み")

    try:
        validation_dir = os.path.join(base_dir, 'test')
        os.mkdir(validation_dir)
    except FileExistsError:
        print(validation_dir + "は作成済み")

    #クラスフォルダの作成
    train_dir_path_lists = []
    val_dir_path_lists = []

    for D in dir_lists:
        train_class_dir_path = os.path.join(train_dir, D)
        try:
            os.mkdir(train_class_dir_path)
        except FileExistsError:
            print(train_class_dir_path + "は作成済み")
        train_dir_path_lists += [train_class_dir_path]
        val_class_dir_path = os.path.join(validation_dir, D)
        try:
            os.mkdir(val_class_dir_path)
        except FileExistsError:
            print(val_class_dir_path + "は作成済み")
        val_dir_path_lists += [val_class_dir_path]


    #元データをシャッフルしたものを上で作ったフォルダにコピーします。
    #ファイル名を取得してシャッフル
    for i,path in enumerate(original_dir_path):
        files_class = os.listdir(path)
        random.shuffle(files_class)
        # 分割地点のインデックスを取得
        num_bunkatu = int(len(files_class) * train_size)
        #トレインへファイルをコピー
        for fname in files_class[:num_bunkatu]:
            src = os.path.join(path, fname)
            dst = os.path.join(train_dir_path_lists[i], fname)
            shutil.copyfile(src, dst)
        #valへファイルをコピー
        for fname in files_class[num_bunkatu:]:
            src = os.path.join(path, fname)
            dst = os.path.join(val_dir_path_lists[i], fname)
            shutil.copyfile(src, dst)
        print(path + "コピー完了")

    print("分割終了")


def main():
    original_dir = '/mnt/media/irielab/ubuntu_data/datasets/my_10class_ImageNet2012/val'
    base_dir = "/mnt/media/irielab/ubuntu_data/datasets/my_10class_ImageNet2012"
    train_size = 0.7
    image_dir_train_test_sprit(original_dir, base_dir, train_size)


if __name__ == "__main__":
    main()

'''from curses.ascii import SP
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil

SPLIT_SIZE = 0.7

class Split():

    def __init__(self):
        global SPLIT_SIZE

        originalDataPath = '/mnt/media/irielab/HDD(3TB)/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/10class'
        outputDir = "/mnt/media/irielab/HDD(3TB)/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/10class_train_val"

        dirName = {'train':'train',
                    'val': 'val'}

        getDir = os.listdir(originalDataPath)
        lenDir = len(getDir)
        if not os.path.isdir(outputDir):
            os.makedirs(outputDir)
        for key in ['train', 'val']:
            if not os.path.isdir(os.path.join(outputDir, dirName[key])):
                os.makedirs(os.path.join(outputDir, dirName[key]))
            
        for i, dir in enumerate(getDir):
            classDirPath = os.path.join(originalDataPath, dir)
            trainData, valData = train_test_split(classDirPath, test_size=SPLIT_SIZE)
            outputClassDirPath = os.path.join(outputDir,dir)
            if not os.path.isdir(outputClassDirPath):
                os.makedir(outputClassDirPath)

            for datalist in [trainData, valData]:
                if datalist:
                    for f in datalist:
                        shutil.copy2(f, outputClassDirPath)
        
if __name__ =='__main__':
    print('exec Split()')
    Split()
    print('done')'''