import os
import sys
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axg
import numpy as np
from PIL import Image


dataPath = "/mnt/media/irielab/win_drive/ImageNet/imagenet-object-localization-challenge-2012/ILSVRC/Data/CLS-LOC/10class_train_val/val"
picks = 5

def main():
    dirs = os.listdir(dataPath)
    labels = []
    grid = []
    for dir in dirs:
        classPath = os.path.join(dataPath,dir)
        files = os.listdir(classPath)
        labels.append(str(dir))
        i = 0
        imgs = []
        while i < picks:
            file = files[i]
            img = Image.open(os.path.join(classPath,file))
            img = img.resize((224,224)).rotate(180)
            imgs.append(img)
            i += 1
        grid.append(imgs)
    fig= plt.figure(figsize=(20.,10.,))
    gs = fig.add_gridspec(5,10,
                            wspace=0.05,
                            hspace=0.05)
    print(labels)
    for i in range(len(grid)):
        for j, im in enumerate(grid[i]):
            ax = fig.add_subplot(gs[j,i])
            ax.axis('off')
            if j == picks - 1:
                ax.set_title(labels[i], y=-0.2)
            plt.imshow(im)
        
    plt.show()

if __name__ == '__main__':
    main()
