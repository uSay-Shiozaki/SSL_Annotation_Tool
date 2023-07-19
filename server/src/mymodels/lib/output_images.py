
import matplotlib.pyplot as plt
import numpy as np
import os
from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import json

logger = getLogger()

STEPS = 1000
def show_images(map, loader):
    #imgFolderPath = os.path.join(args.data_path, foldername)
    # show images in each cluster
    loader_list = []
    for i, (input, _) in enumerate(loader):
        if i + 1 > STEPS:
            break
        #logger.info("================ iterate steps: %s ===============",i + 1)
        input = input.cuda(non_blocking=True)
        loader_list.append(input)
       

    for key, values in map:
        plt.figure(figsize=(20,10))

        for i, value in enumerate(values):
            # 20 is the limit of subplot's size
            if i + 1 > 20:
                break
            input = loader_list[value]
            image_np = input[0].cpu().detach().numpy().copy()
            img = np.transpose(image_np, (1,2,0))
            img = (img + 1) / 2
            ax = plt.subplot(2,10,i + 1)
            ax.set_title(key,c='k',fontsize=20)
            plt.axis('off')
            plt.imshow(img)
        if not os.path.isdir('./output'):
            os.makedirs('./output')
        plt.savefig("./output/{}_images.png".format(key))
           
    logger.info("Save images")
    #plt.savefig("./output/output-2/output_images")

if __name__ == '__main__':
    data_path = '/mnt/media/irielab/HDD(3TB)/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC'
    test_folderName = 'test'
    test_dataset = datasets.ImageFolder(os.path.join(data_path, test_folderName))
    test_dataset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        ])
    test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    )
    cp_path = './class_map.json'

    if os.path.isfile(cp_path):
        map = json.load(open(cp_path,'r'))
    show_images(map,test_loader)