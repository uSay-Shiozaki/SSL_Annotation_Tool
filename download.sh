# install gdown to download file from my public drive.
sudo apt install gdown

# Pretrained-weight from CIFAR10 for iBOT
# with output dim 1000
if [ ! -e $"./database/clustering_cifar10_1000_ckp.pth" ];then
    gdown https://drive.google.com/uc?id=1jhE1uu_vRFoybDD301KoD_8YsvKQvobm
    mv ./clustering_cifar10_1000_ckp.pth ./database
fi

# with output dim 8192
# gdown https://drive.google.com/uc?id=1nJwAcx8pDRTHo3eeyGz520DkY_dVP24G


# Pretrained-weight from ImageNet for iBOT
if [ ! -e $"./server/weights/clustering_imagenet_ckp.pth" ];then
    gdown https://drive.google.com/uc?id=1BSnxAYuAAumpK9pgTRnKoC6UJGeDs_GC
    sudo mv ./clustering_imagenet_ckp.pth ./server/weights
fi

# CIFAR 10 Image Dataset
if [ ! -e $"./IMAGE_DATA/cifar10_imagedata.zip" ];then
    gdown https://drive.google.com/uc?id=1rdCZwn5jMQbKurscJTjM8jz3WkFgSlEc
    mv ./cifar10_imagedata.zip ./IMAGE_DATA
fi
if [ ! -e $"./IMAGE_DATA/cifar10_imagedata" ];then
    unzip ./IMAGE_DATA/cifar10_iamgedata.zip -d ./IMAGE_DATA/cifar-10
fi

# CIFAR 10 with 1000 images
if [ ! -e $"./IMAGE_DATA/cifar10_1000images.zip" ];then
    gdown https://drive.google.com/uc?id=1cjv5dFF9GcUGkCMNR7RwqTeiuCrsekPa
    mv ./cifar10_1000images.zip ./IMAGE_DATA
fi

if [ ! -e $"./IMAGE_DATA/cifar10_1000images" ];then
    unzip ./IMAGE_DATA/cifar10_1000images.zip 
    sudo mv ./cifar10_1000images ./IMAGE_DATA
fi

# CIFAR 10 with 500 images
if [ ! -e $"./IMAGE_DATA/cifar10_500images.zip" ];then
    gdown https://drive.google.com/uc?id=1MChYXm9-FnAOegRe5W-1VAqu890D1bwf
    mv ./cifar10_500images.zip ./IMAGE_DATA
fi

if [ ! -e $"./IMAGE_DATA/cifar10_500images" ];then
    unzip ./IMAGE_DATA/cifar10_500images.zip 
    sudo mv ./cifar10_500images ./IMAGE_DATA
    
fi

# CIFAR 10 with 100 images
if [ ! -e $"./IMAGE_DATA/cifar10_100images.zip" ];then
    gdown https://drive.google.com/uc?id=1YhAzHUgULbSRHkZE_vXJr-Fqpyes6ftA
    sudo mv ./cifar10_100images.zip ./IMAGE_DATA
fi

if [ ! -e $"./IMAGE_DATA/cifar10_100images" ];then
    unzip ./IMAGE_DATA/cifar10_100images.zip 
    sudo mv ./cifar10_100images ./IMAGE_DATA    
fi

# CIFAR 10 with 50 images
if [ ! -e $"./IMAGE_DATA/cifar10_50images.zip" ];then
    gdown https://drive.google.com/uc?id=1quh-ljI-8L5YR_KBWB9Z2d2ByVsQZCFe
    mv ./cifar10_50images.zip ./IMAGE_DATA
fi

if [ ! -e $"./IMAGE_DATA/cifar10_50images" ];then
    unzip ./IMAGE_DATA/cifar10_50images.zip 
    sudo mv ./cifar10_50images ./IMAGE_DATA    
fi
# pretrained-weight from ImageNet for SwAV
if [ ! -e $"./server/weights/swav_800ep_pretrain.pth.tar" ];then
    gdown https://drive.google.com/uc?id=15_K21qGvUvEqbkZKu0NHT2VyOtFgn5RR
    sudo mv ./swav_800ep_pretrain.pth.tar ./server/weights
fi

# create docker network
sudo docker network create app_network