# if you want install kivy on your root environemnts
# python3 -m pip install "kivy[full]" kivymd

# install gdown to download file from my public drive.
apt get install gdown

# Pretrained-weight from CIFAR10 for iBOT
gdown https://drive.google.com/uc?id=1nJwAcx8pDRTHo3eeyGz520DkY_dVP24G
# move pth file to weight folder
mv clustering_cifar10_ckp.pth ./server/weight/

# Pretrained-weight from ImageNet for iBOT
# gdown https://drive.google.com/uc?id=1BSnxAYuAAumpK9pgTRnKoC6UJGeDs_GC
# mv clutering_imagenet_ckp.pth ./server/weight/

# CIFAR 10 Image Dataset
# gdown https://drive.google.com/uc?id=1rdCZwn5jMQbKurscJTjM8jz3WkFgSlEc

# pretrained-weight from ImageNet for SwAV
# gdown https://drive.google.com/uc?id=15_K21qGvUvEqbkZKu0NHT2VyOtFgn5RR

# create docker network
sudo docker network create app_network