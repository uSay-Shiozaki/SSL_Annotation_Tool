# SSL_Annotation_Tool
A Tool for building image datasets without every labeling by using self-supervised learning models

# Requirements
- OS: Linux 
- GPU: nVidia GPU 
# services
- Docker Compose

# Installation
Clone our repository

```
git clone https://github.com/uSay-Shiozaki/SSL_Annotation_Tool.git
```

Edit env file
Set your image directory locations.

```
~/SSL_Annotation_Tool$ vim .env
```
```
YOUR_IMAGE_DATASET_PATH={YOUR/IMAGE/DATASET/PATH}
```
Confirm cuDNN and CUDA versions.
Change the docker image in Dockerfile in "server" directory if the CUDA doesn't match. [nvidia docker images](https://hub.docker.com/r/nvidia/cuda)
Default image is nvcr.io/nvidia/pytorch:22.07-py3

```
~/SSL_Annotation_Tools$ vim ./server/Dockerfile
FROM nvcr.io/nvidia/pytorch:22.07-py3 <- change this image to the matched image.
```

```
~$ docker-compose up
```

# Manual
## Operations
### Start
   #### SSL  
   Run Self-Supervised Learning with the JSON file of a user, using small iBOT weights [available](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth).
   #### SmSL iBOT  
   Run Semi-Supervised Learning on the selected images with [iBOT](https://github.com/bytedance/ibot).
   #### SmSL SwAV  
   Run Semi-Supervised Learning on the selected images with [SwAV](https://github.com/facebookresearch/swav). We recommend using SwAV if the images are low-resolution images.
   #### Load Annotation Data  
   Load a JSON file of annotation data and Display images.
 ### Export  
   Export the JSON file of annotation data the user checked.

## Menu Bar
 ### Current: XX  
   Display the number of the current clusters.
 ### Label Entry  
   Entry to register a category name of the cluster images.
 ### Save  
   Save the category name in the entry box.
 ### Remove/Keep Target  
   Select Remove or Keep mode to operate the images a user selected.
 ### Node Previous/Next  
   Move to the previous or next cluster nodes.
 ### Page Previous/Next  
   Move to the previous or next pages in a cluster.
 ### Remain  
   Display the rest of the images removed by the operations.

## Main Grid View  
  Display images in a cluster on each page.
## Preview  
  Display an image the user clicked.




