# SSL_Annotation_Tool
A Tool for building image datasets without every labeling by using self-supervised learning models

# Requirements
- OS: Linux 
- GPU: Geforce GTX2080
services
- Docker Compose

# Installation
Edit env file
Set your image directory locations
```
~/SSL_Annotation_Tool$ vim .env
```
```
YOUR_IMAGE_DATASET_PATH={YOUR_IMAGE_DATASET_PATH}
```
```
~$ docker-compose up
```
# Manual
## Operations
### Start
   #### SSL  
   Run Self-Supervised Learning with the JSON file of a user, using small iBOT weights [available].
   #### SmSL iBOT  
   Run Semi-Supervised Learning on the selected images with iBOT.
   #### SmSL SwAV  
   Run Semi-Supervised Learning on the selected images with SwAV. We recommend using SwAV if the images are low-resolution images.
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

# Example



