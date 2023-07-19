import numpy as np
import os
import sys
sys.path.append('/mnt/home/irielab/workspace/projects/imageTransactionTest_2')

import argparse
import copy
import torch
import torch.backends.cudnn as cudnn
from ibot_main import utils as utils
from vit import VisionTransformer, vitSmall, vitBase as vit

from sklearn import metrics
from munkres import Munkres
from torchvision import transforms
from ibot_main.models.head import DINOHead
from ibot_main.loader import ImageFolder
from torchinfo import summary
import logging
import tqdm
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from utils import getFiles, getClusterMap
import json

# ======== loading env parameters =========
DIR_PATH = os.getenv("DIRPATH")
TARGET = os.getenv("TARGET")
BATCH_SIZE = os.getenv("BATCH_SIZE")
NUM_WORKERS = os.getenv("NUM_WORKERS")

if NUM_WORKERS is not os.cpu_count():
  logging.warning(f"Your machine has {os.cpu_count()} cpu cores. \
                  I strongly recommend to set the number of NUM_WORKERS \
                  {os.cpu_count()}")
  
VIT_ARCH = os.getenv("VIT_ARCH")
PATCH_SIZE = os.getenv("PATCH_SIZE")
OUT_DIM = os.getenv("OUT_DIM")
LOCAL_RANK = os.getenv("LOCAL_RANK")
RANDOM_STATE = os.getenv("RANDOM_STATE")



@torch.no_grad()
def clustering(model, dataLoader, nClusters=10):
  trueLabels, predLabels = [], []
  k_meansLabels = []
  
  # ======== getting file list ========
  path = os.path.join(DIR_PATH, TARGET)
  files = getFiles(path)
  
  with torch.no_grad():
    print("Start Evaluations")
    for i ,(inp, label) in enumerate(tqdm.tqdm(dataLoader)):
        inp = inp.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        trueLabels.append(utils.concat_all_gather(label))
        output = model(inp)
        
        # to pass k-means layer
        k_meansLabels.append(utils.concat_all_gather(output))
        
        # predicting in linear layer on the last layer
        pred = output.max(dim=1)[1]
        pred = utils.concat_all_gather(pred)
        predLabels.append(pred)

    linearPreds = torch.cat(predLabels).cpu().detach().numpy()
    trues= torch.cat(trueLabels).cpu().detach().numpy()
    
    # output is expected as shape(2600, 1000)
    k_meansPreds = torch.cat(k_meansLabels).cpu().detach().numpy()
    k_meansPreds= KMeans(n_clusters=nClusters, random_state=251).fit_predict(k_meansPreds)

    predMap = {}
    for i, pred in enumerate(k_meansPreds):
      path = os.path.join(os.path.join(DIR_PATH, TARGET), files[i])
      if not str(pred) in predMap.keys():
          predMap[str(pred)] = [path]
      else:
          predMap[str(pred)].append(path)
          
    for i, pred in enumerate(linearPreds):
      path = os.path.join(os.path.join(DIR_PATH, TARGET), files[i])
      if not str(pred) in predMap.keys():
        predMap[str(pred)] = [path] 
      else:
        predMap[str(pred)].append(path)
    
    with open('./mycodes/for_ibot/cluster_map.json', 'w') as f:
        json.dump(predMap, f, indent=4)


@torch.no_grad()
def main(args):
  # This flag allows you to enable the inbuilt cudnn auto-tuner 
  # to find the best algorithm to use for your hardware if
  # input sizes are constant.
  cudnn.benchmark = True
  
  # ======== preparing data ========
  # interpolation 0:NEAREST 1:BOX 2:BILINEAR 3:HAMMING→BICUBIC 4:LANCZOS
  # Nomrmalize(mean, std) in 3 channels
  transform = transforms.Compose([
    transforms.Resize(156, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
  ])
  

  targetDir = os.path.join(DIR_PATH, TARGET)
  dataset = ImageFolder(targetDir, transform=transform)
  # pin_memory: automatic pin memory to make training faster.
  # drop_last: When fetching from iterable-style datasets \
  # with multi-processing, the drop_last argument drops \
  # the last non-full batch of each worker’s dataset replica. \
  #WARNING Enabling drop_last possibly reduce the number of images for our experiment.
  
  dataLoader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workders=NUM_WORKERS,
    pin_memory=True,
    drop_last=True
  )
  
  logging.info(f"Image Data loaded with {len(dataset)} images.")
  
  # ======== building ViT network ========
  if VIT_ARCH not in vit.__dict__:
    raise ValueError(f"VIT_ARCH is {VIT_ARCH}. Select a value named \
      {vit.__dict__[1:]}" )
  else:
    model = vit.__dict__[VIT_ARCH](
      patchSize=PATCH_SIZE,
      numClasses=0
    )
    embedDim = model.embedDim
    
  logging.info(f"Model {VIT_ARCH}, {PATCH_SIZE}x{PATCH_SIZE} built.")
  
  #TODO check DINO
  model = utils.MultiCropWrapper(model, DINOHead(
    embedDim,
    OUT_DIM,
    act='gelu'
  ))
  
  model.cuda(LOCAL_RANK)

  model  = torch.nn.parallel.DistributedDataParallel(model, device_ids=LOCAL_RANK)
  # add chieckpoint_key as a kwarg. 
  # ** before dict means unpaccking to send args.
  utils.restart_from_checkpoint(PRETRAINED_WEIGHTS, **{CHECKPOINT_KEY: model})
  model.eval()
  
  # ======== evaluate unsup cls ========
  logging.info("Clustering...")
  
  if __name__ == '__main__':
  
    main()
  