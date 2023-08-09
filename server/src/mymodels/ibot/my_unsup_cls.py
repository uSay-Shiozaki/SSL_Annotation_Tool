# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
np.set_printoptions(threshold=np.inf)
import os
import sys
sys.path.append('/mnt/home/irielab/workspace/projects/my_research')

import argparse
import copy
import torch
import torch.backends.cudnn as cudnn
from mymodels.ibot import utils as utils
from mymodels.ibot import models as models

from sklearn import metrics
from munkres import Munkres
from torchvision import transforms as pth_transforms
from mymodels.ibot.models.head import DINOHead
from mymodels.ibot.loader import ImageFolder

import logging
import tqdm
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mymodels.lib.make_myDataset import MakeImageNetDataset, MakeMNISTDataset
import json
from sklearn import preprocessing
import torch.nn as nn
import torch.distributed as dist

def eval_pred(label, pred, calc_acc=False):
    print("Calling eval_pred()")
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    if not calc_acc:
        return nmi, ari, f, -1
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    print(pred_adjusted)
    # metrics.accuracy_score(true, pred)???
    print("test acc = > {}".format(metrics.accuracy_score(pred_adjusted, label)))
    acc = metrics.accuracy_score(label, pred_adjusted)
    return nmi, ari, f, acc


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred

@torch.no_grad()
def main_eval(arch="vit_small",
              patch_size: int=16,
              window_size: int=7,
              out_dim: int=1000,
              local_rank: int=0,
              num_workers: int = 10,
              pretrained_weights: str =None,
              checkpoint_key: str="student",
              data_path: str=None,
              target: str = "val"):
    cudnn.benchmark = True
    
    if arch == "vit_small":
        pretrained_weights = '/weights/ibot_small_pretrain.pth'
    elif arch == "vit_base":
        pretrained_weights = '/weights/ibot_base_pretrain.pth'
    elif arch == "vit_large":
        pretrained_weights = '/weights/ibot_large_pretrain.pth'
    else:
        sys.exit(1)

    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # valdir = os.path.join(params.data_path, "val")
    valdir = os.path.join(data_path, target)
    dataset_val = ImageFolder(valdir, transform=transform)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1, # for get data path in my function, default: params.batch_size_per_gpu
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    logging.info(f"Data loaded with {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    if 'swin' in arch:
        patch_size = 4
        model = models.__dict__[arch](
            window_size=window_size,
            patch_size=patch_size,
            num_classes=0)
        embed_dim = model.num_features
    else:
        model = models.__dict__[arch](
            patch_size=patch_size, 
            num_classes=0)
            #default num_classes=0
        embed_dim = model.embed_dim
    logging.info(f"Model {arch} {patch_size}x{patch_size} built.")
    model = utils.MultiCropWrapper(model, DINOHead(
        embed_dim,
        out_dim,
        act='gelu'))
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    if not pretrained_weights == None:
        utils.restart_from_checkpoint(pretrained_weights, **{checkpoint_key: model})
        model.eval()
    else:
        assert FileNotFoundError
        logging.fatal("Pretrained Weight is None")
        sys.exit(1)
    
    # ============ evaluate unsup cls ... ============
    print("Evaluating unsupervised classification for val set...")
    if not data_path == None:
        res = my_eval(model, data_path, data_loader_val, arch, patch_size, n_clusters=10)
    else:
        logging.fatal("Argument Required: ERROR data_path is None")
        
    return res

@torch.no_grad()
def eval_unsup(model, data_loader):
    metric_logger = utils.MetricLogger(delimiter="  ")
    real_labels, pred_labels = [], []
    for samples, labels in tqdm.tqdm(metric_logger.log_every(data_loader, 10)):
        samples = samples.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        output = model(samples)
        pred = utils.concat_all_gather(output.max(dim=1)[1]) 
        pred_labels.append(pred)
        real_labels.append(utils.concat_all_gather(labels))

    print("pred_labels\n{}".format(pred_labels))
    pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
    real_labels = torch.cat(real_labels).cpu().detach().numpy()
    nmi, ari, fscore, adjacc = eval_pred(real_labels, pred_labels, calc_acc=True)
    print("NMI: {}, ARI: {}, F: {}, ACC: {}".format(nmi, ari, fscore, adjacc))

# MyFunctions

def my_eval(model, data_path, data_loader, arch, patch_size, n_clusters=10, k_means: bool = True, target: str = "val"):
    real_labels, pred_labels = [], []
    output_labels = []
    # metric_logger for batches in each GPU of ViT
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    files = []
    for v in os.listdir(os.path.join(data_path, target)):
        fileList = os.listdir(os.path.join(os.path.join(data_path, target), v))
        for file in fileList:
            path = os.path.join(os.path.join(os.path.join(data_path, target), v), file)
            files.append(path)
    
    predMap = {}
    with torch.no_grad():
        # remove last layer in DINOHead to predict with k-means.
        if k_means:
            model.last_layer = nn.Identity()
            
        print("Start Evaluations")
        for i ,(inp, label) in enumerate(tqdm.tqdm(data_loader)):
            inp = inp.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            # real_labels.append(utils.concat_all_gather(label))
            real_labels.append(label)
            output = model(inp)
            
            if k_means:
                # to pass k-means 
                output_labels.append(utils.concat_all_gather(output))
            else:
                # predicting in linear layer on the last layer
                pred = output.max(dim=1)[1]
                pred = utils.concat_all_gather(pred)
                pred_labels.append(pred)
                
        real_labels = torch.cat(real_labels).cpu().detach().numpy()
        if not k_means:
            pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
            pred_scores = eval_scores(real_labels, pred_labels)
            logging.info(f"Pred Scores=> {pred_scores}")
        else:
            # output is expected as shape(2600, 1000)
            output_labels = torch.cat(output_labels).cpu().detach().numpy()
            kmeans_pred = KMeans(n_clusters=n_clusters, random_state=251).fit_predict(output_labels)
            output_scores = eval_scores(real_labels, kmeans_pred)
            logging.info(f"k-means pred Scores=> {output_scores}")
            f = open('log_eval_score.txt', 'a')
            f.write(f"k_means score with {arch} {patch_size}x{patch_size} -> NMI, ARI, FMI = {output_scores}\n")

        for i, pred in enumerate(kmeans_pred):
            path = os.path.join(os.path.join(data_path, target), files[i])
            if not str(pred) in predMap.keys():
                predMap[str(pred)] = [path]
            else:
                predMap[str(pred)].append(path)
        
        with open('/database/cluster_map.json', 'w') as f:
            json.dump(predMap, f, indent=4)
        
        return {"body": predMap}
    
def eval_atCluster(model, jsonFile, target_cluster: str="3", n_clusters=3):
    jsons = open(jsonFile, 'r')
    jsons = json.load(jsons)
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = MakeImageNetDataset(jsons[target_cluster], transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, # labelEncoder needs whole label list
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    real_labels, pred_labels = [], []
    output_labels = []
    # metric_logger for batches in each GPU of ViT
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    with torch.no_grad():
        print("Start Evaluations")
        for i ,(inp, label) in enumerate(tqdm.tqdm(data_loader)):
            inp = inp.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            real_labels.append(utils.concat_all_gather(label))
            output = model(inp)
            output_labels.append(utils.concat_all_gather(output))
            pred = output.max(dim=1)[1]
            pred = utils.concat_all_gather(pred)
            pred_labels.append(pred)

        pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
        real_labels = torch.cat(real_labels).cpu().detach().numpy()
        # output is expected as shape(2600, 1000)
        output_labels = torch.cat(output_labels).cpu().detach().numpy()
        kmeans_pred = KMeans(n_clusters=n_clusters, random_state=251).fit_predict(output_labels)
        pred_scores = eval_scores(real_labels, pred_labels)
        output_scores = eval_scores(real_labels, kmeans_pred)
        print(f"Pred Scores=> {pred_scores}")
        print(f"Output Scores=> {output_scores}")
        print(f"labels => {set(real_labels)}")
        get_true_pred_map(real_labels, kmeans_pred)
        # print(pred_labels)
        # print(kmeans_pred)
    
    
def get_true_pred_map(label, pred):
    assert len(label) == len(pred)
    map = {}
    bar_map = {}
    for i in range(len(pred)):
        if not pred[i] in map.keys():
            map[pred[i]] = []
        map[pred[i]].append(label[i])
    print(map.keys())
    for j in map.keys():
        numList = [0]*len(set(label))
        for v in map[j]:
            numList[v] += 1
        bar_map[j] = numList
    print(bar_map)
    df = pd.DataFrame(bar_map)
    print(df)
    # plot stack bar
    fig, ax = plt.subplots(figsize=(10,6))
    for i in range(len(df)):
        ax.bar(df.columns,
               df.iloc[i],
               bottom=df.iloc[:i].sum(),
               alpha=.7)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("NoI")
    ax.set_xticks(df.columns)
    ax.legend(df.index.tolist())
    plt.show()
    return map

def eval_scores(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_mutual_info_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    return nmi, ari, f

def init_distributed_mode(dist_url=None):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = params.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        rank, gpu, world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    if dist_url == None:
        logging.fatal("dist_url is None")
        sys.exit(1)
    else:
        dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank,
        )

        torch.cuda.set_device(gpu)
        print('| distributed init (rank {}): {}'.format(
            rank, dist_url), flush=True)
        dist.barrier()
        # setup_for_distributed(rank == 0)



if __name__ == '__main__':

    init_distributed_mode(dist_url= "env://")

    weight_path = "./ibot_small_pretrain.pth"
    data_path = "/mnt/media/irielab/win_drive1/ImageNet/imagenet-object-localization-challenge-2012/ILSVRC/Data/CLS-LOC/10class_train_val"
    main_eval(pretrained_weights=weight_path, data_path=data_path)
