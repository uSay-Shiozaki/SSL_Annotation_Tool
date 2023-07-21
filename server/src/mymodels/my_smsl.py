# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from DEiT library:
https://github.com/facebookresearch/deit/blob/main/main.py
"""

import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import json
import os
import math
import sys
import copy
import scipy.io as scio
sys.path.append('/mnt/home/irielab/workspace/projects/my_research')
import ibot_main.models as models
import ibot_main.utils as utils

from pathlib import Path
from typing import Iterable, Optional
from torchvision import datasets, transforms
from torch.nn import functional as F
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, get_state_dict, ModelEma, accuracy
from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets.folder import ImageFolder, default_loader
from timm.optim import create_optimizer

from torch.utils.tensorboard import SummaryWriter
import tqdm
import logging
from torchinfo import summary
import random
import math
from mymodels.lib.make_myDataset import MakeDatasetfromJson, SplitDatasetfromJson
from sklearn import preprocessing
from pydantic import BaseModel
logging.basicConfig(level=logging.INFO)


class ClassProjHead(nn.Module):
    def __init__(self, in_dim, out_dim, cls_dim, norm=None, act='gelu', nlayers=3, 
                 hidden_dim=2048, bottleneck_dim=256, finetune_layer=1, **kwargs):
        super().__init__(**kwargs)
        if norm is not None:
            norm = self._build_norm(norm) #nn.Identity()
        act = self._build_act(act) #nn.Identity()

        assert (nlayers >= finetune_layer) and (finetune_layer >= 1)

        self.last_layer = None
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            if bottleneck_dim > 0:
                layers = [nn.Linear(in_dim, bottleneck_dim)]
            else:
                layers = [nn.Linear(in_dim, out_dim)]
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            cls_in_dim = hidden_dim
            if norm is not None:
                layers.append(norm)
            layers.append(act)
        
            for l in range(nlayers - 2):
                if finetune_layer >= l + 2:
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                    if norm is not None:
                        layers.append(norm)
                    layers.append(act)

        if finetune_layer == nlayers:
            if bottleneck_dim > 0:
                layers.append(nn.Linear(hidden_dim, bottleneck_dim))
                self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
                self.last_layer.weight_g.data.fill_(1)
            else:
                layers.append(nn.Linear(hidden_dim, out_dim))
            cls_in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.cls_head = nn.Linear(cls_in_dim, cls_dim) if cls_dim > 0 else nn.Identity()

    def forward(self, x):
        x = self.mlp(x)
        if self.last_layer is not None:
            x = nn.functional.normalize(x, dim=-1, p=2)
            x = self.last_layer(x)
        x = self.cls_head(x)
        return x

    def _build_norm(self, norm, hidden_dim, **kwargs):
        if norm == 'bn':
            norm = nn.BatchNorm1d(hidden_dim, **kwargs)
        elif norm == 'syncbn':
            norm = nn.SyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'ln':
            norm = nn.LayerNorm(hidden_dim, **kwargs)
        else:
            assert norm is None, "unknown norm type {}".format(norm)
        return norm

    def _build_act(self, act):
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'gelu':
            act = nn.GELU()
        else:
            assert False, "unknown act type {}".format(act)
        return act

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[1], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

class CarsDataset(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        data = scio.loadmat(os.path.join(root, \
            f'cars_annos.mat'))['annotations'][0].tolist()
        data = [elem for elem in data if elem[-1].item() == int(not train)]

        targeter = {}
        indexer = 0
        for elem in data:
            catg = elem[4].item()
            if catg not in targeter.keys():
                targeter[catg] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data:
            catg, path = elem[5].item(), elem[0].item()
            path = os.path.join(root, path)
            self.samples.append((path, catg))

class FlwrsDataset(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        
        data = np.array(sorted(os.listdir(os.path.join(root, 'jpg'))))
        labels = scio.loadmat(os.path.join(root, f'imagelabels.mat'))['labels'][0]
        ids = scio.loadmat(os.path.join(root, f'setid.mat'))        
        ids = np.concatenate((ids['trnid'], ids['valid']), axis=1)[0] if train else ids['tstid'][0]
        
        labels -= 1
        ids -= 1

        targeter = {}
        indexer = 0
        for catg in labels[ids]:
            if catg not in targeter.keys():
                targeter[catg] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for path, catg in zip(data[ids], labels[ids]):
            path = os.path.join(root, 'jpg', path)
            self.samples.append((path, catg))

def build_transform(is_train, input_size, color_jitter, aa, train_interpolation, reprob, remode, recount):
    resize_im = params.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=params.input_size,
            is_training=True,
            color_jitter=params.color_jitter,
            auto_augment=params.aa if params.aa.lower() != 'none' else None,
            interpolation=params.train_interpolation,
            re_prob=params.reprob,
            re_mode=params.remode,
            re_count=params.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                params.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * params.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(params.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(3)]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        print(f"criterion type is {type(self.base_criterion)}")

    def forward(self, inputs, outputs, labels):
        logging.debug("Calling DistillationLoss")
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            logging.debug("if not isinstance")
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            logging.debug("if self.distillation_type == 'none'")
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            logging.info("if self.distillation_type == 'soft'")
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            logging.info("elif self.distillation_type == 'hard'")
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        print(f"base_loss: {base_loss}")
        print(f"self.alpha: {self.alpha}")
        print(f"distillation_loss: {distillation_loss}")
        return loss

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)

        le = preprocessing.LabelEncoder()
        targets = torch.from_numpy(le.fit_transform(targets))
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            output = model(samples)
            outputs = model.module.head(output)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # switch to evaluation mode
    model.eval()
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            output = model.module.head(output)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def smslResponce(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    predMap = {}

    # switch to evaluation mode
    model.eval()
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            output = model.module.head(output)
            
            if target not in predMap.keys():
                predMap[target] = [output]
            else:
                predMap[target].append(output)
    
    return predMap

def main(params):
    utils.init_distributed_mode(params)

    if not os.path.exists(params.log_dir):
        os.makedirs(params.log_dir)
    writer = SummaryWriter(log_dir=params.log_dir)
    torch.autograd.set_detect_anomaly(True)

    if params.distillation_type != 'none' and params.finetune and not params.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(params.device)

    # fix the seed for reproducibility
    utils.fix_random_seeds(params.seed)

    cudnn.benchmark = True
    
    ### DEFINE MY DATASETS ###
    # train = val0.1 , val = val

    if not os.path.exists(params.log_dir):
        os.makedirs(params.log_dir)
    with open(os.path.join(params.log_dir, f"val_{params.ratio}_paths.json"), 'w') as f:
        json.dump(map, f, ensure_ascii=False, indent=4)

    transform_train = build_transform(True, params)
    transform_val = build_transform(False, params)
    
    jsonDataset = json.load("/database/XXXXX")
    trainJson, valJson, = SplitDatasetfromJson(jsonDataset)
    dataset_train = MakeDatasetfromJson(trainJson, transform=transform_train)
    params.nb_classes = 10
    dataset_val = MakeDatasetfromJson(valJson, transform=transform_val)

    print(f"length of train dataset is {len(dataset_train)}")

    if True:  # params.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if params.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if params.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    ### for few data
    sampler = torch.utils.data.DistributedSampler(dataset_train)
    while len(dataset_train) < params.batch_size:
        params.batch_size = int(params.batch_size * 0.5)
    # sampler_train: RASampler() -> __len__: math.floor(len(dataset) // 256 * 256 / num_replicas ) ??
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=params.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=params.pin_mem,
        drop_last=False
    )

    print(f"length of train loader is {len(data_loader_train)}")

    mixup_fn = None
    mixup_active = params.mixup > 0 or params.cutmix > 0. or params.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=params.mixup, cutmix_alpha=params.cutmix, cutmix_minmax=params.cutmix_minmax,
            prob=params.mixup_prob, switch_prob=params.mixup_switch_prob, mode=params.mixup_mode,
            label_smoothing=params.smoothing, num_classes=params.nb_classes)

    if 'swin' in params.arch:
        params.patch_size = 4
        model = models.__dict__[params.arch](
            patch_size=params.patch_size, 
            window_size=params.window_size,
            drop_rate=params.drop,
            attn_drop_rate=params.attn_drop_rate,
        )
        embed_dim = model.num_features
    else:
        model = models.__dict__[params.arch](
            patch_size=params.patch_size, 
            drop_rate=params.drop,
            drop_path_rate=params.drop_path,
            attn_drop_rate=params.attn_drop_rate,
            use_mean_pooling=params.avgpool_patchtokens,
        )
        embed_dim = model.embed_dim
    print(f"Model {params.arch} {params.patch_size}x{params.patch_size} built.")
    
    # projection head
    if params.finetune_head_layer > 0:
        head = ClassProjHead(
            embed_dim,
            params.out_dim,
            params.nb_classes,
            finetune_layer=params.finetune_head_layer)
    else:
        head = nn.Linear(embed_dim, params.nb_classes) if params.nb_classes > 0 else nn.Identity()
    # load weights to evaluate
    model.head = head
    model.head.apply(model._init_weights)
    if params.init_scale != 1.0:
        model.head.weight.data.mul_(params.init_scale)
        model.head.bias.data.mul_(params.init_scale)
    utils.load_pretrained_weights(model, params.pretrained_weights, params.checkpoint_key, params.arch, params.patch_size)
    model.cuda()
    
    if params.finetune:
        if params.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                params.finetune, map_location='cpu', check_hash=True)
        else:
            # checkpoint = torch.load(params.finetune, map_location='cpu')
            utils.restart_from_checkpoint(
                os.path.join(params.output_dir, params.finetune),
                state_dict=model,
            )

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_ema = None
    if params.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=params.model_ema_decay,
            device='cpu' if params.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if True:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[params.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = params.lr * params.batch_size * utils.get_world_size() / 512.0
    params.lr = linear_scaled_lr
    optimizer = create_optimizer(params, model_without_ddp) 
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(params, optimizer)

    criterion = LabelSmoothingCrossEntropy()
    
    if params.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif params.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=params.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if params.distillation_type != 'none':
        assert params.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {params.teacher_model}")
        teacher_model = create_model(
            params.teacher_model,
            pretrained=False,
            num_classes=params.nb_classes,
            global_pool='avg',
        )
        if params.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                params.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(params.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if params.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, params.distillation_type, params.distillation_alpha, params.distillation_tau
    )
    

    output_dir = Path(params.output_dir)
    if params.resume:
        if params.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                params.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(params.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not params.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            params.start_epoch = checkpoint['epoch'] + 1
            if params.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if params.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {params.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    if not params.eval:
        for epoch in range(params.start_epoch, params.epochs):
            if True: #params.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                params.clip_grad, model_ema, mixup_fn,
                set_training_mode=params.finetune == '',  # keep in eval mode during finetuning
            )

            lr_scheduler.step(epoch)
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            
            if params.output_dir and (test_stats["acc1"] >= max_accuracy):
                # always only save best checkpoint till now
                checkpoint_paths = [output_dir / 'checkpoint_{}_cls.pth'.format(params.checkpoint_key)]
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'params': params,
                    }, checkpoint_path)
            
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if params.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    if params.output_dir:
        Path(params.output_dir).mkdir(parents=True, exist_ok=True)
    for checkpoint_key in params.checkpoint_key.split(','):
        print("Starting evaluating {}.".format(checkpoint_key))
        args_copy = copy.deepcopy(params)
        args_copy.checkpoint_key = checkpoint_key
        main(args_copy)