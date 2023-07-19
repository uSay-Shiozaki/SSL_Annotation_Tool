import os
from re import T
import time
from swac.src.utils import AverageMeter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim


class Trainer(object):
    
    def train(self,train_loader, model, optimizer, epoch, lr_schedule, queue):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        
        model.train()
        user_the_queue = False
        
        end = time.time()
        for it, inputs in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            
            # update learning rate
            iteration = epoch * len(train_loader) + it
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[iteration]
                
            # normalize the prototypes
            # TODO: need to learn about nn.module.prototype
            with torch.no_grad():
                # model.module.prototypes = nn.Linear()
                # clone() does not share the tensor values unlike detach()
                w = model.module.prototypes.weight.data.clone()
                # p - the exponent value in the norm furmulation.
                w = nn.functional.normalize(w, dim=1, p=2)
                # copy_(src) : copies the elements from src into self tensor and returns self
                # update self weight
                model.module.prototypes.weight.copy_(w)
                
            # ========= multi-res forward passes ... ==========
            # model: in forward(self, x) -> return x, self.prototypes(x)
            embedding, output = model(inputs)
            ## detach cuts a calc graph and make a new tensor
            embedding = embedding.detach()
            # TODO: what's bs
            # the number of rows for inputs[0]
            # batch size?
            bs = inputs[0].size(0)
            
            # ========= swav loss ... ==========
            loss = 0
            for i, crop_id in enumerate(args.crops_for_assign):
                # forbid calculating grad
                with torch.no_grad():
                    # out -> extract protorypes between "bs"
                    out = output[bs * crop_id: bs * (crop_id + 1)].detach()
                    
                    # time to use the queue
                    if queue is not None:
                        if user_the_queue or not torch.all(queue[i, -q, :] == 0):
                            user_the_queue = True
                            # torch.cat: concatenate tensors
                            # torch.mm : matrix multiplication of the matrices
                            out = torch.cat((torch.mm(
                                queue[i],
                                model.module.prototypes.weight.t()                                
                            ), out))
                        # fill the queue
                        # [0 , 1 , 2 , 3 , 4]
                        # [-4, -3, -2, -1, #]
                        print("queue[{}] is {}".format(i, queue[i]))
                        queue[i, bs:] = queue[i, :-bs].clone()
                        queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]
                        
                    # get assignments
                    q = self.distributed_sinkhorn(out)[-bs:]
                    
                # cluster assignment prediction
                subloss = 0
                for v in np.delete(np.arrange(np.sum(args.nmb_crops)), crop_id):
                    x = output[bs * v: bs * (v + 1)] / args.temperature
                    subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                    loss += subloss / (np.sum(args.nmb_crops) - 1)
                loss /= len(args.crops_for_assign)
            loss /= len(args.crops_for_assign)
            
            # =========== backward and optimize step ... ===========
            optimizer.zero_grad()
            loss.backward()
            
            # cancel gradients for the prototypes
            if iteration < args.freeze_prototypes_niters:
                for name, p in model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
            optimizer.step()
            
            # ========== miscellaneous ... ==========
            # AverageMeter.update(val, n)
            # torch.Tensor.size() torch.Size([row.column])                        
            losses.update(loss.item(), inputs[0].size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if args.rank == 0 and it % 50 == 0:
                 logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.optim.param_groups[0]["lr"],
                )
                 )
        return (epoch, losses.avg), queue
    
    @torch.no_grad()
    def distributed_sinkhorn(self,out):
        # Q is K-by-B for consistency with notations from SwAV paper
        Q = torch.exp(out / args.epsilon).t()
        # number of samples to assign
        B = Q.shape[1] * args.world_size
        # how many prototypes
        K = Q.shape[0]
        
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q
        
        for it in range(args.sinkhorn_iterations):
            #normalize each row: total weight per ptorotype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K
            
            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
            
        # the columns must sum to 1 so that Q is an assignment
        Q *= B
        return Q.t()

                    