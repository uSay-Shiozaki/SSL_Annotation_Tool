import torch
import os
from livelossplot import PlotLosses

dirpath = '/mnt/home/irielab/workspace/projects/imageTransactionTest_2/swav/checkpoints'
filelist = os.listdir(dirpath)
liveloss = PlotLosses()
epochs = 1

for i in range(1,len(filelist)):
    logs = {}
    ckp = torch.load(os.path.join(dirpath, filelist[i]))
    logs['loss'] = ckp['scores'][1]
    liveloss.update(logs)
liveloss.send()

'''
# load weights
if os.path.isfile(args.pretrained):
    state_dict = torch.load(args.pretrained, map_location="cuda:" + str(args.gpu_to_work_on))
    
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # remove prefixe "module."
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    for k, v in model.state_dict().items():
        if k not in list(state_dict):
            logger.info('key "{}" could not be found in provided state dict'.format(k))
        elif state_dict[k].shape != v.shape:
            logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
            state_dict[k] = v
    # load weights
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Load pretrained model with msg: {}".format(msg))
else:
    logger.info("No pretrained weights found => training with random weights")
'''