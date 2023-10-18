import tempfile
from pathlib import Path
from xml.dom import NotFoundErr

import numpy as np
from tensorboard.backend.event_processing.plugin_event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import interp1d

def load_tfevent(event_filename: Path, tag: str = 'foo'):
    assert event_filename.exists()
    acc = EventAccumulator(path=str(event_filename))
    acc.Reload()

    # check `tag` exists in event
    assert tag in acc.Tags()['tensors']

    for t in acc.Tensors(tag):
        print(f'walltime = {t.wall_time}, step = {t.step}, value = {t.tensor_proto.float_val[0]}')


def parse_tflog():
    writer = None
    with tempfile.TemporaryDirectory() as dname:
        log_dir = Path(dname)
        with SummaryWriter(log_dir=log_dir) as writer:
            for i in range(100):
                writer.add_scalar('foo', np.random.random(), i)

        tfevents = list(log_dir.glob('events.out.tfevents.*'))
        assert len(tfevents) == 1
        tfevent = tfevents[0]

        load_tfevent(tfevent)

#FIXME Does not work
'''
I tried to make codes which works like tensorboard-like,
but it doesn't work properly.
'''
def pltTFEvent():
    dirPath = '/mnt/media/irielab/ubuntu_data/workspace/swav_checkpoints/MNIST_logs/tensorboard_logs/vals_20ep'
    dirNames = []
    for dir in os.listdir(dirPath):
        dirNames.append(dir)
    # tfevents = list(dirPath.glob('events.out.tfevents.*'))
    fig = plt.figure()
    axes = {}
    for dirName in dirNames:
        path = os.path.join(dirPath, dirName)
        print(path)
        tfevent =  glob.glob(os.path.join(path,'events.out.tfevents.*'))
        print(tfevent)
        tfevent = tfevent[0]
        ratio = dirName.split('_')[1]
        '''
        try:
            os.path.exists(tfevent)
        except:
            FileNotFoundError()
            return
        '''
        acc = EventAccumulator(path = tfevent)
        acc.Reload()
        # print(acc.Tags())
        # tensors = acc.Tags()['tensors']
        tags = acc.Tags()['tensors']
        for tag in tags:
            if not tag in axes.keys():
                ax = fig.add_subplot(111)
                axes[tag] = ax
            epochs = []
            values = []
            for t in acc.Tensors(tag):
                epochs.append(t.step)
                values.append(t.tensor_proto.float_val[0])
            axes[tag].plot(epochs, values, label=f"val_{ratio}")
    plt.legend(loc='best')
    plt.show()

def exec():
    dirPath = '/mnt/media/irielab/ubuntu_data/workspace/swav_checkpoints/IN2012_logs/tensorboard_logs/IN2012_10class_random_200ep_seed251'
    # dirPath = "/mnt/media/irielab/ubuntu_data/workspace/ibot_checkpoints/MNIST/tensorboard_logs/re_MNIST_200ep_seed251_5e-04/vit_small"
    dirNames = ["log_0.01", "log_0.05", "log_0.1", "log_0.5"]
    
    # set this var
    targetTag = 'val/Acc'
    fig = plt.figure()
    ax = fig.add_subplot()
    for dirName in dirNames:
        path = os.path.join(dirPath, dirName)
        try:
            os.path.exists(path)
        except:
            NotFoundErr("NotFoundError No Such Files")
            return
        tfevent =  glob.glob(os.path.join(path,'events.out.tfevents.*'))[0]
        ratio = dirName.split('_')[1]
        # split "log_val{ratio}" format
        if False:
            unit = "val"
            if not ratio == "val":
                ratio = ratio.replace(unit, unit + " ").split()
                ratio = ratio[1]
            else:
                ratio = "1.0"
        print(f"tfevent path: {tfevent.split('/')[-1]}")
        acc = EventAccumulator(path=tfevent)
        acc.Reload()
        epochs = []
        values = []
        accMax = -1e-06
        plots = [0,10,20,30,40,50]
        # plots = [0, 5, 10, 20, 50, 100, 150, 200]
        print(acc.Tags())
        for t in acc.Tensors(targetTag):
            if t.step in plots:
                epochs.append(t.step)
                values.append(t.tensor_proto.float_val[0])
                accMax = max(accMax, t.tensor_proto.float_val[0])
                if t.step == 20:
                    print(f"maxACC at 20 epochs: {accMax}")
        print(f"maxACC: {accMax}")
        '''
        model = interp1d(epochs, values, kind='cubic')
        xs = np.linspace(np.min(epochs), np.max(epochs), np.size(epochs)*100)
        ys = model(xs)
        ax.plot(xs, ys, label=f'val_{ratio}')
        '''
        
        perc = float(ratio) * 100
        print(perc)
        ax.plot(epochs, values, label=f'{perc}%', marker='x')
    plt.grid()
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("ACC", fontsize=16)
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.show()



if __name__ == '__main__':
    exec()
