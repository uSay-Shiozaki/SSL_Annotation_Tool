
import os
import sys
sys.path.append('/mnt/home/irielab/workspace/projects/my_research')
from ibot_main import models as models
from ibot_main import utils
from ibot_main.models.head import DINOHead
print(os.getcwd())
model = models.__dict__["vit_small"](patch_size=24, num_classes=0)
model = utils.MultiCropWrapper(model, DINOHead(
    1000,
    10,
    act='gelu'))
print(model)

