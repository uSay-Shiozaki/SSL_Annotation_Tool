from fastapi import FastAPI
from pydantic import BaseModel
from mymodels.my_unsup_cls import main_eval, init_distributed_mode

app = FastAPI()

# initialize distributed mode once
init_distributed_mode(dist_url="env://")

@app.get('/helloworld')
def get_hello_message():
    return {"message": "Hello World"}

class SchemaOfInputDataPathRequest(BaseModel):
    pretrained_weights: str
    data_path: str

class SchemaOfTableResponse(BaseModel):
    body: dict
    
class SchemaOfSmSLTableRequest(BaseModel):
        batch-size: int = 128
        epochs: int = 400
        distill_epochs: int = 150
        
        avgpool_patchtokens: int = 0
        arch: str = "vit_small"
        patch_size: int = 16
        window_size: int = 7
        input-size: int = 224
        pretrained_weights: str
        checkpoint_key: str = "student"
        drop: float = 0.0
        attn_drop_rate: float = 0.0
        drop-path: float = 0.1
        
        model-ema: bool = True
        model-ema-decay: float = 0.99996
        model-ema-force-cpu: bool = False
        
        opt: str = "adamw"
        opt-eps: float = 1e-8
        opt-betas: bool = None
        clip-grad: bool = None
        momentum: float = 0.9
        weight-decay: float = 0.05
        
        ched: str = "cosine"
        lr: float = 0.001
        lr-noise: bool = None
        lr-noise-pct: float = 0.67
        lr-noise-std: float = 1.0
        warmup-lr: float = 1e-06
        min-lr: float = 1e-05
        layer_decay: float = 1.0
        decay-epochs: int = 30
        warmup-epochs: int = 5
        cooldown-epochs: int = 10
        patience-epochs: int = 10
        decay-rate: float = 0.1
        dr: float = 0.1
        
        color-jitter: float = 0.4
        aa: str = "rand-m9-mstd0.5-incl"
        smoothing: float = 0.1
        train-interpolation: str = "bicubic"
        repeated-aug: bool = True
        
        reprob: float = 0.25
        remode: str = "pixel"
        recount: int = 1
        resplit: bool = False
        
        mixup: float = 0.8
        cutmix: float = 1.0
        cutmix-minmax: bool = None
        mixup-prob: float = 1.0
        mixup-switch-prob: float = 0.5
        mixup-mode: str = "batch"
        
        teacher-model: str = "self"
        teacher-path: str = ""
        distillation-type: str = "none"
        distillation-alpha: float = 0.5
        distillation-tau: float = 1.0
        
        finetune: str = ""
        disable_weight_decay_on_cls_pos: bool = False
        disable_weight_decay_on_bias_norm: bool = False
        disable_weight_decay_on_rel_pos_bias: bool = False
        init_scale: float = 1.0
        layer_scale_init_value: int = 0
        
        data_path: str
        
        data_set: str = "IMNET"
        inat-category: str = "name"
        output_dir: str
        device: str = "cuda"
        seed: int = 0
        resume: str =  ""
        start_epoch: int = 0
        eval: bool = True
        num_workers: int = 10
        pin-mem: bool = True
        
        local_rank: int = 0
        dist_url: str = "env://"
        
        finetune_head_layer: int = 0
        out_dim: int = 1000
        
        log_dir: str = "./"
        ratio: int = 1 

class SchemeOfSmSLParams(BaseModel):
    params: SchemaOfSmSLTableRequest

@app.post('/api/clustering', response_model=SchemaOfTableResponse)
def getClusteringTable():
    params = {
        "pretrained_weights": "/weights/ibot_small_pretrain.pth",
        "data_path": "/dataset"
    }
    
    # convert to dict
    clusteringTable = main_eval(**params)
    return clusteringTable


@app.post('/api/semi-supervised-learning', response_model=SchemaOfTableResponse)
def getSmSLTable():
    params = {
        "pretrained_weights": "/weights/ibot_small_pretrain.pth",
        "data_path": "/dataset"
    }
    params = SchemaOfSmSLTableRequest(**params)
    
    smslTable = main_eval(params)
    return smslTable
