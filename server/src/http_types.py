from pydantic import BaseModel

class SchemaOfInputDataPathRequest(BaseModel):
    arch: str
    data_path: str

class SchemaOfTableResponse(BaseModel):
    body: dict
    
class SchemaOfSmSLwithiBOTRequest(BaseModel):
    batch_size: int = 128
    epochs: int = 400
    distill_epochs: int = 150
    
    avgpool_patchtokens: int = 0
    arch: str = "vit_small"
    patch_size: int = 16
    window_size: int = 7
    input_size: int = 224
    pretrained_weights: str
    checkpoint_key: str = "student"
    drop: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path: float = 0.1
    
    model_ema: bool = True
    model_ema_decay: float = 0.99996
    model_ema_force_cpu: bool = False
    
    opt: str = "adamw"
    opt_eps: float = 1e-8
    opt_betas: bool = None
    clip_grad: bool = None
    momentum: float = 0.9
    weight_decay: float = 0.05
    
    ched: str = "cosine"
    lr: float = 0.001
    lr_noise: bool = None
    lr_noise_pct: float = 0.67
    lr_noise_std: float = 1.0
    warmup_lr: float = 1e-06
    min_lr: float = 1e-05
    layer_decay: float = 1.0
    decay_epochs: int = 30
    warmup_epochs: int = 5
    cooldown_epochs: int = 10
    patience_epochs: int = 10
    decay_rate: float = 0.1
    dr: float = 0.1
    
    color_jitter: float = 0.4
    aa: str = "rand-m9-mstd0.5-incl"
    smoothing: float = 0.1
    train_interpolation: str = "bicubic"
    repeated_aug: bool = True
    
    reprob: float = 0.25
    remode: str = "pixel"
    recount: int = 1
    resplit: bool = False
    
    mixup: float = 0.8
    cutmix: float = 1.0
    cutmix_minmax: bool = None
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = "batch"
    
    teacher_model: str = "self"
    teacher_path: str = ""
    distillation_type: str = "none"
    distillation_alpha: float = 0.5
    distillation_tau: float = 1.0
    
    finetune: str = ""
    disable_weight_decay_on_cls_pos: bool = False
    disable_weight_decay_on_bias_norm: bool = False
    disable_weight_decay_on_rel_pos_bias: bool = False
    init_scale: float = 1.0
    layer_scale_init_value: int = 0
    
    data_path: str
    
    data_set: str = "IMNET"
    inat_category: str = "name"
    output_dir: str
    device: str = "cuda"
    seed: int = 0
    resume: str =  ""
    start_epoch: int = 0
    eval: bool = True
    num_workers: int = 10
    pin_mem: bool = True
    
    local_rank: int = 0
    dist_url: str = "env://"
    
    finetune_head_layer: int = 0
    out_dim: int = 1000
    
    log_dir: str = "./"
    ratio: int = 1 

class SchemeOfSmSLwithiBOTParams(BaseModel):
    params: SchemaOfSmSLwithiBOTRequest

class SchemaOfSmSLwithSwAVRequest(BaseModel):
    labels_perc: str = "10"
    dump_path: str
    seed: int = 31
    data_path: str
    workers: int = 10
    
    arch: str = "resnet50"
    pretrained: str
    
    epochs: int = 20
    batch_size: int = 32
    lr: float = 0.01
    lr_last_layer: float = 0.2
    decay_epochs: int = [12, 16]
    gamma: float = 0.2
    
    dist_url: str = "env://"
    world_size: int = -1
    rank: int = 0
    local_rank: int = 0    