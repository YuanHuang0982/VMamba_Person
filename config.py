import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 128
_C.DATA.BATCH_SIZE_test = 8
#_C.DATA.TRAIN_PATH = './data/train'  # train
#_C.DATA.VAL_PATH = './data/val'  # test
#_C.DATA.DATASET = 'market1501'

#_C.DATA.TRAIN_PATH = './Occluded_REID_MarketStyle/train01'  # train
#_C.DATA.VAL_PATH = './Occluded_REID_MarketStyle/val'  # test
#_C.DATA.DATASET = 'Occluded_REID_MarketStyle'

#_C.DATA.TRAIN_PATH = './P_ETHZ_MarketStyle/train'  # train
#_C.DATA.VAL_PATH = './P_ETHZ_MarketStyle/val'  # test
#_C.DATA.DATASET = 'P_ETHZ_MarketStyle'

#_C.DATA.TRAIN_PATH = './occluded_reid1/train'  # train
#_C.DATA.VAL_PATH = './occluded_reid1/val'  # test
#_C.DATA.DATASET = 'occluded_reid1'

_C.DATA.TRAIN_PATH = './occluded_split1/train'  # train
_C.DATA.VAL_PATH = './occluded_split1/val'  # val
_C.DATA.TEST_PATH = './occluded_split1/test'  # test
_C.DATA.DATASET = 'occluded_reid_split1'


_C.DATA.IMG_SIZE = 224
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.ZIP_MODE = False
_C.DATA.CACHE_MODE = 'part'
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 4
_C.DATA.MASK_PATCH_SIZE = 32
_C.DATA.MASK_RATIO = 0.6

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'vssm'
_C.MODEL.NAME = 'vmamba_reid'
_C.MODEL.PRETRAINED = ''
_C.MODEL.RESUME = ''
#_C.MODEL.NUM_CLASSES = 751
#_C.MODEL.NUM_CLASSES = 160
#_C.MODEL.NUM_CLASSES = 68
#_C.MODEL.NUM_CLASSES = 140
_C.MODEL.NUM_CLASSES = 120

_C.MODEL.DROP_RATE = 0.0
_C.MODEL.DROP_PATH_RATE = 0.2
_C.MODEL.LABEL_SMOOTHING = 0.0
_C.MODEL.MMCKPT = False
_C.MODEL.USE_EMA = True



_C.MODEL.TOKEN_MERGE = True     # ✅ 병합 기능을 켤지 여부 (기본 True)
_C.MODEL.FUSION_TOKEN = 48
_C.MODEL.MERGE_STRATEGY = "upper" # all/upper/lower/alternate
_C.MODEL.MERGE_LAYER = 6



_C.MODEL.VSSM = CN()
_C.MODEL.VSSM.EMBED_DIM = 96
_C.MODEL.VSSM.DEPTHS = [2, 2, 8, 2]
_C.MODEL.VSSM.SSM_D_STATE = 1
_C.MODEL.VSSM.SSM_DT_RANK = "auto"
_C.MODEL.VSSM.SSM_RATIO = 1.0
_C.MODEL.VSSM.SSM_CONV = 3
_C.MODEL.VSSM.SSM_CONV_BIAS = False
_C.MODEL.VSSM.SSM_FORWARDTYPE = "v05_noz"
_C.MODEL.VSSM.MLP_RATIO = 4.0
_C.MODEL.VSSM.DOWNSAMPLE = "v3"
_C.MODEL.VSSM.PATCHEMBED = "v2"
_C.MODEL.VSSM.NORM_LAYER = "ln2d"

_C.MODEL.VSSM.PATCH_SIZE = 4
_C.MODEL.VSSM.IN_CHANS = 3
_C.MODEL.VSSM.SSM_RANK_RATIO = 2.0
_C.MODEL.VSSM.SSM_ACT_LAYER = "silu"
_C.MODEL.VSSM.SSM_DROP_RATE = 0.0
_C.MODEL.VSSM.SSM_INIT = "v0"
_C.MODEL.VSSM.MLP_ACT_LAYER = "gelu"
_C.MODEL.VSSM.MLP_DROP_RATE = 0.0
_C.MODEL.VSSM.PATCH_NORM = True
_C.MODEL.VSSM.POSEMBED = False
_C.MODEL.VSSM.GMLP = False
_C.MODEL.VSSM.BACKEND = "oflex"

_C.MODEL.VSSM.VISUALIZE = False

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN() 
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 8      #3
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-5     #3e-4
_C.TRAIN.WARMUP_LR = 1e-6
_C.TRAIN.MIN_LR = 1e-7      #1e-6
_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.ACCUMULATION_STEPS = 2    #1
_C.TRAIN.USE_CHECKPOINT = False

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.LAYER_DECAY = 1.0

_C.TRAIN.MOE = CN()
_C.TRAIN.MOE.SAVE_MASTER = False

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.COLOR_JITTER = 0.35      #0.4
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
_C.AUG.REPROB = 0.4           #0.25
_C.AUG.REMODE = 'pixel'
_C.AUG.RECOUNT = 1
_C.AUG.MIXUP = 0.2
_C.AUG.CUTMIX = 0.2              #0.0
_C.AUG.CUTMIX_MINMAX = (0.2,0.8)           #None
_C.AUG.MIXUP_PROB = 1.0
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.CROP = True
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.ENABLE_AMP = True
_C.AMP_ENABLE = True
_C.AMP_OPT_LEVEL = ''
_C.OUTPUT = './output/oM486val6tvterase4'     # 모델과 결과 저장 위치
_C.TAG = 'baseline'        # 실험 이름
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 10
_C.SEED = 42
_C.EVAL_MODE = False
_C.THROUGHPUT_MODE = False
_C.TRAINCOST_MODE = False
_C.FUSED_LAYERNORM = False

_C.LOSS = CN()
_C.LOSS.MARGIN = 0.3

_C.DISABLE_MIXUP_EPOCH = 10  # Epoch after which mixup is disabled
# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), cfg))
    print('=> Merged config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    if args.cfg:
        _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # OUTPUT 경로 자동 완성
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    config.freeze()

def get_config(args):
    config = _C.clone()
    update_config(config, args)
    return config
