from config.pretrained_model_path import (
    SEGFORMER,
    MASKFORMER,
    MASK2FORMER,
    VISIONTRANSFORMER,
)
from config.utils import find_last_index
from omegaconf import OmegaConf

cfg = OmegaConf.load("./config/model_config.yaml")
WEIGHT = [
    0.07174091612736269,
    0.33986588177825566,
    0.2162811413135844,
    0.01362717262196377,
    0.14729804975888477,
    0.1886382959867921,
    0.0186154332103708,
    4.777205413696111e-06,
    0.0031530449781162016,
    0.0005825182448825697,
    0.00019276877437331393,
]

SCHEDULER = {
    "StepLR": dict(lr=1e-4, step_size=5, gamma=0.95),
    "CLR": dict(max_lr=6e-4, base_lr=2e-4, step_size=25),
    "ONECLR": dict(max_lr=1e-3, pct_start=0.2, div_factor=10),
}

PATH = cfg.PATH

STATS_MEAN = [
    mean * 255
    for mean in [0.3584, 0.3111, 0.2654, 0.2578, 0.3299, 0.3653, 0.3547, 0.2965, 0.2266]
]
STATS_STD = [
    std * 255
    for std in [0.064, 0.072, 0.0095, 0.048, 0.067, 0.080, 0.085, 0.074, 0.060]
]

STATS_MEAN_Sen1Flood = [
    1189.2240032217355,
    1335.7876259041898,
    1369.555378215939,
    1432.4846319273197,
    2329.1874191028446,
    2776.599471278697,
    2559.5920587145415,
    1986.2344462964788,
    1150.1873175658327,
]
STATS_STD_Sen1Flood = [
    504.48473658532214,
    409.57012279339966,
    395.4498976071994,
    459.6583462832101,
    640.6401557563074,
    771.8186296160205,
    743.4712046089181,
    695.772232693198,
    522.1633726674359,
]
mode = "run"
class_of_interest = cfg.MODEL.class_of_interest
customized_weight = True
cuda = True
TASK = "segmentation"
MODEL_TYPE = "dinov2"
MODEL_VERSION = "flood"
idx = find_last_index(f"{MODEL_TYPE}_{MODEL_VERSION}", PATH["model_outdir"])
MODEL_NAME = f"{MODEL_TYPE}_{MODEL_VERSION}_{idx}"
if TASK == "segmentation":
    if MODEL_TYPE == "segformer":
        MODEL_CONFIG = dict(
            model_name=MODEL_NAME,
            model_type=MODEL_TYPE,
            model_version=MODEL_VERSION,
            pretrained_path=SEGFORMER[MODEL_VERSION],
            num_classes=11,
            image_size=512,
        )
    if MODEL_TYPE == "maskformer":
        MODEL_CONFIG = dict(
            model_name=MODEL_NAME,
            model_type=MODEL_TYPE,
            model_version=MODEL_VERSION,
            pretrained_path=MASKFORMER[MODEL_VERSION],
            num_classes=1,
            image_size=512,
        )
    if MODEL_TYPE == "mask2former":
        MODEL_CONFIG = dict(
            model_name=MODEL_NAME,
            model_type=MODEL_TYPE,
            model_version=MODEL_VERSION,
            pretrained_path=MASK2FORMER[MODEL_VERSION],
            num_classes=1,
            image_size=224,
        )
    if MODEL_TYPE == "unet":
        MODEL_CONFIG = dict(
            model_name=MODEL_NAME,
            model_type=MODEL_TYPE,
            model_version=MODEL_VERSION,
            pretrained_path=None,
            num_classes=1,
            image_size=224,
        )
    if MODEL_TYPE == "dinov2":
        MODEL_CONFIG = dict(
            model_name=MODEL_NAME,
            model_type=MODEL_TYPE,
            model_version=MODEL_VERSION,
            pretrained_path=None,
            num_classes=len(class_of_interest) + 1,
            image_size=512,
        )
if TASK == "classification":
    if MODEL_TYPE == "vit":
        MODEL_CONFIG = dict(
            model_name=MODEL_NAME,
            model_type=MODEL_TYPE,
            model_version=MODEL_VERSION,
            pretrained_path=VISIONTRANSFORMER[MODEL_VERSION],
            num_classes=1,
            image_size=224,
        )
HYPERPARAM = dict(
    batch_size=cfg.MODEL.batch_size,
    epochs=cfg.MODEL.num_epoch,
    weight=WEIGHT,
    optimizer=cfg.MODEL.optimizer,
    scheduler=None,
    weight_decay=0,
    model_config=MODEL_CONFIG,
)
