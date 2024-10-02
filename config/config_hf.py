from pretrained_model_path import SEGFORMER, MASKFORMER, MASK2FORMER, VISIONTRANSFORMER
from utils import find_last_index

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

PATH = {
    "data_dir": "/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas",
    "output_dir": "/NAS6/Members/linchenxi/projects/RS_foundation_model/model",
    "model_dir": "/NAS6/Members/linchenxi/projects/RS_foundation_model/model",
}

STATS_MEAN = [
    mean * 255
    for mean in [0.3584, 0.3111, 0.2654, 0.2578, 0.3299, 0.3653, 0.3547, 0.2965, 0.2266]
]
STATS_STD = [
    std * 255
    for std in [0.064, 0.072, 0.0095, 0.048, 0.067, 0.080, 0.085, 0.074, 0.060]
]

mode = "run"
customized_weight = True
encoder_weights = True
cuda = True
TASK = "segmentation"
MODEL_TYPE = "dinov2"
MODEL_VERSION = "linear"
idx = find_last_index(f"{MODEL_TYPE}_{MODEL_VERSION}", PATH["output_dir"])
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
            num_classes=1,
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
    batch_size=8,
    epochs=500,
    weight=WEIGHT,
    optimizer="AdamW",
    scheduler=None,
    weight_decay=0,
    model_config=MODEL_CONFIG,
)
