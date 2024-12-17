from omegaconf import OmegaConf

SCHEDULER = {
    "StepLR": dict(lr=1e-4, step_size=5, gamma=0.95),
    "CLR": dict(max_lr=6e-4, base_lr=2e-4, step_size=25),
    "ONECLR": dict(max_lr=1e-3, pct_start=0.2, div_factor=10),
}


def default_setup(config):
    cfg = OmegaConf.load(config)
    task = cfg.PROJECT.task
    pretrain = cfg.PROJECT.pretrain
    downstream = cfg.PROJECT.downstream
    model_idx = cfg.PROJECT.model_idx
    model_name = f"{task}_"
    model_name += f"{pretrain}_" if pretrain else "np_"  # np denotes no pretraining
    if downstream:
        model_name += downstream + "_"
    model_name += model_idx
    if task == "pretrain":
        class_of_interest = cfg.MODEL.class_of_interest.pretrain_class
        class_weight = cfg.MODEL.class_of_interest.pretrain_class_weight
        pretrain_weight = cfg.PRETRAIN.weights_dino_pretrain
    if task == "finetune":
        class_of_interest = cfg.MODEL.class_of_interest.finetune_class
        class_weight = cfg.MODEL.class_of_interest.finetune_class_weight
        pretrain_weight = cfg.PRETRAIN.weights_satlas_pretrain
    model_info = dict(
        MODEL_INFO=dict(
            model_name=model_name,
            class_of_interest=class_of_interest,
            pretrained_weight=pretrain_weight,
            class_weight=class_weight,
        )
    )
    cfg = OmegaConf.merge(cfg, model_info)
    return cfg
