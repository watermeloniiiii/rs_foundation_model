from omegaconf import OmegaConf

SCHEDULER = {
    "StepLR": dict(lr=1e-4, step_size=5, gamma=0.95),
    "CLR": dict(max_lr=6e-4, base_lr=2e-4, step_size=25),
    "ONECLR": dict(max_lr=1e-3, pct_start=0.2, div_factor=10),
}


def default_setup(config):
    cfg = OmegaConf.load(config)
    task = cfg.PROJECT.task
    model_idx = cfg.PROJECT.model_idx
    model_name = f"{task}_"
    model_name += model_idx
    model_info = dict(
        MODEL_INFO=dict(
            model_name=model_name,
        )
    )
    cfg = OmegaConf.merge(cfg, model_info)
    return cfg
