root_dir = r"F:\DigitalAG\morocco\unet"
train_dir = r"F:\DigitalAG\morocco\unet\baseline\32\training\img"
vali_dir = r"F:\DigitalAG\morocco\unet\baseline\32\validation\img"
model_dir = r"F:\DigitalAG\morocco\unet\baseline\32\model"

viz_param = {
    "CIG": {"ylim": (0, 4.6), "yticks": [0, 1, 2, 3, 4]},
    "NDVI": {"ylim": (0, 1), "yticks": [0, 0.2, 0.4, 0.6, 0.8]},
    "EVI": {"ylim": (-0.2, 0.6), "yticks": [-0.2, 0, 0.2, 0.4, 0.6]},
    "LSWI": {"ylim": (-0.4, 0.4), "yticks": [-0.4, -0.2, 0, 0.2, 0.4]},
    "OSAVI": {"ylim": (0, 0.6), "yticks": [0, 0.15, 0.3, 0.45, 0.6]},
}

cuda = True
mode = "shallow"
if mode == "vit":
    general = dict(
        mode="vit",
        optimizer="Adam_StepLR",
        samples="NA",
        # model_index='finetune_200_2',
        model_index="vit_6",
        info="take average after sigmoid/ndvi/cnn with batchnorm and relu",
    )
    hyperparameters = dict(
        bands=3,
        batch_size=16,
        epochs=400,
        SGD_StepLR=dict(lr=1e-4, step_size=20, gamma=0.8, momentum=0.9),
        SGD_CLR=dict(lr=1e-4, max_lr=7e-6, base_lr=5e-7, step_size=20, momentum=0),
        Adam_StepLR=dict(lr=1e-4, step_size=10, gamma=0.5),
        weight_decay=1,
        weight=[1, 4],
        pool="mean",
        image_size=120,
        patch_size=30,
        num_classes=1,
        dim=64,
        depth=5,
        heads=5,
        mlp_dim=1024,
        dropout=0,
        emb_dropout=0,
    )
if mode == "unet":
    general = dict(mode="unet", optimizer="SGD_CLR", model_index="unet_1", info="NA")
    hyperparameters = dict(
        bands=3,
        batch_size=16,
        epochs=200,
        SGD_StepLR=dict(lr=1e-4, step_size=5, gamma=0.95, momentum=0),
        SGD_CLR=dict(lr=1e-4, max_lr=5e-3, base_lr=1e-3, step_size=15, momentum=0.9),
        Adam_StepLR=dict(lr=1e-5, step_size=10, gamma=0.95),
        hidden_layer=[16, 32, 64, 128],
        weight_decay=0,
        weight=[1, 5],
        dropout=0.3,
    )
if mode == "shallow":
    general = dict(
        mode="shallow", optimizer="SGD_CLR", model_index="shallow_32_1", info="NA"
    )
    hyperparameters = dict(
        bands=3,
        batch_size=16,
        epochs=100,
        SGD_StepLR=dict(lr=1e-4, step_size=5, gamma=0.95, momentum=0),
        SGD_CLR=dict(lr=1e-4, max_lr=1e-4, base_lr=7e-5, step_size=15, momentum=0.9),
        Adam_StepLR=dict(lr=1e-5, step_size=10, gamma=0.95),
        hidden_layer=[16, 32, 64, 128],
        fc_layer=[128, 128, 256],
        weight_decay=0.1,
        weight=[1, 5],
        dropout=0.3,
    )
if mode == "pretrain":
    general = dict(
        mode="pretrain",
        optimizer="SGD_CLR",
        model_index="pretrain_base_16",
        info="1.no adversarial learning, directly take the prob of all sub-patches from ViT\t2.freeze all layers but the last one",
    )
    hyperparameters = dict(
        bands=3,
        batch_size=8,
        epochs=200,
        SGD_StepLR=dict(lr=1e-4, step_size=5, gamma=0.95, momentum=0),
        SGD_CLR=dict(lr=1e-4, max_lr=5e-5, base_lr=2e-5, step_size=15, momentum=0.9),
        Adam_StepLR=dict(lr=1e-4, step_size=20, gamma=0.8),
        weight=[1, 3],
        sub_weight=[1, 1],
        weight_decay=0.6,
        threshold=[0.5, 0.1],
        regularization=60,
    )
