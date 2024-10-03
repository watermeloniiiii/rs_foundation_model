SEGFORMER = {
    "b0": "nvidia/segformer-b0-finetuned-ade-512-512",  # 13677505
    "b1": "nvidia/segformer-b1-finetuned-ade-512-512",  # 27347393
    "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "b3": "nvidia/segformer-b3-finetuned-ade-512-512",  # 47223233
    "b4": "nvidia/segformer-b4-finetuned-ade-512-512",  # 63993793
    "b5": "nvidia/segformer-b5-finetuned-ade-640-640",
}

MASKFORMER = {
    "swin_tiny_ade": "facebook/maskformer-swin-tiny-ade",  # 41720252
    "swin_tiny_coco": "facebook/maskformer-swin-tiny-coco",  # 41720252
    "satellite": "thiagohersan/maskformer-satellite-trees",
    "swin_small_ade": "facebook/maskformer-swin-small-ade",  # 63038413
    "swin_base_ade": "facebook/maskformer-swin-base-ade",  # 101793403
    "swin_large_ade": "facebook/maskformer-swin-large-ade",  # 211540663
}

MASK2FORMER = {
    "swin_large_cs": "facebook/mask2former-swin-large-cityscapes-semantic",
    "swin_large_ade": "facebook/mask2former-swin-large-ade-semantic",
    "swin_tiny_ade": "facebook/mask2former-swin-tiny-ade-semantic",  # 47401597
    "swin_base_ade": "facebook/mask2former-swin-base-IN21k-ade-semantic",
}

VISIONTRANSFORMER = {
    "large_16": "google/vit-large-patch16-224",
    "base_16": "google/vit-base-patch16-224",
    "base_32": "google/vit-base-patch32-224-in21k",
}
