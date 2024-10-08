from models.dinov2_model import DINOV2PretrainedModel
import torch
import torch.nn as nn
from transformers.modeling_outputs import SemanticSegmenterOutput


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_class=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_class, (1, 1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.classifier(embeddings)


class Dinov2ForSemanticSegmentation(nn.Module):
    def __init__(self, cfg, img_size=512, patch_size=16, num_class=1):
        super().__init__()

        self.dinov2 = DINOV2PretrainedModel(cfg)
        self.config = self.dinov2.config
        self.num_class = num_class
        self.hidden_size = self.dinov2.model.patch_embed.proj.out_channels
        self.patch_size = self.dinov2.config.student.patch_size
        self.num_register = self.dinov2.config.student.num_register_tokens
        self.height = self.width = img_size // patch_size
        self.feature_fusion = nn.Linear(self.hidden_size * 4, self.hidden_size)
        self.classifier = LinearClassifier(
            self.hidden_size, self.width, self.height, num_class=num_class
        )

    def forward(self, pixel_values, labels=None, doy=None):
        # use frozen features
        outputs = self.dinov2.model.get_intermediate_layers(pixel_values, 4, doy=doy)
        # outputs = self.dinov2(pixel_values)
        # get the patch embeddings - so we exclude the CLS token
        # patch_embeddings = outputs["x_norm_patchtokens"]
        patch_embeddings = torch.concatenate(outputs, dim=2)

        # convert to logits and upsample to the size of the pixel values
        fused_embeddings = self.feature_fusion(patch_embeddings)
        logits = self.classifier(fused_embeddings)
        logits = torch.nn.functional.interpolate(
            logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
        )

        loss = None
        if labels is not None:
            # important: we're going to use 0 here as ignore index instead of the default -100
            # as we don't want the model to learn to predict background
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            if labels.dtype != torch.cuda.LongTensor:
                labels = labels.type(torch.cuda.LongTensor)
            # labels = labels[:, None, :, :] if len(labels.shape) == 3 else labels
            logits = logits.squeeze() if self.num_class != 1 else logits
            loss = loss_fct(logits, labels)

        return SemanticSegmenterOutput(loss=loss, logits=logits)
