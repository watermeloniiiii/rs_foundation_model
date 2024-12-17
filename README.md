## <span style=color:#4DBBD5;font-size:15px;font-weight:bold>Overview</span>
This repo implements the continued-pretraining and finetune based on the DINOv2 pretrained foundation model. Three different tasks are supported:

`satlas_multiclass`: perform continued-pretraining to construct a powerful projection head. The logic is that the DINOv2 pretrained model provides the backbone to extract image features whereas the target downstream task (e.g., the flood prediction) may not have enough data to finetune the projection head. Therefore, we can leverage a larger dataset to pretrain the projection head (you can choose to freeze the backbone or not, see the following code snippet). We use the Satlas dataset here (as well as its pixel-level labels) and defined 5 classes (background, water, developed, vegetation and cropland) to train the projection head. 
```python
  if config.MODEL.architecture.freeze_backbone:
      for param in model.parameters():
          param.requires_grad = False
      '''
      - feature_fusion is to fuse the feature from last N encoder layers
      - classifier is the projection head
      - please refer to models/customized_segmention_model.py for more details
      '''
      for module in [model.feature_fusion, model.classifier]:
          for param in module.parameters():
              param.requires_grad = True
```

`flood_prediction`: perform the finetune on the downstream task. One major difference from `satlas_multiclass` is that the we may need to load the weights twice. In `satlas_multiclass`, when you initialize the DINOv2 model, the DINOv2 pretrained weights will be loaded. However, in `flood_prediction`, if the projection head has already be pretrained, we will need to load the new weights again to overwrite the DINOv2 pretrained weights. Beside, as the number of class could be different between the continued-pretraining dataset and finetune dataset (e.g, Satlas vs. Sen1Floods11), the last layer of the projection head need to be adjusted.        
```python
  if config.PROJECT.task == "finetune":
      assert (
          config.PRETRAIN.weights_satlas_pretrain
      ), "please provide pretrained weights"
      state_dict = get_fp32_state_dict_from_zero_checkpoint(
          config.PRETRAIN.weights_satlas_pretrain, tag=""
      )
      model.load_state_dict(state_dict, strict=False)
      model.classifier.classifier = nn.utils.weight_norm(
          nn.Conv2d(model.classifier.bottleneck_dim, 2, 1)
      )
      model.classifier.classifier.weight_g.data.fill_(1)
      for param in model.parameters():
          param.requires_grad = False
      for param in model.classifier.parameters():
          param.requires_grad = True
      model.classifier.classifier.weight_g.requires_grad = False
```
## <span style=color:#4DBBD5;font-size:15px;font-weight:bold>Code Structure</span>  
```text
📦config
 ┣ 📜accelerate_deepspeed_config.yaml   # the config file for integrating deepspeed and accelerate
 ┣ 📜config_deepspeed.json              # the config file for deepspeed
 ┣ 📜continued_pretraining.yaml         # the config file for continue-pretrainig (monomodal)
 ┣ 📜finetune.yaml                      # the config file for downstream finetune
 ┣ 📜multimodal_test.yaml               # the config file for continue-pretrainig (multimodal)
 ┗ 📜setup.py                           # parse the config file
📦data
 ┣ 📜satlas.py                          # the dataset class for Satlas
 ┣ 📜sen12ms.py                         # the dataset class for Sen12MS-CR-TS
 ┗ 📜sen1floods11.py                    # the dataset class for Sen1Floods11
📦main
 ┣ 📂bash
 ┃ ┣ 📜flood_prediction.sh              # bash file to run finetune (flood prediction)
 ┃ ┣ 📜multimodal_test.sh               # bash file to run continue-pretraining (multimodal)
 ┃ ┣ 📜satlas_multiclass.sh             # bash file to run continue-pretraining (monomodal)
 ┃ ┗ 📜submit2slurm.sh                  # submit the job to slurm
 ┣ 📜flood_prediction.py                # run finetune (flood prediction)
 ┣ 📜inference.py                       # inference on flood prediction
 ┣ 📜multimodal_test.py                 # run continue-pretraining (multimodal)
 ┣ 📜satlas_multiclass.py               # run continue-pretraining (monomodal)
 ┗ 📜trainer_deepspeed.py               # run training/validation
📦models
 ┣ 📜customized_segmention_model.py     # costumized model for segmentation task
 ┗ 📜dinov2_model.py                    # dinov2 model
 ```
</span>

---

## <span style=color:#4DBBD5;font-size:15px;font-weight:bold>Start with the config file</span>  
<span style=font-size:13px;color:#00A087>

Basically the config files for continued-pretraining and finetune are the same. Here are few differens you might need to pay attention to.
1. During finetune, `task` should be set to `finetune` (otherwise `pretrain`), some parameters are different for this two mode
```python
# for finetune
PROJECT:
  task: "finetune" # either "finetune" or "pretrain" 
  pretrain: "dinov2" # the name to denote where the pretrained model comes from
  downstream: null # the name to denote what downstream task is doing now
  model_idx: "8"
  description: "finetune with Sen1Floods11 data"

# for continued pretraining
PROJECT:
  task: "pretrain" # either "finetune" or "pretrain" 
  pretrain: "dinov2"
  downstream: null
  model_idx: "12" 
  description: "pretraining with satlas multiple classes from scratch, freeze the backbone and use mimic_DINOHEAD"
```
2. `class_of_interest`
Note that this part was completed for continued-pretraining with Satlas and finetune with Sen1Floods11, whereas for multimodal model, as we haven't figured out what dataset to use, we used dummy "class_of_interest" where the target was defined to have random values ranging from 0 to 5 but with no physical meaning. 
```python
MODEL:
  class_of_interest:
    pretrain_class: # for satlas dataset, we combined its label into four categories, and by default, there will be a background category
      - water
      - developed
      -             # pixels labelled as "tree", "shrub" and "grass" will be grouped 
        - tree
        - shrub
        - grass
      - crop
    pretrain_class_weight: []   # one can set customized weights for the class of interest
    finetune_class:
      - water
    finetune_class_weight:
      - 0.1
      - 0.5
```
3. Model checkpoints.
For simplicity, we include all checkpoints in the same config file, no matter it will be used or not. 
`cfg_dino` denotes the pretrained DINOv2 model, this checkpoint can either be used for continued pretraining or be directly used for finetune
```python
PRETRAIN:
  cfg_dino: "/NAS3/Members/linchenxi/projects/DINOV2/model3/config.yaml"
  cfg_satlas: "/NAS3/Members/linchenxi/projects/foundation_model/model/pretrain_dinov2_12/config.yaml"
  weights_dino_pretrain: "/NAS3/Members/linchenxi/projects/DINOV2/model3/eval/training_12499/teacher_checkpoint.pth"
  weights_satlas_pretrain: "/NAS3/Members/linchenxi/projects/foundation_model/model/pretrain_dinov2_12/best"
```
---
## <span style=color:#4DBBD5;font-size:15px;font-weight:bold>Run the code</span>  
<span style=font-size:13px;color:#00A087>